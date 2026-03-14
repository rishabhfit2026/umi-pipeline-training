"""
Diffusion Policy Trainer for Robot Manipulation
Input  : camera0_rgb  (224,224,3) uint8
Output : eef_pos(3) + eef_rot_axis_angle(3) + gripper_width(1) = 7 actions

Architecture : ResNet18 vision encoder → 1D UNet denoiser (Diffusion Policy)
Action chunk  : predict 16 future actions at once (action chunking)
Obs horizon   : use 2 past frames as context

conda activate maniskill2
pip install diffusers einops timm --break-system-packages
python train_diffusion_policy.py
"""

import os, math, time, numpy as np, zarr, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import timm

# ── Config ────────────────────────────────────────────────────────
ZARR_PATH       = '/home/rishabh/Downloads/umi-pipeline-training/outputs/sim_replay_buffer3.zarr'
SAVE_DIR        = './checkpoints_diffusion'
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset
OBS_HORIZON     = 2       # use last 2 frames as input context
ACTION_HORIZON  = 16      # predict 16 future actions
ACTION_DIM      = 7       # pos(3)+rot(3)+grip(1)
IMG_SIZE        = 224

# Training
BATCH_SIZE      = 64
LR              = 1e-4
EPOCHS          = 100
SAVE_EVERY      = 10
NUM_WORKERS     = 4

# Diffusion
NUM_DIFFUSION_STEPS = 100   # training noise steps
NUM_INFERENCE_STEPS = 16    # DDIM inference steps (fast)

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Device: {DEVICE}")

# ── Normalizer ────────────────────────────────────────────────────
class ActionNormalizer:
    """Normalize actions to [-1, 1] for stable diffusion training"""
    def __init__(self, actions):
        self.min = torch.tensor(actions.min(0), dtype=torch.float32)
        self.max = torch.tensor(actions.max(0), dtype=torch.float32)
        self.scale = self.max - self.min
        self.scale[self.scale < 1e-6] = 1.0  # avoid div by zero

    def normalize(self, x):
        return 2.0 * (x - self.min.to(x.device)) / self.scale.to(x.device) - 1.0

    def denormalize(self, x):
        return (x + 1.0) / 2.0 * self.scale.to(x.device) + self.min.to(x.device)

    def save(self, path):
        torch.save({'min': self.min, 'max': self.max}, path)

    @classmethod
    def load(cls, path):
        n = cls.__new__(cls)
        d = torch.load(path, map_location='cpu')
        n.min, n.max = d['min'], d['max']
        n.scale = n.max - n.min
        n.scale[n.scale < 1e-6] = 1.0
        return n

# ── Dataset ───────────────────────────────────────────────────────
class RobotDataset(Dataset):
    def __init__(self, zarr_path, obs_horizon, action_horizon, normalizer):
        z = zarr.open(zarr_path, 'r')
        self.images  = z['data']['camera0_rgb'][:]          # (N,224,224,3) uint8
        pos          = z['data']['robot0_eef_pos'][:]        # (N,3)
        rot          = z['data']['robot0_eef_rot_axis_angle'][:]  # (N,3)
        grip         = z['data']['robot0_gripper_width'][:]  # (N,1)
        ends         = z['meta']['episode_ends'][:]
        starts       = np.concatenate([[0], ends[:-1]])

        self.actions = np.concatenate([pos, rot, grip], axis=-1).astype(np.float32)  # (N,7)
        self.normalizer = normalizer
        self.obs_h   = obs_horizon
        self.act_h   = action_horizon

        # Build valid sample indices (must have full obs + action window inside episode)
        self.indices = []
        for s, e in zip(starts, ends):
            for t in range(s + obs_horizon - 1, e - action_horizon):
                self.indices.append(t)
        print(f"Dataset: {len(self.indices)} samples from {len(ends)} episodes")

        self.transform = transforms.Compose([
            transforms.ToTensor(),               # (3,H,W) float [0,1]
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std =[0.229,0.224,0.225]),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]

        # Observation: last obs_horizon frames stacked → (obs_h*3, H, W)
        imgs = []
        for i in range(self.obs_h):
            frame = self.images[t - self.obs_h + 1 + i]  # (H,W,3) uint8
            imgs.append(self.transform(frame))
        obs_img = torch.cat(imgs, dim=0)  # (obs_h*3, 224, 224)

        # Action chunk: next action_horizon steps
        act = torch.tensor(self.actions[t : t + self.act_h])  # (act_h, 7)
        act = self.normalizer.normalize(act)

        return obs_img, act

# ── Vision Encoder ────────────────────────────────────────────────
class VisionEncoder(nn.Module):
    """ResNet18 with pretrained weights, outputs 512-dim feature per frame"""
    def __init__(self, obs_horizon, freeze_bn=True):
        super().__init__()
        resnet = timm.create_model('resnet18', pretrained=True,
                                   num_classes=0, global_pool='avg')
        self.backbone = resnet
        self.obs_h    = obs_horizon
        feat_dim      = 512
        # Project stacked features → single embedding
        self.proj = nn.Sequential(
            nn.Linear(feat_dim * obs_horizon, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        if freeze_bn:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters(): p.requires_grad = False

    def forward(self, x):
        # x: (B, obs_h*3, H, W)
        B = x.shape[0]
        feats = []
        for i in range(self.obs_h):
            frame = x[:, i*3:(i+1)*3, :, :]  # (B, 3, H, W)
            feats.append(self.backbone(frame)) # (B, 512)
        feats = torch.cat(feats, dim=-1)       # (B, 512*obs_h)
        return self.proj(feats)                # (B, 512)

# ── 1D UNet Denoiser ──────────────────────────────────────────────
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv1   = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2   = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.cond_fc = nn.Linear(cond_dim, out_ch * 2)
        self.norm1   = nn.GroupNorm(8, out_ch)
        self.norm2   = nn.GroupNorm(8, out_ch)
        self.res     = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        h = F.mish(self.norm1(self.conv1(x)))
        # FiLM conditioning
        scale, bias = self.cond_fc(cond).chunk(2, dim=-1)
        h = h * (scale[:, :, None] + 1) + bias[:, :, None]
        h = F.mish(self.norm2(self.conv2(h)))
        return h + self.res(x)

class DiffusionUNet1D(nn.Module):
    """
    1D UNet that denoises action sequences
    Input  : noisy_action (B, act_h, act_dim) + timestep + visual_cond
    Output : predicted noise (B, act_h, act_dim)
    """
    def __init__(self, action_dim, action_horizon, cond_dim=512,
                 dims=[256, 512, 1024]):
        super().__init__()
        self.act_dim = action_dim
        self.act_h   = action_horizon

        # Time embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(128),
            nn.Linear(128, 512),
            nn.Mish(),
            nn.Linear(512, 512)
        )

        # Combine time + visual cond
        total_cond = 512 + cond_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(total_cond, 512),
            nn.Mish(),
            nn.Linear(512, 512)
        )

        # Input projection
        self.in_proj = nn.Conv1d(action_dim, dims[0], 1)

        # Encoder
        self.enc0 = ResidualBlock1D(dims[0], dims[0], 512)
        self.enc1 = ResidualBlock1D(dims[0], dims[1], 512)
        self.down1 = nn.Conv1d(dims[1], dims[1], 3, stride=2, padding=1)
        self.enc2 = ResidualBlock1D(dims[1], dims[2], 512)
        self.down2 = nn.Conv1d(dims[2], dims[2], 3, stride=2, padding=1)

        # Bottleneck
        self.mid = ResidualBlock1D(dims[2], dims[2], 512)

        # Decoder  (skip connections: e2 has dims[2] ch, e1 has dims[1] ch)
        self.up2   = nn.ConvTranspose1d(dims[2], dims[1], 4, stride=2, padding=1)
        self.dec2  = ResidualBlock1D(dims[1] + dims[2], dims[1], 512)
        self.up1   = nn.ConvTranspose1d(dims[1], dims[0], 4, stride=2, padding=1)
        self.dec1  = ResidualBlock1D(dims[0] + dims[1], dims[0], 512)
        self.dec0  = ResidualBlock1D(dims[0], dims[0], 512)

        # Output
        self.out_proj = nn.Conv1d(dims[0], action_dim, 1)

    def forward(self, noisy_action, timestep, visual_cond):
        # noisy_action: (B, act_h, act_dim) → transpose → (B, act_dim, act_h)
        x = noisy_action.transpose(1, 2)
        B = x.shape[0]

        t_emb  = self.time_emb(timestep)                          # (B, 512)
        cond   = self.cond_proj(torch.cat([t_emb, visual_cond], -1))  # (B, 512)

        x  = self.in_proj(x)           # (B, dims[0], T)
        e0 = self.enc0(x,  cond)
        e1 = self.enc1(e0, cond)
        x  = self.down1(e1)
        e2 = self.enc2(x, cond)
        x  = self.down2(e2)
        x  = self.mid(x, cond)
        x  = self.up2(x)
        # Crop to match skip connection size
        x  = x[:, :, :e2.shape[-1]]
        x  = self.dec2(torch.cat([x, e2], 1), cond)
        x  = self.up1(x)
        x  = x[:, :, :e1.shape[-1]]
        x  = self.dec1(torch.cat([x, e1], 1), cond)
        x  = self.dec0(x, cond)
        x  = self.out_proj(x)          # (B, act_dim, T)
        return x.transpose(1, 2)       # (B, T, act_dim)

# ── Full Model ────────────────────────────────────────────────────
class DiffusionPolicy(nn.Module):
    def __init__(self, obs_horizon, action_horizon, action_dim):
        super().__init__()
        self.encoder = VisionEncoder(obs_horizon)
        self.unet    = DiffusionUNet1D(action_dim, action_horizon, cond_dim=512)

    def forward(self, obs_img, noisy_action, timestep):
        cond = self.encoder(obs_img)
        return self.unet(noisy_action, timestep, cond)

# ── Training ──────────────────────────────────────────────────────
def train():
    # Load data & build normalizer
    print("Loading zarr data for normalizer...")
    z      = zarr.open(ZARR_PATH, 'r')
    pos    = z['data']['robot0_eef_pos'][:]
    rot    = z['data']['robot0_eef_rot_axis_angle'][:]
    grip   = z['data']['robot0_gripper_width'][:]
    all_actions = np.concatenate([pos, rot, grip], axis=-1).astype(np.float32)
    normalizer  = ActionNormalizer(all_actions)
    normalizer.save(os.path.join(SAVE_DIR, 'normalizer.pt'))
    print(f"Action min: {all_actions.min(0)}")
    print(f"Action max: {all_actions.max(0)}")

    # Dataset & loader
    dataset = RobotDataset(ZARR_PATH, OBS_HORIZON, ACTION_HORIZON, normalizer)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    # Model
    model     = DiffusionPolicy(OBS_HORIZON, ACTION_HORIZON, ACTION_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * len(loader))

    # Noise scheduler (DDPM for training)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=NUM_DIFFUSION_STEPS,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params/1e6:.1f}M")
    print(f"Training for {EPOCHS} epochs on {len(dataset)} samples...")
    print(f"Batches per epoch: {len(loader)}\n")

    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for obs_img, actions in loader:
            obs_img = obs_img.to(DEVICE)   # (B, obs_h*3, 224, 224)
            actions = actions.to(DEVICE)   # (B, act_h, 7)

            # Sample random timesteps
            B = actions.shape[0]
            timesteps = torch.randint(0, NUM_DIFFUSION_STEPS, (B,),
                                      device=DEVICE).long()

            # Add noise
            noise = torch.randn_like(actions)
            noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)

            # Predict noise
            pred_noise = model(obs_img, noisy_actions, timesteps)

            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler_lr.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        elapsed  = time.time() - t0

        print(f"Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.5f}  "
              f"lr={scheduler_lr.get_last_lr()[0]:.2e}  time={elapsed:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch, 'loss': avg_loss,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'obs_horizon': OBS_HORIZON,
                'action_horizon': ACTION_HORIZON,
                'action_dim': ACTION_DIM,
            }, os.path.join(SAVE_DIR, 'best_model.pt'))
            print(f"  ✅ Saved best model (loss={best_loss:.5f})")

        if epoch % SAVE_EVERY == 0:
            torch.save({
                'epoch': epoch, 'loss': avg_loss,
                'model_state': model.state_dict(),
            }, os.path.join(SAVE_DIR, f'checkpoint_ep{epoch}.pt'))

    print(f"\nTraining complete! Best loss: {best_loss:.5f}")
    print(f"Checkpoints saved to: {SAVE_DIR}")

if __name__ == '__main__':
    train()