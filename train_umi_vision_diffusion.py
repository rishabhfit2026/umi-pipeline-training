"""
Vision-based Diffusion Policy on real UMI data
Uses camera0_rgb images + robot state for pick-and-place

conda activate maniskill2
python train_umi_vision_diffusion.py
"""

import os, math, time, numpy as np, zarr, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torchvision.models as tvm
import torchvision.transforms as T

ZARR_PATH = '/home/rishabh/Downloads/umi-pipeline-training/replay_buffer.zarr'
SAVE_DIR  = './checkpoints_umi_vision'
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'

OBS_HORIZON    = 2   # fewer obs frames — images are heavy
ACTION_HORIZON = 16
ACTION_DIM     = 7
STATE_DIM      = 7
IMG_FEAT_DIM   = 512   # ResNet18 output
# obs = state(7)*2 + img_feat(512)*2 = 1038
OBS_IN         = STATE_DIM * OBS_HORIZON + IMG_FEAT_DIM * OBS_HORIZON  # 1038

BATCH_SIZE          = 32   # smaller — images are large
LR                  = 1e-4
EPOCHS              = 300
NUM_DIFFUSION_STEPS = 100
IMG_SIZE            = 96   # resize images to 96x96

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Device: {DEVICE}")

# ── Image transform ───────────────────────────────────────────────
img_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ── Normalizer ────────────────────────────────────────────────────
class Normalizer:
    def __init__(self, data):
        self.min   = torch.tensor(data.min(0), dtype=torch.float32)
        self.max   = torch.tensor(data.max(0), dtype=torch.float32)
        self.scale = self.max - self.min
        self.scale[self.scale < 1e-6] = 1.0
    def normalize(self, x):
        return 2.0*(x-self.min.to(x.device))/self.scale.to(x.device)-1.0
    def denormalize(self, x):
        return (x+1.0)/2.0*self.scale.to(x.device)+self.min.to(x.device)
    def save(self, path):
        torch.save({'min':self.min,'max':self.max,'scale':self.scale}, path)
    @classmethod
    def load(cls, path):
        n=cls.__new__(cls); d=torch.load(path,map_location='cpu')
        n.min,n.max,n.scale=d['min'],d['max'],d['scale']; return n

# ── Visual encoder (frozen ResNet18) ─────────────────────────────
class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        # Remove final FC layer — keep up to avgpool
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        # Freeze first 2 layers, finetune rest
        for i, child in enumerate(self.encoder.children()):
            if i < 4:
                for p in child.parameters():
                    p.requires_grad = False
    def forward(self, x):
        # x: (B, 3, H, W)
        feat = self.encoder(x)   # (B, 512, 1, 1)
        return feat.squeeze(-1).squeeze(-1)  # (B, 512)

# ── Dataset ───────────────────────────────────────────────────────
class UMIVisionDataset(Dataset):
    def __init__(self, zarr_path, obs_norm, act_norm):
        z     = zarr.open(zarr_path, 'r')
        self.pos   = z['data']['robot0_eef_pos'][:]
        self.rot   = z['data']['robot0_eef_rot_axis_angle'][:]
        self.grip  = z['data']['robot0_gripper_width'][:]
        self.imgs  = z['data']['camera0_rgb']  # keep as zarr — lazy load
        ends  = z['meta']['episode_ends'][:]
        starts= np.concatenate([[0], ends[:-1]])

        self.states = np.concatenate([self.pos, self.rot, self.grip],
                                      axis=-1).astype(np.float32)
        self.obs_norm = obs_norm
        self.act_norm = act_norm

        # Build sample indices
        self.indices = []
        for ep_i, (s, e) in enumerate(zip(starts, ends)):
            for t in range(int(s)+OBS_HORIZON-1, int(e)-ACTION_HORIZON):
                self.indices.append(t)
        print(f"Vision dataset: {len(self.indices)} samples")

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]

        # State obs (OBS_HORIZON frames)
        state_obs = [self.obs_norm.normalize(
                         torch.tensor(self.states[t-OBS_HORIZON+1+i],
                                      dtype=torch.float32))
                     for i in range(OBS_HORIZON)]
        state_flat = torch.cat(state_obs)   # (STATE_DIM * OBS_HORIZON,)

        # Image obs (OBS_HORIZON frames) — load and transform
        imgs = []
        for i in range(OBS_HORIZON):
            frame_idx = t - OBS_HORIZON + 1 + i
            img = self.imgs[frame_idx]  # (H, W, 3) uint8
            imgs.append(img_transform(img))
        imgs = torch.stack(imgs)  # (OBS_HORIZON, 3, H, W)

        # Action
        act = self.act_norm.normalize(
                  torch.tensor(self.states[t:t+ACTION_HORIZON],
                               dtype=torch.float32))
        return state_flat, imgs, act

# ── Diffusion model with visual conditioning ──────────────────────
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim=dim
    def forward(self, t):
        half=self.dim//2
        emb=math.log(10000)/(half-1)
        emb=torch.exp(torch.arange(half,device=t.device)*-emb)
        emb=t.float()[:,None]*emb[None,:]
        return torch.cat([emb.sin(),emb.cos()],dim=-1)

class ResBlock(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.net  = nn.Sequential(nn.Linear(dim,dim),nn.Mish(),nn.Linear(dim,dim))
        self.cond = nn.Linear(cond_dim,dim*2)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, cond):
        scale,bias=self.cond(cond).chunk(2,dim=-1)
        return x+self.net(self.norm(x)*(scale+1)+bias)

class VisionDiffusionNet(nn.Module):
    def __init__(self, hidden=512, depth=8):
        super().__init__()
        flat_act = ACTION_DIM * ACTION_HORIZON
        cond_dim = 512

        # Visual encoder (shared across obs horizon frames)
        self.visual_enc = VisualEncoder()

        # Fuse state + image features
        # state_flat = STATE_DIM*OBS_HORIZON = 14
        # img_feats  = IMG_FEAT_DIM*OBS_HORIZON = 1024
        fuse_in = STATE_DIM*OBS_HORIZON + IMG_FEAT_DIM*OBS_HORIZON
        self.obs_fuse = nn.Sequential(
            nn.Linear(fuse_in, 512), nn.Mish(),
            nn.Linear(512, 512),     nn.Mish(),
            nn.Linear(512, 256))

        self.time_emb  = nn.Sequential(
            SinusoidalPosEmb(128), nn.Linear(128,256),
            nn.Mish(), nn.Linear(256,256))
        self.cond_proj = nn.Sequential(
            nn.Linear(512,cond_dim), nn.Mish(),
            nn.Linear(cond_dim,cond_dim))
        self.in_proj   = nn.Linear(flat_act, hidden)
        self.blocks    = nn.ModuleList([ResBlock(hidden,cond_dim) for _ in range(depth)])
        self.out_proj  = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden,flat_act))

    def forward(self, noisy, timestep, state_flat, imgs):
        """
        noisy:      (B, ACTION_HORIZON, ACTION_DIM)
        timestep:   (B,)
        state_flat: (B, STATE_DIM*OBS_HORIZON)
        imgs:       (B, OBS_HORIZON, 3, H, W)
        """
        B = noisy.shape[0]

        # Encode images — process each frame
        img_feats = []
        for i in range(OBS_HORIZON):
            feat = self.visual_enc(imgs[:, i])   # (B, 512)
            img_feats.append(feat)
        img_feats = torch.cat(img_feats, dim=-1)  # (B, 512*OBS_HORIZON)

        # Fuse state + image
        obs_combined = torch.cat([state_flat, img_feats], dim=-1)
        obs_emb = self.obs_fuse(obs_combined)   # (B, 256)

        # Build conditioning
        cond = self.cond_proj(
            torch.cat([obs_emb, self.time_emb(timestep)], dim=-1))

        # Denoise
        x = self.in_proj(noisy.reshape(B, -1))
        for blk in self.blocks: x = blk(x, cond)
        return self.out_proj(x).reshape(B, ACTION_HORIZON, ACTION_DIM)

# ── Train ─────────────────────────────────────────────────────────
def train():
    z      = zarr.open(ZARR_PATH, 'r')
    pos    = z['data']['robot0_eef_pos'][:]
    rot    = z['data']['robot0_eef_rot_axis_angle'][:]
    grip   = z['data']['robot0_gripper_width'][:]
    states = np.concatenate([pos,rot,grip], axis=-1).astype(np.float32)

    # Check image shape
    imgs_zarr = z['data']['camera0_rgb']
    print(f"Camera shape: {imgs_zarr.shape}, dtype: {imgs_zarr.dtype}")
    print(f"Image size: {imgs_zarr.shape[1]}x{imgs_zarr.shape[2]} → resizing to {IMG_SIZE}x{IMG_SIZE}")
    print(f"Total data: ~{imgs_zarr.nbytes/1024**3:.1f} GB (lazy loaded)")

    obs_norm = Normalizer(states)
    act_norm = Normalizer(states)
    obs_norm.save(f'{SAVE_DIR}/obs_normalizer.pt')
    act_norm.save(f'{SAVE_DIR}/act_normalizer.pt')

    dataset = UMIVisionDataset(ZARR_PATH, obs_norm, act_norm)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=4, pin_memory=True, drop_last=True,
                         prefetch_factor=2)

    model       = VisionDiffusionNet().to(DEVICE)
    # Separate LR for visual encoder (lower) vs rest (higher)
    visual_params = list(model.visual_enc.parameters())
    other_params  = [p for n,p in model.named_parameters()
                     if 'visual_enc' not in n]
    optimizer = torch.optim.AdamW([
        {'params': visual_params, 'lr': LR*0.1},
        {'params': other_params,  'lr': LR}
    ], weight_decay=1e-4)

    sched_lr    = torch.optim.lr_scheduler.CosineAnnealingLR(
                      optimizer, T_max=EPOCHS*len(loader))
    noise_sched = DDPMScheduler(num_train_timesteps=NUM_DIFFUSION_STEPS,
                                 beta_schedule='squaredcos_cap_v2',
                                 clip_sample=True, prediction_type='epsilon')

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {total/1e6:.1f}M params ({trainable/1e6:.1f}M trainable)")
    print(f"Batches/epoch: {len(loader)}")
    print(f"Training {EPOCHS} epochs with vision conditioning\n")

    best_loss = float('inf')
    for epoch in range(1, EPOCHS+1):
        model.train(); eloss=0.0; t0=time.time()
        for state_flat, imgs, actions in loader:
            state_flat = state_flat.to(DEVICE)
            imgs       = imgs.to(DEVICE)
            actions    = actions.to(DEVICE)
            B = actions.shape[0]

            ts    = torch.randint(0, NUM_DIFFUSION_STEPS, (B,), device=DEVICE).long()
            noise = torch.randn_like(actions)
            na    = noise_sched.add_noise(actions, noise, ts)
            pred  = model(na, ts, state_flat, imgs)
            loss  = F.mse_loss(pred, noise)

            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); sched_lr.step()
            eloss += loss.item()

        avg = eloss/len(loader)
        if epoch%10==0 or epoch==1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  loss={avg:.5f}  "
                  f"lr={sched_lr.get_last_lr()[0]:.2e}  {time.time()-t0:.1f}s")
        if avg < best_loss:
            best_loss = avg
            torch.save({'epoch':epoch,'loss':avg,
                        'model_state':model.state_dict()},
                       f'{SAVE_DIR}/best_model.pt')

    print(f"\n✅ Done! Best loss: {best_loss:.5f}")
    print(f"Checkpoints: {SAVE_DIR}/")

if __name__ == '__main__':
    train()