"""
Pose-Only Diffusion Policy — no camera needed
Images in zarr are blank gray → use EEF poses directly

conda activate maniskill2
python train_pose_only.py
"""

import os, math, time, numpy as np, zarr, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

ZARR_PATH           = '/home/rishabh/Downloads/umi-pipeline-training/outputs/sim_replay_buffer3.zarr'
SAVE_DIR            = './checkpoints_pose_only'
DEVICE              = 'cuda' if torch.cuda.is_available() else 'cpu'
OBS_HORIZON         = 4
ACTION_HORIZON      = 16
ACTION_DIM          = 7
OBS_DIM             = 7
BATCH_SIZE          = 256
LR                  = 1e-4
EPOCHS              = 200
SAVE_EVERY          = 20
NUM_WORKERS         = 4
NUM_DIFFUSION_STEPS = 100

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Device: {DEVICE}")
print(f"Pose-only training — no camera needed!")

# ── Normalizer ────────────────────────────────────────────────────
class Normalizer:
    def __init__(self, data):
        self.min   = torch.tensor(data.min(0), dtype=torch.float32)
        self.max   = torch.tensor(data.max(0), dtype=torch.float32)
        self.scale = self.max - self.min
        self.scale[self.scale < 1e-6] = 1.0

    def normalize(self, x):
        # works for shape (7,) or (N, 7)
        return 2.0 * (x - self.min.to(x.device)) / self.scale.to(x.device) - 1.0

    def denormalize(self, x):
        return (x + 1.0) / 2.0 * self.scale.to(x.device) + self.min.to(x.device)

    def save(self, path):
        torch.save({'min': self.min, 'max': self.max, 'scale': self.scale}, path)

    @classmethod
    def load(cls, path):
        n = cls.__new__(cls)
        d = torch.load(path, map_location='cpu')
        n.min, n.max, n.scale = d['min'], d['max'], d['scale']
        return n

# ── Dataset ───────────────────────────────────────────────────────
class PoseDataset(Dataset):
    def __init__(self, zarr_path, obs_horizon, action_horizon,
                 obs_normalizer, act_normalizer):
        z      = zarr.open(zarr_path, 'r')
        pos    = z['data']['robot0_eef_pos'][:]
        rot    = z['data']['robot0_eef_rot_axis_angle'][:]
        grip   = z['data']['robot0_gripper_width'][:]
        ends   = z['meta']['episode_ends'][:]
        starts = np.concatenate([[0], ends[:-1]])

        self.states   = np.concatenate([pos, rot, grip], axis=-1).astype(np.float32)
        self.obs_h    = obs_horizon
        self.act_h    = action_horizon
        self.obs_norm = obs_normalizer
        self.act_norm = act_normalizer

        self.indices = []
        for s, e in zip(starts, ends):
            for t in range(s + obs_horizon - 1, e - action_horizon):
                self.indices.append(t)
        print(f"Dataset: {len(self.indices)} samples from {len(ends)} episodes")

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]

        # ✅ normalize each 7-dim frame individually then concatenate → (obs_h*7,)
        obs = []
        for i in range(self.obs_h):
            s = torch.tensor(self.states[t - self.obs_h + 1 + i])  # (7,)
            obs.append(self.obs_norm.normalize(s))                  # (7,)
        obs_flat = torch.cat(obs)                                   # (obs_h*7,)

        # action chunk (act_h, 7) — normalizer broadcasts over last dim fine
        act = torch.tensor(self.states[t : t + self.act_h])        # (act_h, 7)
        act = self.act_norm.normalize(act)

        return obs_flat, act

# ── Time Embedding ────────────────────────────────────────────────
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        device = t.device
        half   = self.dim // 2
        emb    = math.log(10000) / (half - 1)
        emb    = torch.exp(torch.arange(half, device=device) * -emb)
        emb    = t.float()[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

# ── Model ─────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.net  = nn.Sequential(nn.Linear(dim, dim), nn.Mish(), nn.Linear(dim, dim))
        self.cond = nn.Linear(cond_dim, dim * 2)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, cond):
        scale, bias = self.cond(cond).chunk(2, dim=-1)
        h = self.norm(x) * (scale + 1) + bias
        return x + self.net(h)

class PoseDiffusionNet(nn.Module):
    def __init__(self, action_dim, action_horizon, obs_dim, obs_horizon,
                 hidden=512, depth=6):
        super().__init__()
        self.act_h   = action_horizon
        self.act_dim = action_dim
        flat_act     = action_dim * action_horizon
        obs_in       = obs_dim * obs_horizon
        cond_dim     = 512

        self.obs_emb   = nn.Sequential(nn.Linear(obs_in, 256), nn.Mish(), nn.Linear(256, 256))
        self.time_emb  = nn.Sequential(SinusoidalPosEmb(128), nn.Linear(128, 256),
                                       nn.Mish(), nn.Linear(256, 256))
        self.cond_proj = nn.Sequential(nn.Linear(512, cond_dim), nn.Mish(),
                                       nn.Linear(cond_dim, cond_dim))
        self.in_proj   = nn.Linear(flat_act, hidden)
        self.blocks    = nn.ModuleList([ResBlock(hidden, cond_dim) for _ in range(depth)])
        self.out_proj  = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, flat_act))

    def forward(self, noisy_action, timestep, obs_flat):
        B    = noisy_action.shape[0]
        x    = noisy_action.reshape(B, -1)
        cond = self.cond_proj(torch.cat([self.obs_emb(obs_flat),
                                         self.time_emb(timestep)], dim=-1))
        x = self.in_proj(x)
        for blk in self.blocks: x = blk(x, cond)
        return self.out_proj(x).reshape(B, self.act_h, self.act_dim)

# ── Train ─────────────────────────────────────────────────────────
def train():
    z    = zarr.open(ZARR_PATH, 'r')
    pos  = z['data']['robot0_eef_pos'][:]
    rot  = z['data']['robot0_eef_rot_axis_angle'][:]
    grip = z['data']['robot0_gripper_width'][:]
    all_states = np.concatenate([pos, rot, grip], axis=-1).astype(np.float32)

    obs_norm = Normalizer(all_states)
    act_norm = Normalizer(all_states)
    obs_norm.save(os.path.join(SAVE_DIR, 'obs_normalizer.pt'))
    act_norm.save(os.path.join(SAVE_DIR, 'act_normalizer.pt'))
    print(f"State min: {all_states.min(0)}")
    print(f"State max: {all_states.max(0)}")

    dataset = PoseDataset(ZARR_PATH, OBS_HORIZON, ACTION_HORIZON, obs_norm, act_norm)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    model        = PoseDiffusionNet(ACTION_DIM, ACTION_HORIZON, OBS_DIM, OBS_HORIZON).to(DEVICE)
    optimizer    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS*len(loader))
    noise_sched  = DDPMScheduler(num_train_timesteps=NUM_DIFFUSION_STEPS,
                                 beta_schedule='squaredcos_cap_v2',
                                 clip_sample=True, prediction_type='epsilon')

    print(f"\nModel: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    print(f"Batches/epoch: {len(loader)}  (~{len(loader)*BATCH_SIZE/1000:.0f}k samples)\n")

    best_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        model.train(); epoch_loss = 0.0; t0 = time.time()
        for obs_flat, actions in loader:
            obs_flat = obs_flat.to(DEVICE)
            actions  = actions.to(DEVICE)
            B        = actions.shape[0]
            timesteps     = torch.randint(0, NUM_DIFFUSION_STEPS, (B,), device=DEVICE).long()
            noise         = torch.randn_like(actions)
            noisy_actions = noise_sched.add_noise(actions, noise, timesteps)
            pred_noise    = model(noisy_actions, timesteps, obs_flat)
            loss          = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler_lr.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.5f}  "
              f"lr={scheduler_lr.get_last_lr()[0]:.2e}  time={time.time()-t0:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'epoch': epoch, 'loss': avg_loss,
                        'model_state': model.state_dict(),
                        'obs_horizon': OBS_HORIZON, 'action_horizon': ACTION_HORIZON,
                        'action_dim': ACTION_DIM, 'obs_dim': OBS_DIM},
                       os.path.join(SAVE_DIR, 'best_model.pt'))
            print(f"  ✅ Saved best model (loss={best_loss:.5f})")

        if epoch % SAVE_EVERY == 0:
            torch.save({'epoch': epoch, 'loss': avg_loss, 'model_state': model.state_dict()},
                       os.path.join(SAVE_DIR, f'checkpoint_ep{epoch}.pt'))

    print(f"\nDone! Best loss: {best_loss:.5f}")

if __name__ == '__main__':
    train()