"""
Inference for vision diffusion policy
Runs on held-out episodes and measures trajectory error vs ground truth

conda activate maniskill2
python infer_umi_vision.py
"""

import os, math, numpy as np, zarr, torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import matplotlib.pyplot as plt

ZARR_PATH  = '/home/rishabh/Downloads/umi-pipeline-training/replay_buffer.zarr'
CKPT_DIR   = './checkpoints_umi_vision'
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

OBS_HORIZON    = 2
ACTION_HORIZON = 16
ACTION_DIM     = 7
STATE_DIM      = 7
IMG_FEAT_DIM   = 512
IMG_SIZE       = 96
NUM_STEPS      = 100

img_transform = T.Compose([
    T.ToPILImage(), T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# ── Model classes ─────────────────────────────────────────────────
class Normalizer:
    def __init__(self): pass
    def normalize(self, x):
        return 2.0*(x - self.min.to(x.device)) / self.scale.to(x.device) - 1.0
    def denormalize(self, x):
        return (x + 1.0) / 2.0 * self.scale.to(x.device) + self.min.to(x.device)
    @classmethod
    def load(cls, path):
        n = cls()
        d = torch.load(path, map_location='cpu', weights_only=False)
        n.min, n.max, n.scale = d['min'], d['max'], d['scale']
        return n

class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = tvm.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
    def forward(self, x):
        return self.encoder(x).squeeze(-1).squeeze(-1)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        emb  = math.log(10000) / (half - 1)
        emb  = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb  = t.float()[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.net  = nn.Sequential(nn.Linear(dim,dim), nn.Mish(), nn.Linear(dim,dim))
        self.cond = nn.Linear(cond_dim, dim*2)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, cond):
        scale, bias = self.cond(cond).chunk(2, dim=-1)
        return x + self.net(self.norm(x) * (scale+1) + bias)

class VisionDiffusionNet(nn.Module):
    def __init__(self, hidden=512, depth=8):
        super().__init__()
        flat_act = ACTION_DIM * ACTION_HORIZON
        cond_dim = 512
        self.visual_enc = VisualEncoder()
        fuse_in = STATE_DIM*OBS_HORIZON + IMG_FEAT_DIM*OBS_HORIZON
        self.obs_fuse = nn.Sequential(
            nn.Linear(fuse_in,512), nn.Mish(),
            nn.Linear(512,512),     nn.Mish(),
            nn.Linear(512,256))
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(128), nn.Linear(128,256),
            nn.Mish(), nn.Linear(256,256))
        self.cond_proj = nn.Sequential(
            nn.Linear(512,cond_dim), nn.Mish(),
            nn.Linear(cond_dim,cond_dim))
        self.in_proj  = nn.Linear(flat_act, hidden)
        self.blocks   = nn.ModuleList([ResBlock(hidden,cond_dim) for _ in range(depth)])
        self.out_proj = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden,flat_act))

    def forward(self, noisy, timestep, state_flat, imgs):
        B = noisy.shape[0]
        img_feats = torch.cat([self.visual_enc(imgs[:,i])
                                for i in range(OBS_HORIZON)], dim=-1)
        obs_emb = self.obs_fuse(torch.cat([state_flat, img_feats], dim=-1))
        cond    = self.cond_proj(torch.cat([obs_emb, self.time_emb(timestep)], dim=-1))
        x = self.in_proj(noisy.reshape(B,-1))
        for blk in self.blocks: x = blk(x, cond)
        return self.out_proj(x).reshape(B, ACTION_HORIZON, ACTION_DIM)

# ── Load ──────────────────────────────────────────────────────────
print("Loading model and data...")
obs_norm = Normalizer.load(f'{CKPT_DIR}/obs_normalizer.pt')
act_norm = Normalizer.load(f'{CKPT_DIR}/act_normalizer.pt')

model = VisionDiffusionNet().to(DEVICE)
ck    = torch.load(f'{CKPT_DIR}/best_model.pt', map_location=DEVICE, weights_only=False)
model.load_state_dict(ck['model_state'])
model.eval()
print(f"Loaded checkpoint: epoch={ck['epoch']}, loss={ck['loss']:.5f}")

noise_sched = DDPMScheduler(
    num_train_timesteps=NUM_STEPS,
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True,
    prediction_type='epsilon')

z      = zarr.open(ZARR_PATH, 'r')
pos    = z['data']['robot0_eef_pos'][:]
rot    = z['data']['robot0_eef_rot_axis_angle'][:]
grip   = z['data']['robot0_gripper_width'][:]
imgs_z = z['data']['camera0_rgb']
ends   = z['meta']['episode_ends'][:]
starts = np.concatenate([[0], ends[:-1]])
states = np.concatenate([pos, rot, grip], axis=-1).astype(np.float32)

# ── Single-step inference ─────────────────────────────────────────
@torch.no_grad()
def infer(t_idx):
    state_obs = []
    for i in range(OBS_HORIZON):
        s = torch.tensor(states[t_idx - OBS_HORIZON + 1 + i], dtype=torch.float32)
        state_obs.append(obs_norm.normalize(s))
    state_flat = torch.cat(state_obs).unsqueeze(0).to(DEVICE)

    imgs = []
    for i in range(OBS_HORIZON):
        fi = t_idx - OBS_HORIZON + 1 + i
        imgs.append(img_transform(imgs_z[fi]))
    imgs = torch.stack(imgs).unsqueeze(0).to(DEVICE)

    noisy = torch.randn(1, ACTION_HORIZON, ACTION_DIM, device=DEVICE)
    noise_sched.set_timesteps(NUM_STEPS)
    for ts in noise_sched.timesteps:
        ts_b = ts.unsqueeze(0).to(DEVICE)
        pred  = model(noisy, ts_b, state_flat, imgs)
        noisy = noise_sched.step(pred, ts, noisy).prev_sample

    return act_norm.denormalize(noisy.squeeze(0)).cpu().numpy()

# ── Evaluate held-out episodes ────────────────────────────────────
n_test   = int(len(ends) * 0.8)
test_eps = list(range(n_test, len(ends)))[:6]
print(f"\nEvaluating {len(test_eps)} held-out episodes...\n")

# Print episode info upfront
for ep_i in test_eps:
    s, e = int(starts[ep_i]), int(ends[ep_i])
    n_windows = len(range(s + OBS_HORIZON, e - ACTION_HORIZON,
                          max(1, ACTION_HORIZON // 2)))
    print(f"  Ep {ep_i+1}: frames={e-s}  windows={n_windows}")
print()

all_means, all_maxs, all_finals, all_pct = [], [], [], []
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, ep_i in enumerate(test_eps):
    s, e      = int(starts[ep_i]), int(ends[ep_i])
    ep_len    = e - s
    ep_states = states[s:e]

    # FIX 1 — skip episodes too short to produce any window
    min_len = OBS_HORIZON + ACTION_HORIZON + 1
    if ep_len < min_len:
        print(f"Ep {ep_i+1:3d}: SKIPPED (only {ep_len} frames, need {min_len})")
        axes[idx].set_title(f'Ep {ep_i+1} — too short ({ep_len} frames)', fontsize=10)
        axes[idx].axis('off')
        continue

    pred_traj, gt_traj = [], []
    step = max(1, ACTION_HORIZON // 2)

    for t in range(s + OBS_HORIZON, e - ACTION_HORIZON, step):
        pred = infer(t)

        # FIX 2 — index gt within episode slice, not global array
        local_t  = t - s
        gt       = ep_states[local_t : local_t + ACTION_HORIZON]

        # FIX 3 — skip window if gt is shorter than ACTION_HORIZON
        if len(gt) < ACTION_HORIZON:
            continue

        n = min(len(pred), len(gt))
        pred_traj.append(pred[:n])
        gt_traj.append(gt[:n])

    # FIX 4 — guard before concatenation
    if len(pred_traj) == 0:
        print(f"Ep {ep_i+1:3d}: SKIPPED (no valid windows in {ep_len} frames)")
        axes[idx].set_title(f'Ep {ep_i+1} — no windows', fontsize=10)
        axes[idx].axis('off')
        continue

    pred_arr = np.concatenate(pred_traj)
    gt_arr   = np.concatenate(gt_traj)

    # FIX 5 — guard for empty arrays after concat
    if pred_arr.shape[0] == 0 or gt_arr.shape[0] == 0:
        print(f"Ep {ep_i+1:3d}: SKIPPED (empty after concat)")
        axes[idx].axis('off')
        continue

    errs    = np.linalg.norm(pred_arr[:, :3] - gt_arr[:, :3], axis=-1) * 100
    mean_e  = float(errs.mean())
    max_e   = float(errs.max())
    final_e = float(errs[-1])
    pct2    = float(np.mean(errs < 2.0) * 100)

    all_means.append(mean_e); all_maxs.append(max_e)
    all_finals.append(final_e); all_pct.append(pct2)

    ax = axes[idx]
    ax.plot(errs, color='#1D9E75', linewidth=1.2, alpha=0.9)
    ax.axhline(2.0, color='#BA7517', linewidth=1, linestyle='--', label='2cm threshold')
    ax.axhline(mean_e, color='#533AB7', linewidth=1, linestyle=':',
               label=f'mean={mean_e:.1f}cm')
    ax.fill_between(range(len(errs)), 0, errs, alpha=0.15, color='#1D9E75')
    ax.set_title(f'Ep {ep_i+1} | mean={mean_e:.1f}cm | <2cm={pct2:.0f}%', fontsize=10)
    ax.set_xlabel('Step')
    ax.set_ylabel('Position error (cm)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(errs.max() * 1.1, 3.0))

    print(f"Ep {ep_i+1:3d}: mean={mean_e:.1f}cm  max={max_e:.1f}cm  "
          f"final={final_e:.1f}cm  <2cm={pct2:.0f}%  windows={len(pred_traj)}")

# ── Summary ───────────────────────────────────────────────────────
if len(all_means) == 0:
    print("\nNo valid episodes found.")
else:
    om = np.mean(all_means)
    op = np.mean(all_pct)
    plt.suptitle(
        f'Vision Diffusion Policy — Trajectory Error\n'
        f'Overall mean={om:.1f}cm  <2cm={op:.0f}%  '
        f'({len(all_means)}/{len(test_eps)} episodes valid)',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    out_path = f'{CKPT_DIR}/vision_inference_results.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')

    print(f"\n{'='*50}")
    print(f"Valid episodes     : {len(all_means)} / {len(test_eps)}")
    print(f"Overall mean error : {om:.2f} cm")
    print(f"Overall max error  : {np.mean(all_maxs):.2f} cm")
    print(f"Overall final error: {np.mean(all_finals):.2f} cm")
    print(f"% frames < 2cm     : {op:.1f}%")
    print(f"\nSaved → {out_path}")

plt.show()