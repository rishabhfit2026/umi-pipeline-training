"""
Train Diffusion Policy on REAL UMI dataset
Uses pose-only data (eef_pos + rot + gripper) from replay_buffer.zarr
No camera images needed for first pass.

conda activate maniskill2
python train_umi_diffusion.py
"""

import os, math, time, numpy as np, zarr, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

ZARR_PATH = '/home/rishabh/Downloads/umi-pipeline-training/replay_buffer.zarr'
SAVE_DIR  = './checkpoints_umi'
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'

OBS_HORIZON    = 4
ACTION_HORIZON = 16
ACTION_DIM     = 7   # eef_pos(3) + rot(3) + gripper(1)
STATE_DIM      = 7
GOAL_DIM       = 12   # demo_start_pose(3) + demo_end_pose(3)
OBS_IN         =  STATE_DIM * OBS_HORIZON + GOAL_DIM  # 34

BATCH_SIZE          = 128
LR                  = 1e-4
EPOCHS              = 500
NUM_DIFFUSION_STEPS = 100

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Device: {DEVICE}")

# ── Normalizer ────────────────────────────────────────────────────
class Normalizer:
    def __init__(self, data):
        self.min   = torch.tensor(data.min(0), dtype=torch.float32)
        self.max   = torch.tensor(data.max(0), dtype=torch.float32)
        self.scale = self.max - self.min
        self.scale[self.scale < 1e-6] = 1.0
    def normalize(self, x):
        return 2.0*(x - self.min.to(x.device))/self.scale.to(x.device) - 1.0
    def denormalize(self, x):
        return (x+1.0)/2.0*self.scale.to(x.device) + self.min.to(x.device)
    def save(self, path):
        torch.save({'min':self.min,'max':self.max,'scale':self.scale}, path)
    @classmethod
    def load(cls, path):
        n=cls.__new__(cls); d=torch.load(path, map_location='cpu')
        n.min,n.max,n.scale=d['min'],d['max'],d['scale']; return n

# ── Load & inspect data ───────────────────────────────────────────
def load_umi_data(zarr_path):
    z     = zarr.open(zarr_path, 'r')
    pos   = z['data']['robot0_eef_pos'][:]
    rot   = z['data']['robot0_eef_rot_axis_angle'][:]
    grip  = z['data']['robot0_gripper_width'][:]
    ends  = z['meta']['episode_ends'][:]

    # demo start/end poses as goal conditioning
    # If these are per-episode (not per-frame), we need to broadcast
    try:
        start_poses = z['data']['robot0_demo_start_pose'][:]
        end_poses   = z['data']['robot0_demo_end_pose'][:]
        has_demo_poses = True
        print(f"demo_start_pose shape: {start_poses.shape}")
        print(f"demo_end_pose shape:   {end_poses.shape}")
    except:
        has_demo_poses = False
        print("No demo poses found — using first/last frame of each episode as goal")

    starts = np.concatenate([[0], ends[:-1]])
    print(f"\nDataset info:")
    print(f"  Episodes:     {len(ends)}")
    print(f"  Total frames: {len(pos)}")
    print(f"  Avg ep len:   {len(pos)/len(ends):.1f} frames")
    print(f"  EEF pos range: x=[{pos[:,0].min():.3f},{pos[:,0].max():.3f}]"
          f" y=[{pos[:,1].min():.3f},{pos[:,1].max():.3f}]"
          f" z=[{pos[:,2].min():.3f},{pos[:,2].max():.3f}]")
    print(f"  Gripper range: [{grip.min():.4f}, {grip.max():.4f}]")

    # Build per-frame goal: use end pose of each episode
    # This teaches the model "given where you need to end up, predict actions"
    ep_goals = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        if has_demo_poses:
            # Use demo start+end pose directly
            if len(start_poses.shape) == 2 and start_poses.shape[0] == len(ends):
                # per-episode
                goal = np.concatenate([start_poses[i], end_poses[i]])
            else:
                # per-frame — take first/last frame of episode
                goal = np.concatenate([start_poses[s], end_poses[e-1]])
        else:
            # fallback: use first and last EEF pos of episode as goal
            goal = np.concatenate([pos[s], pos[e-1]])
        ep_goals.append(goal.astype(np.float32))

    ep_goals = np.array(ep_goals)  # (N_eps, GOAL_DIM)
    print(f"  Goal shape: {ep_goals.shape}")

    states = np.concatenate([pos, rot, grip], axis=-1).astype(np.float32)
    return states, ep_goals, ends, starts

# ── Dataset ───────────────────────────────────────────────────────
class UMIDataset(Dataset):
    def __init__(self, states, ep_goals, ends, starts,
                 obs_norm, act_norm, goal_norm):
        self.states    = states
        self.ep_goals  = ep_goals
        self.obs_norm  = obs_norm
        self.act_norm  = act_norm
        self.goal_norm = goal_norm

        self.indices = []
        for ep_i, (s, e) in enumerate(zip(starts, ends)):
            for t in range(int(s)+OBS_HORIZON-1, int(e)-ACTION_HORIZON):
                self.indices.append((ep_i, t))
        print(f"Dataset: {len(self.indices)} samples from {len(ends)} episodes")

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        ep_i, t = self.indices[idx]
        obs = [self.obs_norm.normalize(
                   torch.tensor(self.states[t-OBS_HORIZON+1+i], dtype=torch.float32))
               for i in range(OBS_HORIZON)]
        obs_flat = torch.cat(obs)
        goal     = self.goal_norm.normalize(
                       torch.tensor(self.ep_goals[ep_i], dtype=torch.float32))
        obs_goal = torch.cat([obs_flat, goal])
        act      = self.act_norm.normalize(
                       torch.tensor(self.states[t:t+ACTION_HORIZON], dtype=torch.float32))
        return obs_goal, act

# ── Model ─────────────────────────────────────────────────────────
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
        self.cond = nn.Linear(cond_dim, dim*2)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, cond):
        scale,bias = self.cond(cond).chunk(2,dim=-1)
        return x + self.net(self.norm(x)*(scale+1)+bias)

class UMIDiffusionNet(nn.Module):
    def __init__(self, hidden=512, depth=8):
        super().__init__()
        flat_act = ACTION_DIM * ACTION_HORIZON
        cond_dim = 512
        self.obs_emb   = nn.Sequential(
            nn.Linear(OBS_IN,256),nn.Mish(),
            nn.Linear(256,256),nn.Mish(),nn.Linear(256,256))
        self.time_emb  = nn.Sequential(
            SinusoidalPosEmb(128),nn.Linear(128,256),
            nn.Mish(),nn.Linear(256,256))
        self.cond_proj = nn.Sequential(
            nn.Linear(512,cond_dim),nn.Mish(),nn.Linear(cond_dim,cond_dim))
        self.in_proj   = nn.Linear(flat_act, hidden)
        self.blocks    = nn.ModuleList([ResBlock(hidden,cond_dim) for _ in range(depth)])
        self.out_proj  = nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,flat_act))

    def forward(self, noisy, timestep, obs_goal):
        B = noisy.shape[0]; x = noisy.reshape(B,-1)
        cond = self.cond_proj(torch.cat([
            self.obs_emb(obs_goal), self.time_emb(timestep)],dim=-1))
        x = self.in_proj(x)
        for blk in self.blocks: x = blk(x, cond)
        return self.out_proj(x).reshape(B, ACTION_HORIZON, ACTION_DIM)

# ── Train ─────────────────────────────────────────────────────────
def train():
    states, ep_goals, ends, starts = load_umi_data(ZARR_PATH)

    obs_norm  = Normalizer(states)
    act_norm  = Normalizer(states)
    goal_norm = Normalizer(ep_goals)
    obs_norm.save(f'{SAVE_DIR}/obs_normalizer.pt')
    act_norm.save(f'{SAVE_DIR}/act_normalizer.pt')
    goal_norm.save(f'{SAVE_DIR}/goal_normalizer.pt')

    dataset = UMIDataset(states, ep_goals, ends, starts,
                         obs_norm, act_norm, goal_norm)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=2, pin_memory=True, drop_last=True)

    model       = UMIDiffusionNet().to(DEVICE)
    optimizer   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched_lr    = torch.optim.lr_scheduler.CosineAnnealingLR(
                      optimizer, T_max=EPOCHS*len(loader))
    noise_sched = DDPMScheduler(num_train_timesteps=NUM_DIFFUSION_STEPS,
                                 beta_schedule='squaredcos_cap_v2',
                                 clip_sample=True, prediction_type='epsilon')

    total = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total/1e6:.1f}M params  |  OBS_IN={OBS_IN}")
    print(f"Training {EPOCHS} epochs on REAL UMI data\n")

    best_loss = float('inf')
    for epoch in range(1, EPOCHS+1):
        model.train(); eloss=0.0; t0=time.time()
        for obs_goal, actions in loader:
            obs_goal = obs_goal.to(DEVICE); actions = actions.to(DEVICE)
            B = actions.shape[0]
            ts    = torch.randint(0, NUM_DIFFUSION_STEPS, (B,), device=DEVICE).long()
            noise = torch.randn_like(actions)
            na    = noise_sched.add_noise(actions, noise, ts)
            pred  = model(na, ts, obs_goal)
            loss  = F.mse_loss(pred, noise)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); sched_lr.step()
            eloss += loss.item()
        avg = eloss/len(loader)
        if epoch%20==0 or epoch==1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  loss={avg:.5f}  "
                  f"lr={sched_lr.get_last_lr()[0]:.2e}  {time.time()-t0:.1f}s")
        if avg < best_loss:
            best_loss = avg
            torch.save({
                'epoch': epoch, 'loss': avg,
                'model_state': model.state_dict(),
                'obs_in': OBS_IN, 'goal_dim': GOAL_DIM
            }, f'{SAVE_DIR}/best_model.pt')

    print(f"\n✅ Done! Best loss: {best_loss:.5f}")
    print(f"Checkpoints: {SAVE_DIR}/")

if __name__ == '__main__':
    train()