"""
Goal-Conditioned Diffusion Policy
Input: last 4 EEF poses + marker_xy + box_xy → predict next 16 actions
NOW the model KNOWS where to go!

conda activate maniskill2
python goal_diffusion_train.py
"""

import os, math, time, numpy as np, zarr, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

ZARR_PATH  = '/home/rishabh/Downloads/umi-pipeline-training/outputs/sim_replay_buffer3.zarr'
SAVE_DIR   = './checkpoints_goal'
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

OBS_HORIZON    = 4
ACTION_HORIZON = 16
ACTION_DIM     = 7
STATE_DIM      = 7   # per timestep
GOAL_DIM       = 4   # marker_x, marker_y, box_x, box_y
OBS_IN         = STATE_DIM * OBS_HORIZON + GOAL_DIM  # 28 + 4 = 32

BATCH_SIZE = 256
LR         = 1e-4
EPOCHS     = 300
SAVE_EVERY = 50
NUM_DIFFUSION_STEPS = 100

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Device: {DEVICE}")
print(f"Goal-conditioned diffusion: obs(28) + goal(4) = {OBS_IN} input dims")

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

# ── Dataset ───────────────────────────────────────────────────────
class GoalDataset(Dataset):
    def __init__(self, zarr_path, obs_norm, act_norm, goal_norm):
        z      = zarr.open(zarr_path, 'r')
        pos    = z['data']['robot0_eef_pos'][:]
        rot    = z['data']['robot0_eef_rot_axis_angle'][:]
        grip   = z['data']['robot0_gripper_width'][:]
        ends   = z['meta']['episode_ends'][:]
        starts = np.concatenate([[0], ends[:-1]])

        # Reconstruct marker+box positions using same RNG as generation
        # EE home ~(0.250, 0.000)
        TX, TY = 0.250, 0.000
        rng = np.random.default_rng(42)
        self.ep_goals = []
        for _ in range(len(ends)):
            mx = TX + rng.uniform(-0.05, 0.05)
            my = TY + rng.uniform(-0.10, 0.10)
            bx = TX + rng.uniform(-0.05, 0.05)
            by = TY + rng.uniform( 0.05, 0.15)
            while abs(mx-bx)<0.04 and abs(my-by)<0.04:
                bx = TX + rng.uniform(-0.05, 0.05)
                by = TY + rng.uniform( 0.05, 0.15)
            self.ep_goals.append([mx, my, bx, by])
        self.ep_goals = np.array(self.ep_goals, dtype=np.float32)

        self.states   = np.concatenate([pos, rot, grip], axis=-1).astype(np.float32)
        self.ends     = ends
        self.starts   = starts
        self.obs_norm = obs_norm
        self.act_norm = act_norm
        self.goal_norm= goal_norm

        self.indices = []
        for ep_i, (s, e) in enumerate(zip(starts, ends)):
            for t in range(s + OBS_HORIZON - 1, e - ACTION_HORIZON):
                self.indices.append((ep_i, t))
        print(f"Dataset: {len(self.indices)} samples, {len(ends)} episodes")

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        ep_i, t = self.indices[idx]

        # Obs: last 4 states normalized individually
        obs = []
        for i in range(OBS_HORIZON):
            s = torch.tensor(self.states[t - OBS_HORIZON + 1 + i])
            obs.append(self.obs_norm.normalize(s))
        obs_flat = torch.cat(obs)  # (28,)

        # Goal: marker_xy + box_xy normalized
        goal = torch.tensor(self.ep_goals[ep_i])  # (4,)
        goal = self.goal_norm.normalize(goal)       # (4,)

        # Concat obs + goal
        obs_goal = torch.cat([obs_flat, goal])  # (32,)

        # Actions
        act = torch.tensor(self.states[t : t + ACTION_HORIZON])
        act = self.act_norm.normalize(act)

        return obs_goal, act

# ── Model ─────────────────────────────────────────────────────────
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim=dim
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
        scale,bias=self.cond(cond).chunk(2,dim=-1)
        h=self.norm(x)*(scale+1)+bias
        return x+self.net(h)

class GoalDiffusionNet(nn.Module):
    def __init__(self, hidden=512, depth=8):
        super().__init__()
        flat_act = ACTION_DIM * ACTION_HORIZON  # 112
        cond_dim = 512

        # Obs+goal encoder
        self.obs_emb = nn.Sequential(
            nn.Linear(OBS_IN, 256), nn.Mish(),
            nn.Linear(256, 256), nn.Mish(),
            nn.Linear(256, 256))

        # Time encoder
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(128),
            nn.Linear(128,256), nn.Mish(),
            nn.Linear(256,256))

        self.cond_proj = nn.Sequential(
            nn.Linear(512, cond_dim), nn.Mish(),
            nn.Linear(cond_dim, cond_dim))

        self.in_proj  = nn.Linear(flat_act, hidden)
        self.blocks   = nn.ModuleList([ResBlock(hidden, cond_dim) for _ in range(depth)])
        self.out_proj = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, flat_act))

    def forward(self, noisy_action, timestep, obs_goal):
        B   = noisy_action.shape[0]
        x   = noisy_action.reshape(B, -1)
        cond= self.cond_proj(torch.cat([self.obs_emb(obs_goal),
                                        self.time_emb(timestep)], dim=-1))
        x   = self.in_proj(x)
        for blk in self.blocks: x = blk(x, cond)
        return self.out_proj(x).reshape(B, ACTION_HORIZON, ACTION_DIM)

# ── Train ─────────────────────────────────────────────────────────
def train():
    z    = zarr.open(ZARR_PATH,'r')
    pos  = z['data']['robot0_eef_pos'][:]
    rot  = z['data']['robot0_eef_rot_axis_angle'][:]
    grip = z['data']['robot0_gripper_width'][:]
    all_states = np.concatenate([pos,rot,grip],axis=-1).astype(np.float32)

    # Reconstruct all goals for normalizer
    TX,TY=0.250,0.000
    rng=np.random.default_rng(42)
    ends=z['meta']['episode_ends'][:]
    goals=[]
    for _ in range(len(ends)):
        mx=TX+rng.uniform(-0.05,0.05); my=TY+rng.uniform(-0.10,0.10)
        bx=TX+rng.uniform(-0.05,0.05); by=TY+rng.uniform(0.05,0.15)
        while abs(mx-bx)<0.04 and abs(my-by)<0.04:
            bx=TX+rng.uniform(-0.05,0.05); by=TY+rng.uniform(0.05,0.15)
        goals.append([mx,my,bx,by])
    goals=np.array(goals,dtype=np.float32)

    obs_norm  = Normalizer(all_states)
    act_norm  = Normalizer(all_states)
    goal_norm = Normalizer(goals)
    obs_norm.save(f'{SAVE_DIR}/obs_normalizer.pt')
    act_norm.save(f'{SAVE_DIR}/act_normalizer.pt')
    goal_norm.save(f'{SAVE_DIR}/goal_normalizer.pt')

    print(f"Goal ranges:")
    print(f"  marker_x: {goals[:,0].min():.3f}→{goals[:,0].max():.3f}")
    print(f"  marker_y: {goals[:,1].min():.3f}→{goals[:,1].max():.3f}")
    print(f"  box_x:    {goals[:,2].min():.3f}→{goals[:,2].max():.3f}")
    print(f"  box_y:    {goals[:,3].min():.3f}→{goals[:,3].max():.3f}")

    dataset = GoalDataset(ZARR_PATH, obs_norm, act_norm, goal_norm)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=4, pin_memory=True, drop_last=True)

    model        = GoalDiffusionNet().to(DEVICE)
    optimizer    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
                       optimizer, T_max=EPOCHS*len(loader))
    noise_sched  = DDPMScheduler(num_train_timesteps=NUM_DIFFUSION_STEPS,
                                 beta_schedule='squaredcos_cap_v2',
                                 clip_sample=True, prediction_type='epsilon')

    total=sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total/1e6:.1f}M params  depth=8 (larger than before)")
    print(f"Training {EPOCHS} epochs...\n")

    best_loss=float('inf')
    for epoch in range(1, EPOCHS+1):
        model.train(); eloss=0.0; t0=time.time()
        for obs_goal, actions in loader:
            obs_goal=obs_goal.to(DEVICE); actions=actions.to(DEVICE)
            B=actions.shape[0]
            ts   =torch.randint(0,NUM_DIFFUSION_STEPS,(B,),device=DEVICE).long()
            noise=torch.randn_like(actions)
            na   =noise_sched.add_noise(actions,noise,ts)
            pred =model(na,ts,obs_goal)
            loss =F.mse_loss(pred,noise)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step(); scheduler_lr.step()
            eloss+=loss.item()

        avg=eloss/len(loader)
        print(f"Epoch {epoch:3d}/{EPOCHS}  loss={avg:.5f}  "
              f"lr={scheduler_lr.get_last_lr()[0]:.2e}  time={time.time()-t0:.1f}s")

        if avg<best_loss:
            best_loss=avg
            torch.save({'epoch':epoch,'loss':avg,'model_state':model.state_dict()},
                       f'{SAVE_DIR}/best_model.pt')
            print(f"  ✅ Saved (loss={best_loss:.5f})")

        if epoch%SAVE_EVERY==0:
            torch.save({'epoch':epoch,'loss':avg,'model_state':model.state_dict()},
                       f'{SAVE_DIR}/checkpoint_ep{epoch}.pt')

    print(f"\nDone! Best loss: {best_loss:.5f}")

if __name__=='__main__':
    train()