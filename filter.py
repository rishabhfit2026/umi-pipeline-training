"""
Filter dataset to ONLY successful pick+place episodes, then retrain.
200 clean episodes → model learns ONLY correct behavior.

conda activate maniskill2
python filter_and_retrain.py
"""

import os, math, time, numpy as np, zarr, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

ZARR_PATH = '/home/rishabh/Downloads/umi-pipeline-training/outputs/perfect_data.zarr'
SAVE_DIR  = './checkpoints_clean'
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'

OBS_HORIZON    = 4
ACTION_HORIZON = 16
ACTION_DIM     = 7
STATE_DIM      = 7
GOAL_DIM       = 4
OBS_IN         = STATE_DIM * OBS_HORIZON + GOAL_DIM  # 32

BATCH_SIZE = 128
LR         = 1e-4
EPOCHS     = 500
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
        n=cls.__new__(cls); d=torch.load(path,map_location='cpu')
        n.min,n.max,n.scale=d['min'],d['max'],d['scale']; return n

# ── All 200 episodes are good — no filtering needed ───────────────
def find_successful_episodes(zarr_path):
    z      = zarr.open(zarr_path, 'r')
    pos    = z['data']['robot0_eef_pos'][:]
    grip   = z['data']['robot0_gripper_width'][:][:,0]
    ends   = z['meta']['episode_ends'][:]
    starts = np.concatenate([[0], ends[:-1]])

    TX, TY = 0.249, 0.000
    rng = np.random.default_rng(42)
    ep_markers = []; ep_boxes = []
    for _ in range(len(ends)):
        mx=TX+rng.uniform(-0.05,0.05); my=TY+rng.uniform(-0.10,0.10)
        bx=TX+rng.uniform(-0.05,0.05); by=TY+rng.uniform(0.05,0.15)
        while abs(mx-bx)<0.04 and abs(my-by)<0.04:
            bx=TX+rng.uniform(-0.05,0.05); by=TY+rng.uniform(0.05,0.15)
        ep_markers.append([mx,my]); ep_boxes.append([bx,by])

    # Perfect data — accept ALL episodes, just verify they look good
    CLOSED_THR = 0.0035   # OPEN*0.1
    OPEN_THR   = 0.0173   # OPEN*0.5
    good_eps = []
    for i,(s,e) in enumerate(zip(starts,ends)):
        g = grip[s:e]
        # Must have gripper closed AND later opened
        gr = False; ok = False
        for f in range(len(g)):
            if not gr and g[f] < CLOSED_THR: gr = True
            if gr and g[f] > OPEN_THR: ok = True; break
        if ok:
            good_eps.append(i)

    print(f"\nGood episodes found: {len(good_eps)}/{len(ends)}")
    print(f"Episode indices: {good_eps[:20]}...")
    return good_eps, ep_markers, ep_boxes, pos, grip, starts, ends

# ── Dataset ───────────────────────────────────────────────────────
class CleanDataset(Dataset):
    def __init__(self, zarr_path, obs_norm, act_norm, goal_norm,
                 good_eps, ep_markers, ep_boxes):
        z     = zarr.open(zarr_path,'r')
        pos   = z['data']['robot0_eef_pos'][:]
        rot   = z['data']['robot0_eef_rot_axis_angle'][:]
        grip  = z['data']['robot0_gripper_width'][:]
        ends  = z['meta']['episode_ends'][:]
        starts= np.concatenate([[0],ends[:-1]])

        self.states    = np.concatenate([pos,rot,grip],axis=-1).astype(np.float32)
        self.ep_goals  = np.array([[ep_markers[i][0],ep_markers[i][1],
                                     ep_boxes[i][0],  ep_boxes[i][1]]
                                    for i in range(len(ends))],dtype=np.float32)
        self.obs_norm  = obs_norm
        self.act_norm  = act_norm
        self.goal_norm = goal_norm

        self.indices = []
        for ep_i in good_eps:
            s,e = int(starts[ep_i]), int(ends[ep_i])
            for t in range(s+OBS_HORIZON-1, e-ACTION_HORIZON):
                self.indices.append((ep_i, t))
        print(f"Clean dataset: {len(self.indices)} samples from {len(good_eps)} episodes")

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        ep_i,t = self.indices[idx]
        obs = [self.obs_norm.normalize(torch.tensor(self.states[t-OBS_HORIZON+1+i]))
               for i in range(OBS_HORIZON)]
        obs_flat = torch.cat(obs)
        goal = self.goal_norm.normalize(torch.tensor(self.ep_goals[ep_i]))
        obs_goal = torch.cat([obs_flat, goal])
        act = self.act_norm.normalize(torch.tensor(self.states[t:t+ACTION_HORIZON]))
        return obs_goal, act

# ── Model ─────────────────────────────────────────────────────────
class SinusoidalPosEmb(nn.Module):
    def __init__(self,dim): super().__init__(); self.dim=dim
    def forward(self,t):
        half=self.dim//2
        emb=math.log(10000)/(half-1)
        emb=torch.exp(torch.arange(half,device=t.device)*-emb)
        emb=t.float()[:,None]*emb[None,:]
        return torch.cat([emb.sin(),emb.cos()],dim=-1)

class ResBlock(nn.Module):
    def __init__(self,dim,cond_dim):
        super().__init__()
        self.net =nn.Sequential(nn.Linear(dim,dim),nn.Mish(),nn.Linear(dim,dim))
        self.cond=nn.Linear(cond_dim,dim*2)
        self.norm=nn.LayerNorm(dim)
    def forward(self,x,cond):
        scale,bias=self.cond(cond).chunk(2,dim=-1)
        return x+self.net(self.norm(x)*(scale+1)+bias)

class GoalDiffusionNet(nn.Module):
    def __init__(self,hidden=512,depth=8):
        super().__init__()
        flat_act = ACTION_DIM*ACTION_HORIZON
        cond_dim = 512
        self.obs_emb  = nn.Sequential(
            nn.Linear(OBS_IN,256),nn.Mish(),nn.Linear(256,256),nn.Mish(),nn.Linear(256,256))
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(128),nn.Linear(128,256),nn.Mish(),nn.Linear(256,256))
        self.cond_proj = nn.Sequential(
            nn.Linear(512,cond_dim),nn.Mish(),nn.Linear(cond_dim,cond_dim))
        self.in_proj  = nn.Linear(flat_act,hidden)
        self.blocks   = nn.ModuleList([ResBlock(hidden,cond_dim) for _ in range(depth)])
        self.out_proj = nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,flat_act))
    def forward(self,noisy,timestep,obs_goal):
        B=noisy.shape[0]; x=noisy.reshape(B,-1)
        cond=self.cond_proj(torch.cat([self.obs_emb(obs_goal),
                                        self.time_emb(timestep)],dim=-1))
        x=self.in_proj(x)
        for blk in self.blocks: x=blk(x,cond)
        return self.out_proj(x).reshape(B,ACTION_HORIZON,ACTION_DIM)

# ── Train ─────────────────────────────────────────────────────────
def train():
    good_eps, ep_markers, ep_boxes, pos, grip, starts, ends = \
        find_successful_episodes(ZARR_PATH)

    z     = zarr.open(ZARR_PATH,'r')
    rot   = z['data']['robot0_eef_rot_axis_angle'][:]
    gripw = z['data']['robot0_gripper_width'][:]
    all_states = np.concatenate([pos,rot,gripw],axis=-1).astype(np.float32)

    goals = np.array([[ep_markers[i][0],ep_markers[i][1],
                        ep_boxes[i][0],  ep_boxes[i][1]]
                       for i in range(len(ends))],dtype=np.float32)

    # Normalizers built on clean episodes only
    clean_states = np.concatenate([all_states[int(starts[i]):int(ends[i])]
                                    for i in good_eps])
    clean_goals  = goals[good_eps]
    obs_norm  = Normalizer(clean_states)
    act_norm  = Normalizer(clean_states)
    goal_norm = Normalizer(clean_goals)
    obs_norm.save(f'{SAVE_DIR}/obs_normalizer.pt')
    act_norm.save(f'{SAVE_DIR}/act_normalizer.pt')
    goal_norm.save(f'{SAVE_DIR}/goal_normalizer.pt')

    dataset = CleanDataset(ZARR_PATH, obs_norm, act_norm, goal_norm,
                           good_eps, ep_markers, ep_boxes)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=2, pin_memory=True, drop_last=True)

    model       = GoalDiffusionNet().to(DEVICE)
    optimizer   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched_lr    = torch.optim.lr_scheduler.CosineAnnealingLR(
                      optimizer, T_max=EPOCHS*len(loader))
    noise_sched = DDPMScheduler(num_train_timesteps=NUM_DIFFUSION_STEPS,
                                 beta_schedule='squaredcos_cap_v2',
                                 clip_sample=True, prediction_type='epsilon')

    total = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total/1e6:.1f}M params")
    print(f"Training {EPOCHS} epochs on {len(good_eps)} CLEAN episodes\n")

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
            torch.save({'epoch':epoch,'loss':avg,'model_state':model.state_dict(),
                        'good_eps':good_eps},
                       f'{SAVE_DIR}/best_model.pt')

    print(f"\n✅ Done! Best loss: {best_loss:.5f}")
    print(f"Checkpoints saved to {SAVE_DIR}/")
    print(f"\nNow run: python goalinfrencediff.py")

if __name__=='__main__':
    train()