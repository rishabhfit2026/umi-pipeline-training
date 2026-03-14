"""
Train MLP with marker+box position as input
Input:  ee_pos(3) + ee_rot(3) + gripper(1) + marker_pos(3) + box_pos(3) = 13 dims
Output: next 8 steps of ee_pos+rot+gripper = 56 dims
Run: source umi_env/bin/activate && python step3_poseonly2.py
"""
import torch, zarr, numpy as np, os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ZARR       = '/home/rishabh/Downloads/umi-pipeline-training/outputs/sim_replay_buffer3.zarr'
OUT        = '/home/rishabh/Downloads/umi-pipeline-training/outputs/poseonly_model2'
DEVICE     = 'cuda:0'
PRED_STEPS = 8
N_DOFS     = 7
IN_DIMS    = 13   # ee(7) + marker(3) + box(3)
BATCH      = 512
MAX_STEPS  = 5000
LR         = 3e-4
os.makedirs(OUT, exist_ok=True)

print("Loading zarr data...")
z    = zarr.open(ZARR, 'r')
pos  = z['data']['robot0_eef_pos'][:]
rot  = z['data']['robot0_eef_rot_axis_angle'][:]
grip = z['data']['robot0_gripper_width'][:]
ends = z['meta']['episode_ends'][:]
acts = np.concatenate([pos, rot, grip], axis=1)  # (N,7)

print(f"Episodes: {len(ends)}  Frames: {len(acts)}")
print(f"z range: {pos[:,2].min():.3f} → {pos[:,2].max():.3f}")

# Reconstruct marker and box positions per episode
# During data gen: marker starts at (mx,my,0.076), box at (bx,by,0.075)
# After grasp, marker follows EE. We reconstruct from the zarr episode structure.
# Simpler: use the EE position at grasp time (min z frame) as marker pos
# and episode-end EE xy as box pos approximation
starts = np.concatenate([[0], ends[:-1]])

marker_pos_all = np.zeros((len(acts), 3), dtype=np.float32)
box_pos_all    = np.zeros((len(acts), 3), dtype=np.float32)

rng = np.random.default_rng(42)  # same seed as data generation
TX, TY = 0.251, 0.000  # EE home xy

for ep_i, (s, e) in enumerate(zip(starts, ends)):
    # Reconstruct marker/box positions using same RNG as step10
    mx = TX + rng.uniform(-0.05, 0.05)
    my = TY + rng.uniform(-0.10, 0.10)
    bx = TX + rng.uniform(-0.05, 0.05)
    by = TY + rng.uniform( 0.05, 0.15)
    while abs(mx-bx)<0.04 and abs(my-by)<0.04:
        bx = TX + rng.uniform(-0.05, 0.05)
        by = TY + rng.uniform( 0.05, 0.15)
    marker_pos_all[s:e] = [mx, my, 0.076]
    box_pos_all[s:e]    = [bx, by, 0.075]

print(f"Marker pos range: x={marker_pos_all[:,0].min():.3f}→{marker_pos_all[:,0].max():.3f}")

# Normalize
obs_all  = np.concatenate([acts, marker_pos_all, box_pos_all], axis=1)  # (N,13)
obs_mean = obs_all.mean(0); obs_std = obs_all.std(0) + 1e-6
act_mean = acts.mean(0);    act_std  = acts.std(0)  + 1e-6

np.save(f'{OUT}/obs_mean.npy', obs_mean)
np.save(f'{OUT}/obs_std.npy',  obs_std)
np.save(f'{OUT}/act_mean.npy', act_mean)
np.save(f'{OUT}/act_std.npy',  act_std)

obs_n  = (obs_all - obs_mean) / obs_std
acts_n = (acts    - act_mean) / act_std

class PoseDataset(Dataset):
    def __init__(self):
        self.samples = []
        for s, e in zip(starts, ends):
            for i in range(s, e - PRED_STEPS - 1):
                ob  = obs_n[i]
                tgt = acts_n[i+1:i+1+PRED_STEPS].flatten()
                self.samples.append((ob.astype(np.float32),
                                     tgt.astype(np.float32)))
        print(f"Dataset: {len(self.samples)} samples")
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        o,t = self.samples[i]
        return torch.tensor(o), torch.tensor(t)

dataset = PoseDataset()
loader  = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=4)

class PoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IN_DIMS, 512), nn.ReLU(),
            nn.Linear(512, 512),     nn.ReLU(),
            nn.Linear(512, 512),     nn.ReLU(),
            nn.Linear(512, 512),     nn.ReLU(),
            nn.Linear(512, 256),     nn.ReLU(),
            nn.Linear(256, PRED_STEPS * N_DOFS)
        )
    def forward(self, x): return self.net(x)

model = PoseNet().to(DEVICE)
opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sch   = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=MAX_STEPS)

print(f"\nTraining {MAX_STEPS} steps...")
model.train(); step=0; rloss=0.0

for epoch in range(1000):
    for obs_b, tgt_b in loader:
        if step >= MAX_STEPS: break
        obs_b, tgt_b = obs_b.to(DEVICE), tgt_b.to(DEVICE)
        loss = nn.functional.mse_loss(model(obs_b), tgt_b)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        rloss += loss.item(); step += 1
        if step % 200 == 0:
            print(f"  step {step:5d}/{MAX_STEPS}  loss={rloss/200:.5f}")
            rloss = 0.0
        if step % 1000 == 0:
            torch.save(model.state_dict(), f'{OUT}/model_{step}.pt')
    if step >= MAX_STEPS: break

torch.save(model.state_dict(), f'{OUT}/model_final.pt')
print(f"\n✅ Done! Saved to {OUT}/model_final.pt")