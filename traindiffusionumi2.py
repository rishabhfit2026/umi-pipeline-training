"""
Visualize UMI diffusion policy rollout in SAPIEN simulator
Maps real UMI trajectories onto myarm_m750 robot

conda activate maniskill2
python sapien_umi_viz.py
"""

import sapien, numpy as np, torch, zarr, math
from scipy.spatial.transform import Rotation
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

# ── Config ────────────────────────────────────────────────────────
ZARR_PATH = '/home/rishabh/Downloads/umi-pipeline-training/replay_buffer.zarr'
SAVE_DIR  = './checkpoints_umi'
URDF      = '/home/rishabh/Downloads/myarm_m750_fixed.urdf'
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'

OBS_HORIZON    = 4
ACTION_HORIZON = 16
ACTION_DIM     = 7
STATE_DIM      = 7
GOAL_DIM       = 12
OBS_IN         = STATE_DIM * OBS_HORIZON + GOAL_DIM  # 40

# ── Model ─────────────────────────────────────────────────────────
class Normalizer:
    def normalize(self, x):
        return 2.0*(x - self.min.to(x.device))/self.scale.to(x.device) - 1.0
    def denormalize(self, x):
        return (x+1.0)/2.0*self.scale.to(x.device) + self.min.to(x.device)
    @classmethod
    def load(cls, path):
        n=cls.__new__(cls); d=torch.load(path, map_location='cpu')
        n.min,n.max,n.scale=d['min'],d['max'],d['scale']; return n

class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim): super().__init__(); self.dim=dim
    def forward(self, t):
        half=self.dim//2
        emb=math.log(10000)/(half-1)
        emb=torch.exp(torch.arange(half,device=t.device)*-emb)
        emb=t.float()[:,None]*emb[None,:]
        return torch.cat([emb.sin(),emb.cos()],dim=-1)

class ResBlock(torch.nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.net  = torch.nn.Sequential(torch.nn.Linear(dim,dim),torch.nn.Mish(),torch.nn.Linear(dim,dim))
        self.cond = torch.nn.Linear(cond_dim, dim*2)
        self.norm = torch.nn.LayerNorm(dim)
    def forward(self, x, cond):
        scale,bias = self.cond(cond).chunk(2,dim=-1)
        return x + self.net(self.norm(x)*(scale+1)+bias)

class UMIDiffusionNet(torch.nn.Module):
    def __init__(self, hidden=512, depth=8):
        super().__init__()
        flat_act = ACTION_DIM * ACTION_HORIZON
        cond_dim = 512
        self.obs_emb   = torch.nn.Sequential(
            torch.nn.Linear(OBS_IN,256),torch.nn.Mish(),
            torch.nn.Linear(256,256),torch.nn.Mish(),torch.nn.Linear(256,256))
        self.time_emb  = torch.nn.Sequential(
            SinusoidalPosEmb(128),torch.nn.Linear(128,256),
            torch.nn.Mish(),torch.nn.Linear(256,256))
        self.cond_proj = torch.nn.Sequential(
            torch.nn.Linear(512,cond_dim),torch.nn.Mish(),torch.nn.Linear(cond_dim,cond_dim))
        self.in_proj   = torch.nn.Linear(flat_act, hidden)
        self.blocks    = torch.nn.ModuleList([ResBlock(hidden,cond_dim) for _ in range(depth)])
        self.out_proj  = torch.nn.Sequential(torch.nn.LayerNorm(hidden),torch.nn.Linear(hidden,flat_act))
    def forward(self, noisy, timestep, obs_goal):
        B=noisy.shape[0]; x=noisy.reshape(B,-1)
        cond=self.cond_proj(torch.cat([self.obs_emb(obs_goal),self.time_emb(timestep)],dim=-1))
        x=self.in_proj(x)
        for blk in self.blocks: x=blk(x,cond)
        return self.out_proj(x).reshape(B,ACTION_HORIZON,ACTION_DIM)

# ── Load model ────────────────────────────────────────────────────
ckpt = torch.load(f'{SAVE_DIR}/best_model.pt', map_location=DEVICE)
model = UMIDiffusionNet().to(DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()
obs_norm  = Normalizer.load(f'{SAVE_DIR}/obs_normalizer.pt')
act_norm  = Normalizer.load(f'{SAVE_DIR}/act_normalizer.pt')
goal_norm = Normalizer.load(f'{SAVE_DIR}/goal_normalizer.pt')
sched = DDIMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2',
                      clip_sample=True, prediction_type='epsilon')
sched.set_timesteps(16)
print(f"✅ Model loaded — epoch {ckpt['epoch']}, loss={ckpt['loss']:.5f}")

# ── Load UMI data ─────────────────────────────────────────────────
z           = zarr.open(ZARR_PATH, 'r')
pos         = z['data']['robot0_eef_pos'][:]
rot         = z['data']['robot0_eef_rot_axis_angle'][:]
grip        = z['data']['robot0_gripper_width'][:]
start_poses = z['data']['robot0_demo_start_pose'][:]
end_poses   = z['data']['robot0_demo_end_pose'][:]
ends        = z['meta']['episode_ends'][:]
starts      = np.concatenate([[0], ends[:-1]])
states      = np.concatenate([pos, rot, grip], axis=-1).astype(np.float32)

# ── Analyze UMI coordinate ranges ─────────────────────────────────
print(f"\nUMI coordinate ranges:")
print(f"  X: [{pos[:,0].min():.3f}, {pos[:,0].max():.3f}]")
print(f"  Y: [{pos[:,1].min():.3f}, {pos[:,1].max():.3f}]")
print(f"  Z: [{pos[:,2].min():.3f}, {pos[:,2].max():.3f}]")
print(f"  Gripper: [{grip.min():.4f}, {grip.max():.4f}]")

# UMI workspace center
umi_cx = (pos[:,0].min() + pos[:,0].max()) / 2
umi_cy = (pos[:,1].min() + pos[:,1].max()) / 2
umi_cz = (pos[:,2].min() + pos[:,2].max()) / 2
umi_rx = (pos[:,0].max() - pos[:,0].min()) / 2
umi_ry = (pos[:,1].max() - pos[:,1].min()) / 2
umi_rz = (pos[:,2].max() - pos[:,2].min()) / 2
print(f"\nUMI workspace center: ({umi_cx:.3f}, {umi_cy:.3f}, {umi_cz:.3f})")
print(f"UMI workspace radius: x={umi_rx:.3f} y={umi_ry:.3f} z={umi_rz:.3f}")

# ── SAPIEN Scene ───────────────────────────────────────────────────
scene = sapien.Scene()
scene.set_timestep(1/240)
scene.set_ambient_light([0.6,0.6,0.6])
scene.add_directional_light([0,1,-1],[1.0,0.95,0.85])
scene.add_ground(altitude=0)

loader = scene.create_urdf_loader(); loader.fix_root_link=True
robot  = loader.load(URDF); robot.set_pose(sapien.Pose(p=[0,0,0]))
joints = robot.get_active_joints(); N=len(joints)
links  = {l.name:l for l in robot.get_links()}
ee     = links['gripper']
ee_idx = ee.get_index(); pm=robot.create_pinocchio_model()
for jt in joints: jt.set_drive_property(stiffness=30000, damping=3000)

q0=np.zeros(N); q0[1]=-0.3; q0[2]=0.5
robot.set_qpos(q0)
for i,jt in enumerate(joints): jt.set_drive_target(float(q0[i]))
for _ in range(300): scene.step()
real_ee = np.array(ee.get_entity_pose().p)
TX, TY, TZ = real_ee[0], real_ee[1], real_ee[2]
print(f"\nSAPien EE home: ({TX:.3f},{TY:.3f},{TZ:.3f})")

# ── Coordinate mapping: UMI → SAPIEN ──────────────────────────────
# Normalize UMI coords to [-0.5, 0.5] then scale to SAPIEN workspace
SAPIEN_SCALE = 0.25   # how much of the SAPIEN workspace to use
SAPIEN_Z_MIN = 0.08   # never go below table
SAPIEN_Z_MAX = 0.75   # max height

def umi_to_sapien(umi_xyz):
    """Map UMI EEF position to SAPIEN robot workspace."""
    nx = (umi_xyz[0] - umi_cx) / (umi_rx + 1e-6)  # [-1, 1]
    ny = (umi_xyz[1] - umi_cy) / (umi_ry + 1e-6)
    nz = (umi_xyz[2] - umi_cz) / (umi_rz + 1e-6)

    sx = TX + nx * SAPIEN_SCALE
    sy = TY + ny * SAPIEN_SCALE
    sz = TZ + nz * SAPIEN_SCALE * 0.5  # less z range

    sx = np.clip(sx, TX - 0.12, TX + 0.12)
    sy = np.clip(sy, TY - 0.15, TY + 0.15)
    sz = np.clip(sz, SAPIEN_Z_MIN, SAPIEN_Z_MAX)
    return np.array([sx, sy, sz])

def umi_grip_to_sapien(umi_grip):
    """Map UMI gripper [0.0065, 0.0438] → SAPIEN [0.0, 0.0345]"""
    g_min, g_max = 0.0065, 0.0438
    t = (umi_grip - g_min) / (g_max - g_min + 1e-6)
    return float(np.clip(t * 0.0345, 0.0, 0.0345))

# ── IK solver ─────────────────────────────────────────────────────
def solve_ik(xyz, grip_val):
    r=Rotation.from_euler('xyz',[np.pi,0,0]); qv=r.as_quat()
    pose=sapien.Pose(p=list(xyz),q=[qv[3],qv[0],qv[1],qv[2]])
    mask=np.ones(N,dtype=np.int32); mask[6:]=0
    qr,ok,_=pm.compute_inverse_kinematics(ee_idx,pose,
        initial_qpos=robot.get_qpos().astype(np.float64),
        active_qmask=mask,max_iterations=500)
    q=np.array(qr)
    if N>=7: q[6]=grip_val
    if N>=8: q[7]=grip_val
    return q, ok

# ── Scene objects ──────────────────────────────────────────────────
def make_box(half, color, pos, static=True, name=""):
    mt=sapien.render.RenderMaterial(); mt.base_color=color
    b=scene.create_actor_builder()
    b.add_box_visual(half_size=half, material=mt)
    b.add_box_collision(half_size=half)
    a=b.build_static(name=name) if static else b.build(name=name)
    a.set_pose(sapien.Pose(p=pos)); return a

make_box([0.30,0.28,0.025],[0.55,0.36,0.18,1.0],[TX,TY,0.025],True,"table")

# EE trail marker (small sphere that follows EE)
mt_pred=sapien.render.RenderMaterial(); mt_pred.base_color=[1.0,0.2,0.2,0.8]
trail_builder=scene.create_actor_builder()
trail_builder.add_sphere_visual(radius=0.008, material=mt_pred)
ee_marker=trail_builder.build_static(name="ee_marker")

# Goal marker
mt_goal=sapien.render.RenderMaterial(); mt_goal.base_color=[1.0,0.8,0.0,1.0]
gb=scene.create_actor_builder()
gb.add_sphere_visual(radius=0.015, material=mt_goal)
goal_marker=gb.build_static(name="goal_marker")

# ── Rollout one episode ────────────────────────────────────────────
def rollout_and_viz(ep_idx):
    s, e = int(starts[ep_idx]), int(ends[ep_idx])

    # Build goal
    if start_poses.shape[0] == len(ends):
        goal_raw = np.concatenate([start_poses[ep_idx],
                                    end_poses[ep_idx]]).astype(np.float32)
    else:
        goal_raw = np.concatenate([start_poses[s],
                                    end_poses[e-1]]).astype(np.float32)

    goal_t = goal_norm.normalize(torch.tensor(goal_raw, dtype=torch.float32))

    # Place goal marker at mapped end position
    goal_sapien = umi_to_sapien(goal_raw[3:6])  # end pose xyz
    goal_marker.set_pose(sapien.Pose(p=goal_sapien.tolist()))

    # Reset robot
    robot.set_qpos(q0)
    for i,jt in enumerate(joints): jt.set_drive_target(float(q0[i]))
    for _ in range(100): scene.step()

    # Move robot to start position matching UMI start
    start_sapien = umi_to_sapien(states[s, :3])
    q_start, ok = solve_ik(start_sapien, 0.0345)
    if ok:
        q_cur = robot.get_qpos().copy()
        for step in range(60):
            t_s = (step+1)/60; sm = t_s*t_s*(3-2*t_s)
            qi  = q_cur + sm*(q_start - q_cur)
            for j,jt in enumerate(joints): jt.set_drive_target(float(qi[j]))
            for _ in range(2): scene.step()
            scene.update_render(); viewer.render()

    print(f"\nEp {ep_idx+1} | frames={e-s} | "
          f"start=({states[s,0]:.2f},{states[s,1]:.2f},{states[s,2]:.2f}) "
          f"→ sapien=({start_sapien[0]:.3f},{start_sapien[1]:.3f},{start_sapien[2]:.3f})")

    # ── Diffusion rollout ──────────────────────────────────────────
    EXEC_H = 6
    t_frame = s + OBS_HORIZON - 1

    while t_frame < e - ACTION_HORIZON and not viewer.closed:
        # Build obs from ACTUAL UMI data (teacher forcing for first pass)
        # This shows what the model predicts given real observations
        obs_parts = [obs_norm.normalize(
                         torch.tensor(states[t_frame-OBS_HORIZON+1+i],
                                      dtype=torch.float32))
                     for i in range(OBS_HORIZON)]
        obs_goal = torch.cat(obs_parts + [goal_t]).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            noisy = torch.randn(1, ACTION_HORIZON, ACTION_DIM, device=DEVICE)
            for ts in sched.timesteps:
                tst   = torch.tensor([ts], device=DEVICE).long()
                noisy = sched.step(model(noisy, tst, obs_goal), ts, noisy).prev_sample
            pred = act_norm.denormalize(noisy[0]).cpu().numpy()

        # Execute predicted actions on SAPIEN robot
        for i in range(EXEC_H):
            if viewer.closed: break
            pred_xyz  = pred[i, :3]
            pred_grip = pred[i, 6]

            # Map to SAPIEN
            sapien_xyz  = umi_to_sapien(pred_xyz)
            sapien_grip = umi_grip_to_sapien(pred_grip)

            # IK
            q_tgt, ok = solve_ik(sapien_xyz, sapien_grip)
            if ok:
                q_cur = robot.get_qpos().copy()
                for j,jt in enumerate(joints):
                    jt.set_drive_target(float(q_tgt[j]))

            # update EE trail marker
            ee_pos = np.array(ee.get_entity_pose().p)
            ee_marker.set_pose(sapien.Pose(p=ee_pos.tolist()))

            for _ in range(4): scene.step()
            scene.update_render(); viewer.render()

        t_frame += EXEC_H

    # Brief pause between episodes
    for _ in range(120): scene.update_render(); viewer.render()

# ── Viewer + main loop ────────────────────────────────────────────
viewer = scene.create_viewer()
viewer.set_camera_xyz(TX+0.40, TY-0.40, 0.70)
viewer.set_camera_rpy(0, -0.40, 0.55)

# Use held-out test episodes (last 20%)
n_test_start = int(len(ends) * 0.8)
test_episodes = list(range(n_test_start, len(ends)))

print(f"\n{'='*55}")
print(f" UMI DIFFUSION POLICY IN SAPIEN")
print(f" {len(test_episodes)} test episodes starting from ep {n_test_start+1}")
print(f" Yellow sphere = goal end position")
print(f" Red dot = current EE position")
print(f"{'='*55}\n")

ep_ptr = 0
while not viewer.closed:
    ep_idx = test_episodes[ep_ptr % len(test_episodes)]
    rollout_and_viz(ep_idx)
    ep_ptr += 1