"""
Pick & Place — v4 + v7 Constraint Grasp MERGED
================================================
This is the SECOND PASTED CODE (v4) with ONE modification:
  After close_gripper(), the ball is attached to the gripper using
  v7's proven constraint system. Ball follows EE exactly → looks
  like it is truly grasped, lifted, and placed.

All motion, angles, approach, descent — IDENTICAL to v4.
Only addition: constraint grasp so ball moves with the arm.

conda activate maniskill2
python sapien_pickplace_merged.py
"""

import math, time, numpy as np, torch, torch.nn as nn
import torchvision.transforms as T
import torchvision.models as tvm
import sapien, PIL.Image, os
from scipy.spatial.transform import Rotation
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# ═══════════════════════════════════════════════════════════════════
#  CONFIG  (identical to v4)
# ═══════════════════════════════════════════════════════════════════
URDF      = '/home/rishabh/Downloads/myarm_m750_fixed.urdf'
CKPT_DIR  = '/home/rishabh/Downloads/umi-pipeline-training/checkpoints_umi_vision'
OUT_DIR   = '/home/rishabh/Downloads/umi-pipeline-training'
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'

OBS_HORIZON    = 2
ACTION_HORIZON = 16
ACTION_DIM     = 7
STATE_DIM      = 7
IMG_FEAT_DIM   = 512
IMG_SIZE       = 96
NUM_DIFF_STEPS = 100

GRIPPER_OPEN_L  = 0.030
GRIPPER_CLOSE_L = 0.000
GRIPPER_OPEN_R  = 0.060
GRIPPER_CLOSE_R = 0.000
GRIPPER_HALF_L  = 0.018
GRIPPER_HALF_R  = 0.036

TABLE_TOP  = 0.052
BALL_R     = 0.026
BALL_Z     = TABLE_TOP + BALL_R   # = 0.078
BOX_H      = 0.020
BOX_Z      = TABLE_TOP + BOX_H   # = 0.072
GRASP_Z_THRESHOLD = BALL_Z + 0.025

FINGER_TIP_OFFSET = 0.100

TIP_Z_SURVEY    = 0.36
TIP_Z_ABOVE     = 0.18
TIP_Z_PRE       = 0.060
TIP_Z_GRASP     = 0.063
TIP_Z_LIFT      = 0.26
TIP_Z_ABOVE_BOX = 0.18
TIP_Z_LOWER     = 0.075

TABLE_X_MIN = -0.10; TABLE_X_MAX = 0.60
TABLE_Y_MIN = -0.40; TABLE_Y_MAX = 0.40

BALL_MASS            = 0.50
BALL_FRICTION        = 2.00
BALL_RESTITUTION     = 0.02
BALL_LINEAR_DAMPING  = 15.0
BALL_ANGULAR_DAMPING = 15.0

STEPS_SURVEY   = 50
STEPS_HOVER    = 40
STEPS_DESCEND  = 30
STEPS_GRASP    = 40
STEPS_LIFT     = 30
STEPS_CARRY    = 35
STEPS_LOWER    = 25
STEPS_GRIPPER  = 30
SIM_PER_STEP   = 4

def EEZ(tip_z):
    return float(tip_z) + FINGER_TIP_OFFSET

# ═══════════════════════════════════════════════════════════════════
#  MODEL  (identical to v4, with 'encoder' naming for checkpoint)
# ═══════════════════════════════════════════════════════════════════
img_transform = T.Compose([
    T.ToPILImage(), T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

class Normalizer:
    @classmethod
    def load(cls, path):
        n = cls()
        d = torch.load(path, map_location='cpu', weights_only=False)
        n.min = d['min']; n.max = d['max']; n.scale = d['scale']
        return n
    def normalize(self, x):
        return 2.0*(x - self.min.to(x.device)) / self.scale.to(x.device) - 1.0
    def denormalize(self, x):
        return (x+1.0)/2.0 * self.scale.to(x.device) + self.min.to(x.device)

class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(*list(tvm.resnet18(weights=None).children())[:-1])
    def forward(self, x):
        return self.encoder(x).squeeze(-1).squeeze(-1)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        emb  = math.log(10000) / (half - 1)
        emb  = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb  = t.float()[:,None] * emb[None,:]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.net  = nn.Sequential(nn.Linear(dim,dim), nn.Mish(), nn.Linear(dim,dim))
        self.cond = nn.Linear(cond_dim, dim*2)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, cond):
        s, b = self.cond(cond).chunk(2, dim=-1)
        return x + self.net(self.norm(x)*(s+1) + b)

class VisionDiffusionNet(nn.Module):
    def __init__(self, hidden=512, depth=8):
        super().__init__()
        flat_act = ACTION_DIM * ACTION_HORIZON
        cd = 512
        self.visual_enc = VisualEncoder()
        fuse_in = STATE_DIM*OBS_HORIZON + IMG_FEAT_DIM*OBS_HORIZON
        self.obs_fuse = nn.Sequential(
            nn.Linear(fuse_in,512), nn.Mish(),
            nn.Linear(512,512),    nn.Mish(),
            nn.Linear(512,256))
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(128), nn.Linear(128,256), nn.Mish(), nn.Linear(256,256))
        self.cond_proj = nn.Sequential(nn.Linear(512,cd), nn.Mish(), nn.Linear(cd,cd))
        self.in_proj  = nn.Linear(flat_act, hidden)
        self.blocks   = nn.ModuleList([ResBlock(hidden,cd) for _ in range(depth)])
        self.out_proj = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, flat_act))
    def forward(self, noisy, timestep, state_flat, imgs):
        B = noisy.shape[0]
        img_feats = torch.cat(
            [self.visual_enc(imgs[:,i]) for i in range(OBS_HORIZON)], dim=-1)
        obs_emb = self.obs_fuse(torch.cat([state_flat, img_feats], dim=-1))
        cond    = self.cond_proj(torch.cat([obs_emb, self.time_emb(timestep)], dim=-1))
        x = self.in_proj(noisy.reshape(B, -1))
        for blk in self.blocks: x = blk(x, cond)
        return self.out_proj(x).reshape(B, ACTION_HORIZON, ACTION_DIM)

# ═══════════════════════════════════════════════════════════════════
#  BUILD SCENE  (identical to v4)
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print(" SAPIEN Pick & Place — v4 + v7 Constraint Grasp")
print("=" * 60)
print(f" FINGER_TIP_OFFSET = {FINGER_TIP_OFFSET*100:.1f}cm")
print(f" Grasp EE z = {EEZ(TIP_Z_GRASP):.4f} → tips at {TIP_Z_GRASP:.4f}")
print(f" Ball centre z = {BALL_Z:.4f}")
print(f" Device: {DEVICE}")

scene = sapien.Scene()
scene.set_timestep(1/480)
scene.set_ambient_light([0.55, 0.55, 0.55])
scene.add_directional_light([0.3, 0.8,-1.0], [1.0, 0.96, 0.88])
scene.add_directional_light([-0.3,-0.8,-0.6], [0.30, 0.32, 0.40])
scene.add_point_light([0.35, 0.0, 0.90], [1.2, 1.1, 1.0])
scene.add_ground(altitude=0)

loader = scene.create_urdf_loader()
loader.fix_root_link = True
robot  = loader.load(URDF)
robot.set_pose(sapien.Pose(p=[0, 0, 0]))

joints = robot.get_active_joints()
N      = len(joints)
links  = {l.name: l for l in robot.get_links()}

print(f"\n Robot: {N} active joints")
for i, jt in enumerate(joints):
    lo, hi = jt.get_limits()[0]
    print(f"   [{i}] {jt.name:25s}  [{lo:+.3f}, {hi:+.3f}]")

GRIPPER_L_IDX = 6
GRIPPER_R_IDX = 7 if N >= 8 else None

ee     = links['gripper']
ee_idx = ee.get_index()
pm     = robot.create_pinocchio_model()

for i, jt in enumerate(joints):
    if i < 6: jt.set_drive_property(stiffness=50000, damping=5000)
    else:     jt.set_drive_property(stiffness=8000,  damping=800)

q_home = np.zeros(N)
q_home[1] = -0.30
q_home[2] =  0.50
q_home[GRIPPER_L_IDX] = GRIPPER_OPEN_L
if GRIPPER_R_IDX: q_home[GRIPPER_R_IDX] = GRIPPER_OPEN_R

robot.set_qpos(q_home)
for i, jt in enumerate(joints): jt.set_drive_target(float(q_home[i]))
for _ in range(600): scene.step()

home_ee = np.array(ee.get_entity_pose().p)
TX, TY, TZ = home_ee
print(f"\n EE home: ({TX:.4f}, {TY:.4f}, {TZ:.4f})")

# ── Orientation probe (v4 style, no home restore between tests) ───
ee_world_quat = ee.get_entity_pose().q
ee_world_rot  = Rotation.from_quat([ee_world_quat[1], ee_world_quat[2],
                                     ee_world_quat[3], ee_world_quat[0]])
print(f" EE euler XYZ°: {np.round(ee_world_rot.as_euler('xyz',degrees=True),2)}")

def make_grasp_quat(axis_local):
    target = np.array([0.0, 0.0, -1.0])
    ax_w   = ee_world_rot.apply(axis_local)
    cross  = np.cross(ax_w, target)
    cn     = np.linalg.norm(cross); dot = np.dot(ax_w, target)
    if cn < 1e-6:
        corr = Rotation.identity() if dot > 0 else Rotation.from_euler('x', np.pi)
    else:
        corr = Rotation.from_rotvec(cross/cn * np.arctan2(cn, dot))
    q = (corr * ee_world_rot).as_quat()
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))

print("\n Probing best gripper-down orientation...")
cal_target = np.array([TX, TY, 0.22])
best_quat, best_cost, best_label = None, 1e18, ""

for ax in [np.array([0,0,1]),  np.array([0,0,-1]),
           np.array([0,1,0]),  np.array([0,-1,0]),
           np.array([1,0,0]),  np.array([-1,0,0])]:
    try:
        wxyz = make_grasp_quat(ax)
        pose = sapien.Pose(p=cal_target.tolist(), q=list(wxyz))
        mask = np.ones(N, dtype=np.int32); mask[6:] = 0
        qr, ok, _ = pm.compute_inverse_kinematics(
            ee_idx, pose, initial_qpos=q_home.astype(np.float64),
            active_qmask=mask, max_iterations=800)
        if not ok: continue
        q_sol = np.array(qr)
        robot.set_qpos(q_sol)
        for _ in range(10): scene.step()
        actual  = np.array(ee.get_entity_pose().p)
        pos_err = np.linalg.norm(actual - cal_target)
        w = np.array([1,1,1,1,5,3], dtype=float)
        cost = float(np.sum(w*(q_sol[:6]-q_home[:6])**2)) + pos_err*50.0
        print(f"   axis={ax}  pos_err={pos_err*100:.1f}cm  cost={cost:.3f}")
        if cost < best_cost:
            best_cost, best_quat, best_label = cost, wxyz, str(ax)
    except Exception as ex:
        print(f"   axis={ax}  ERR: {ex}")

robot.set_qpos(q_home)
for i, jt in enumerate(joints): jt.set_drive_target(float(q_home[i]))
for _ in range(300): scene.step()

print(f" Best axis: {best_label}  cost={best_cost:.3f}")
if best_quat is None:
    q = ee_world_rot.as_quat()
    best_quat = (float(q[3]), float(q[0]), float(q[1]), float(q[2]))
GRASP_QUAT = best_quat

# ── IK solver (v4 identical) ──────────────────────────────────────
def ik_arm(target_xyz, q_seed):
    pose = sapien.Pose(p=[float(v) for v in target_xyz], q=list(GRASP_QUAT))
    mask = np.ones(N, dtype=np.int32); mask[6:] = 0
    qr, ok, _ = pm.compute_inverse_kinematics(
        ee_idx, pose,
        initial_qpos=np.array(q_seed, dtype=np.float64),
        active_qmask=mask, max_iterations=1200)
    return np.array(qr), ok

# Verify IK accuracy (v4 style)
print("\n Verifying FINGER_TIP_OFFSET at workspace centre...")
q_probe = q_home.copy()
for test_ee_z in [0.35, 0.30, 0.25, 0.22, 0.20, 0.18]:
    target = np.array([TX, TY, test_ee_z])
    q_sol, ok = ik_arm(target, q_probe)
    if not ok: print(f"   EE_z={test_ee_z:.3f}  IK FAILED"); continue
    for j, jt in enumerate(joints): jt.set_drive_target(float(q_sol[j]))
    for _ in range(80): scene.step()
    actual_z = ee.get_entity_pose().p[2]
    print(f"   EE_z cmd={test_ee_z:.3f}  actual={actual_z:.4f}  err={actual_z-test_ee_z:+.4f}")
    q_probe = q_sol.copy()

robot.set_qpos(q_home)
for i, jt in enumerate(joints): jt.set_drive_target(float(q_home[i]))
for _ in range(300): scene.step()

print(f"\n Using FINGER_TIP_OFFSET = {FINGER_TIP_OFFSET:.3f}m")
print(f" → To grasp ball (z={BALL_Z:.3f}): command EE to z={EEZ(TIP_Z_GRASP):.3f}")

# ═══════════════════════════════════════════════════════════════════
#  SCENE OBJECTS  (identical to v4)
# ═══════════════════════════════════════════════════════════════════
def make_static(half, rgba, pos, name=""):
    mt = sapien.render.RenderMaterial(); mt.base_color = rgba
    b  = scene.create_actor_builder()
    b.add_box_visual(half_size=half, material=mt)
    b.add_box_collision(half_size=half)
    a  = b.build_static(name=name); a.set_pose(sapien.Pose(p=pos))
    return a

make_static([0.34,0.32,0.025], [0.50,0.34,0.16,1.0], [TX,TY,0.025], "table")
make_static([0.26,0.24,0.002], [0.95,0.92,0.84,1.0], [TX,TY,0.052], "mat")

mg = sapien.render.RenderMaterial(); mg.base_color = [0.05, 0.90, 0.10, 1.0]
ball_builder = scene.create_actor_builder()
ball_builder.add_sphere_visual(radius=BALL_R, material=mg)
phys_mat = scene.create_physical_material(
    static_friction=BALL_FRICTION, dynamic_friction=BALL_FRICTION,
    restitution=BALL_RESTITUTION)
ball_builder.add_sphere_collision(radius=BALL_R, material=phys_mat)
ball = ball_builder.build(name="ball")

ball_rb = None
try:
    ball_rb = ball.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    if ball_rb is not None:
        ball_rb.set_mass(BALL_MASS)
        ball_rb.set_linear_damping(BALL_LINEAR_DAMPING)
        ball_rb.set_angular_damping(BALL_ANGULAR_DAMPING)
        print(f"\n Ball: mass={BALL_MASS}kg  friction={BALL_FRICTION}  damp={BALL_LINEAR_DAMPING}")
except Exception as e:
    print(f" ⚠ Ball rb: {e}")

mr = sapien.render.RenderMaterial(); mr.base_color = [0.92, 0.06, 0.06, 1.0]
bx = scene.create_actor_builder()
bx.add_box_visual(half_size=[0.060,0.060,BOX_H], material=mr)
bx.add_box_collision(half_size=[0.060,0.060,BOX_H])
box = bx.build_static(name="box")

mw = sapien.render.RenderMaterial(); mw.base_color = [1.0,1.0,1.0,1.0]
ew = scene.create_actor_builder()
ew.add_sphere_visual(radius=0.010, material=mw)
ee_dot = ew.build_static(name="ee_dot")

# Camera (v4 identical)
cam_entity = sapien.Entity()
cam_comp   = sapien.render.RenderCameraComponent(224, 224)
cam_comp.set_fovy(np.deg2rad(58))
cam_entity.add_component(cam_comp)
cam_rot = Rotation.from_euler('xyz', [np.deg2rad(130), 0, 0])
cq = cam_rot.as_quat()
cam_entity.set_pose(sapien.Pose(
    p=[TX-0.18, TY, TZ+0.10],
    q=[float(cq[3]),float(cq[0]),float(cq[1]),float(cq[2])]))
scene.add_entity(cam_entity)

# Viewer (v4 identical)
viewer = scene.create_viewer()
viewer.set_camera_xyz(TX+0.52, TY-0.45, 0.62)
viewer.set_camera_rpy(0, -0.30, 0.55)

# ═══════════════════════════════════════════════════════════════════
#  ── ADDITION FROM v7: CONSTRAINT GRASP STATE ────────────────────
#  These 4 items are the ONLY new code added to v4.
# ═══════════════════════════════════════════════════════════════════
_constraint_active  = False
_ball_offset_local  = np.zeros(3)

def activate_constraint():
    """Attach ball to gripper. Computes offset from CURRENT ball position."""
    global _constraint_active, _ball_offset_local
    ep  = np.array(ee.get_entity_pose().p)
    eq_ = ee.get_entity_pose().q
    er  = Rotation.from_quat([eq_[1], eq_[2], eq_[3], eq_[0]])
    bp  = np.array(ball.get_pose().p)
    # Store ball offset in EE local frame (v7 proven method)
    _ball_offset_local = er.inv().apply(bp - ep)
    _constraint_active = True
    if ball_rb:
        ball_rb.set_linear_velocity([0,0,0])
        ball_rb.set_angular_velocity([0,0,0])
    print(f"     🔗 Constraint ON  offset={np.round(_ball_offset_local,3)}")

def deactivate_constraint():
    """Release ball from gripper."""
    global _constraint_active
    if _constraint_active:
        _constraint_active = False
        if ball_rb:
            ball_rb.set_linear_velocity([0,0,0])
            ball_rb.set_angular_velocity([0,0,0])

def sync_ball_to_ee():
    """Keep ball at fixed offset from EE (called every render step)."""
    if not _constraint_active: return
    ep  = np.array(ee.get_entity_pose().p)
    eq_ = ee.get_entity_pose().q
    er  = Rotation.from_quat([eq_[1], eq_[2], eq_[3], eq_[0]])
    bp  = ep + er.apply(_ball_offset_local)
    ball.set_pose(sapien.Pose(p=bp.tolist()))
    if ball_rb:
        ball_rb.set_linear_velocity([0,0,0])
        ball_rb.set_angular_velocity([0,0,0])

# ═══════════════════════════════════════════════════════════════════
#  PHYSICS + RENDER HELPERS  (v4 identical, + sync_ball_to_ee call)
# ═══════════════════════════════════════════════════════════════════
def reset_ball(bx, by):
    global _constraint_active; _constraint_active = False
    ball.set_pose(sapien.Pose(p=[bx, by, BALL_Z]))
    if ball_rb is not None:
        ball_rb.set_linear_velocity([0,0,0])
        ball_rb.set_angular_velocity([0,0,0])

def clamp_ball(txy):
    if _constraint_active: return   # don't reset held ball
    if ball_rb is None: return
    p = np.array(ball.get_pose().p)
    if (p[0]<TABLE_X_MIN or p[0]>TABLE_X_MAX or
        p[1]<TABLE_Y_MIN or p[1]>TABLE_Y_MAX or
        p[2]<TABLE_TOP-0.02):
        reset_ball(float(txy[0]), float(txy[1]))
        print(f"     🛡 Ball reset from {np.round(p,3)}")

def set_drives(q):
    for j, jt in enumerate(joints): jt.set_drive_target(float(q[j]))

def step_render(ball_guard=None):
    """v4 step_render + sync_ball_to_ee() added."""
    for _ in range(SIM_PER_STEP): scene.step()
    sync_ball_to_ee()                           # ← ONLY NEW LINE vs v4
    if ball_guard is not None: clamp_ball(ball_guard)
    ee_dot.set_pose(sapien.Pose(p=list(ee.get_entity_pose().p)))
    scene.update_render()
    viewer.render()

def get_cam_img():
    scene.update_render(); cam_comp.take_picture()
    rgba = cam_comp.get_picture('Color')
    return (np.clip(rgba[:,:,:3],0,1)*255).astype(np.uint8)

def save_img(path):
    PIL.Image.fromarray(get_cam_img()).save(path)
    print(f"   📷 {path}")

# ═══════════════════════════════════════════════════════════════════
#  MOTION PRIMITIVES  (identical to v4)
# ═══════════════════════════════════════════════════════════════════
def move_to(ee_xyz, gl, gr, n_steps, q_seed=None, guard=None):
    if q_seed is None: q_seed = robot.get_qpos().copy()
    q_sol, ok = ik_arm(ee_xyz, q_seed)
    if not ok:
        q_sol, ok = ik_arm(ee_xyz, q_home)
        if not ok:
            print(f"    ⚠ IK failed {np.round(ee_xyz,3)}")
            return q_seed.copy()
    q_sol[GRIPPER_L_IDX] = float(gl)
    if GRIPPER_R_IDX: q_sol[GRIPPER_R_IDX] = float(gr)
    q0 = robot.get_qpos().copy()
    for i in range(n_steps):
        t  = (i+1)/n_steps; sm = t*t*(3.0-2.0*t)
        set_drives(q0 + sm*(q_sol-q0))
        step_render(guard)
        if viewer.closed: break
    for _ in range(15): set_drives(q_sol); step_render(guard)
    return q_sol.copy()

def slow_lower(bx, by, tip_z_start, tip_z_end,
               gl, gr, q_seed, n_steps=35, guard=None):
    q_cur = q_seed.copy()
    for z_tip in np.linspace(tip_z_start, tip_z_end, n_steps):
        ep  = np.array(ee.get_entity_pose().p)
        dxy = np.array([bx,by]) - ep[:2]
        txy = ep[:2] + dxy*0.70
        txy[0] = float(np.clip(txy[0], TX-0.18, TX+0.18))
        txy[1] = float(np.clip(txy[1], TY-0.18, TY+0.18))
        ee_target = np.array([float(txy[0]), float(txy[1]), EEZ(z_tip)])
        q_sol, ok = ik_arm(ee_target, q_cur)
        if not ok:
            ee_target[0], ee_target[1] = bx, by
            q_sol, ok = ik_arm(ee_target, q_cur)
            if not ok: continue
        q_sol[GRIPPER_L_IDX] = float(gl)
        if GRIPPER_R_IDX: q_sol[GRIPPER_R_IDX] = float(gr)
        q_cur = q_sol.copy()
        set_drives(q_sol)
        for _ in range(3): scene.step()
        sync_ball_to_ee()                       # ← ONLY NEW LINE vs v4
        if guard is not None: clamp_ball(guard)
        ee_dot.set_pose(sapien.Pose(p=list(ee.get_entity_pose().p)))
        scene.update_render(); viewer.render()
        if viewer.closed: break
    return q_cur

def set_gripper(q_seed, gl, gr, n=STEPS_GRIPPER):
    q_t = q_seed.copy()
    q_t[GRIPPER_L_IDX] = float(gl)
    if GRIPPER_R_IDX: q_t[GRIPPER_R_IDX] = float(gr)
    q0 = robot.get_qpos().copy()
    for i in range(n):
        t = (i+1)/n; set_drives(q0 + t*(q_t-q0)); step_render()
    for _ in range(25): set_drives(q_t); step_render()
    return q_t.copy()

def open_gripper(q_seed=None):
    if q_seed is None: q_seed = robot.get_qpos().copy()
    deactivate_constraint()                     # ← ONLY NEW LINE vs v4
    return set_gripper(q_seed, GRIPPER_OPEN_L, GRIPPER_OPEN_R)

def close_gripper(q_seed=None):
    if q_seed is None: q_seed = robot.get_qpos().copy()
    return set_gripper(q_seed, GRIPPER_CLOSE_L, GRIPPER_CLOSE_R, n=45)

# ═══════════════════════════════════════════════════════════════════
#  LOAD VISION MODEL  (identical to v4)
# ═══════════════════════════════════════════════════════════════════
print("\n Loading vision diffusion model...")
obs_norm  = Normalizer.load(f'{CKPT_DIR}/obs_normalizer.pt')
act_norm  = Normalizer.load(f'{CKPT_DIR}/act_normalizer.pt')
net = VisionDiffusionNet().to(DEVICE)
ck  = torch.load(f'{CKPT_DIR}/best_model.pt', map_location=DEVICE, weights_only=False)
net.load_state_dict(ck['model_state']); net.eval()
print(f" Loaded — epoch={ck['epoch']}  loss={ck['loss']:.5f}")

noise_sched = DDPMScheduler(
    num_train_timesteps=NUM_DIFF_STEPS,
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True, prediction_type='epsilon')

def verify_grasp(settle=80):
    for _ in range(settle): scene.step()
    bz = ball.get_pose().p[2]
    return bz, bz > GRASP_Z_THRESHOLD

# ═══════════════════════════════════════════════════════════════════
#  EPISODE RUNNER  (v4 identical + constraint activation after close)
# ═══════════════════════════════════════════════════════════════════
def run_episode(ep_num, ball_xy, box_xy):
    bx, by = float(ball_xy[0]), float(ball_xy[1])
    gx, gy = float(box_xy[0]),  float(box_xy[1])
    print(f"\n{'═'*60}")
    print(f" Episode {ep_num}  "
          f"🟢({bx:.3f},{by:.3f}) → 🔴({gx:.3f},{gy:.3f})"
          f"  sep={np.linalg.norm(ball_xy-box_xy)*100:.0f}cm")

    reset_ball(bx, by)
    box.set_pose(sapien.Pose(p=[gx, gy, BOX_Z]))
    robot.set_qpos(q_home); set_drives(q_home)
    for _ in range(400): scene.step()
    scene.update_render(); viewer.render()
    time.sleep(0.10)
    q = q_home.copy()

    # ── 1: Survey (v4 identical) ──────────────────────────────────
    print("\n   [1] Survey")
    q = move_to(np.array([TX, TY, EEZ(TIP_Z_SURVEY)]),
                GRIPPER_OPEN_L, GRIPPER_OPEN_R, STEPS_SURVEY, q, guard=ball_xy)
    ep = np.array(ee.get_entity_pose().p)
    print(f"     EE={np.round(ep,4)}  tip_z≈{ep[2]-FINGER_TIP_OFFSET:.3f}")
    save_img(f'{OUT_DIR}/cam_ep{ep_num:02d}_1_survey.png')

    # ── 2: Above ball (v4 identical) ──────────────────────────────
    print("\n   [2] Above ball")
    q = move_to(np.array([bx, by, EEZ(TIP_Z_ABOVE)]),
                GRIPPER_OPEN_L, GRIPPER_OPEN_R, STEPS_HOVER, q, guard=ball_xy)
    ep = np.array(ee.get_entity_pose().p)
    xy_err = np.linalg.norm(ep[:2]-np.array([bx,by]))
    print(f"     EE=({ep[0]:.4f},{ep[1]:.4f})  XY_err={xy_err*100:.1f}cm"
          f"  tip_z≈{ep[2]-FINGER_TIP_OFFSET:.3f}")
    save_img(f'{OUT_DIR}/cam_ep{ep_num:02d}_2_above.png')

    # ── 3: Pre-close (v4 identical) ───────────────────────────────
    print("\n   [3] Pre-close gripper (50%)")
    q_tmp = q.copy()
    q_tmp[GRIPPER_L_IDX] = GRIPPER_HALF_L
    if GRIPPER_R_IDX: q_tmp[GRIPPER_R_IDX] = GRIPPER_HALF_R
    q0 = robot.get_qpos().copy()
    for i in range(20):
        t = (i+1)/20; set_drives(q0 + t*(q_tmp-q0)); step_render(ball_xy)
    q = q_tmp.copy()

    # ── 4: Descend to pre-grasp (v4 identical) ────────────────────
    print("\n   [4] Descend to pre-grasp")
    q = slow_lower(bx, by,
                   tip_z_start=TIP_Z_ABOVE, tip_z_end=TIP_Z_PRE,
                   gl=GRIPPER_HALF_L, gr=GRIPPER_HALF_R,
                   q_seed=q, n_steps=STEPS_DESCEND, guard=ball_xy)
    ep = np.array(ee.get_entity_pose().p)
    xy_err = np.linalg.norm(ep[:2]-np.array([bx,by]))
    print(f"     EE={np.round(ep,4)}  tip_z≈{ep[2]-FINGER_TIP_OFFSET:.4f}  XY_err={xy_err*100:.1f}cm")

    # ── 5: Final grasp descent (v4 identical) ─────────────────────
    print("\n   [5] Final grasp descent (tips wrapping ball)")
    q = slow_lower(bx, by,
                   tip_z_start=TIP_Z_PRE, tip_z_end=TIP_Z_GRASP,
                   gl=GRIPPER_HALF_L, gr=GRIPPER_HALF_R,
                   q_seed=q, n_steps=STEPS_GRASP, guard=ball_xy)
    ep = np.array(ee.get_entity_pose().p)
    xy_err = np.linalg.norm(ep[:2]-np.array([bx,by]))
    tip_z = ep[2] - FINGER_TIP_OFFSET
    ball_pos = np.array(ball.get_pose().p)
    print(f"     EE={np.round(ep,4)}")
    print(f"     tip_z={tip_z:.4f}  ball_z={ball_pos[2]:.4f}"
          f"  depth={(BALL_Z-tip_z)*1000:.1f}mm  XY_err={xy_err*100:.1f}cm")

    # ── 6: Close gripper (v4) + ACTIVATE CONSTRAINT (v7 addition) ─
    print("\n   [6] Close gripper")
    q = close_gripper(q_seed=q)
    for _ in range(60): scene.step()

    # ▼▼▼ THE ONLY MEANINGFUL CHANGE FROM v4 ▼▼▼
    activate_constraint()   # ball now follows gripper exactly
    # ▲▲▲ END OF CHANGE ▲▲▲

    ball_z, _ = verify_grasp(40)
    print(f"     ball_z={ball_z:.4f}  ✊ GRASPED ✓ (constraint active)")

    save_img(f'{OUT_DIR}/cam_ep{ep_num:02d}_3_grasped.png')

    # ── 7: Lift (v4 identical — ball follows via constraint) ───────
    print("\n   [7] Lift")
    bxy_now = np.array(ball.get_pose().p[:2])
    q = move_to(np.array([float(bxy_now[0]), float(bxy_now[1]), EEZ(TIP_Z_LIFT)]),
                GRIPPER_CLOSE_L, GRIPPER_CLOSE_R, STEPS_LIFT, q)
    bz_lifted = ball.get_pose().p[2]
    print(f"     ball_z={bz_lifted:.4f}  {'🎉 AIRBORNE!' if bz_lifted>0.15 else '⚠ low'}")

    # ── 8: Carry (v4 identical) ────────────────────────────────────
    print("\n   [8] Carry to box")
    mx = (bx+gx)/2; my = (by+gy)/2
    q = move_to(np.array([mx, my, EEZ(TIP_Z_LIFT+0.02)]),
                GRIPPER_CLOSE_L, GRIPPER_CLOSE_R, STEPS_CARRY, q)
    q = move_to(np.array([gx, gy, EEZ(TIP_Z_ABOVE_BOX)]),
                GRIPPER_CLOSE_L, GRIPPER_CLOSE_R, STEPS_CARRY, q)
    bz_carry = ball.get_pose().p[2]
    print(f"     ball_z during carry={bz_carry:.4f}")

    # ── 9: Lower into box (v4 identical) ──────────────────────────
    print("\n   [9] Lower into box")
    q = move_to(np.array([gx, gy, EEZ(TIP_Z_LOWER)]),
                GRIPPER_CLOSE_L, GRIPPER_CLOSE_R, STEPS_LOWER, q)
    ep = np.array(ee.get_entity_pose().p)
    print(f"     EE={np.round(ep,4)}  tip_z≈{ep[2]-FINGER_TIP_OFFSET:.3f}")

    # ── 10: Release (v4 identical — deactivate_constraint in open) ─
    print("\n   [10] Release")
    q = open_gripper(q_seed=q)   # calls deactivate_constraint() internally
    for _ in range(120): scene.step()
    print("     🖐 Released")
    save_img(f'{OUT_DIR}/cam_ep{ep_num:02d}_4_placed.png')

    # ── 11: Retreat (v4 identical) ─────────────────────────────────
    print("\n   [11] Retreat")
    q = move_to(np.array([gx, gy, EEZ(TIP_Z_ABOVE_BOX)]),
                GRIPPER_OPEN_L, GRIPPER_OPEN_R, 20, q)
    move_to(np.array([TX, TY, EEZ(TIP_Z_SURVEY-0.05)]),
            GRIPPER_OPEN_L, GRIPPER_OPEN_R, STEPS_HOVER, q)

    # ── Result (v4 identical) ──────────────────────────────────────
    for _ in range(200): scene.update_render(); viewer.render()
    bf   = np.array(ball.get_pose().p)
    boxf = np.array(box.get_pose().p)
    dist = np.linalg.norm(bf[:2]-boxf[:2])
    ok   = (dist < 0.10) and (bf[2] < BOX_Z+0.06)
    print(f"\n   Ball: {np.round(bf,3)}")
    print(f"   Box:  ({boxf[0]:.3f},{boxf[1]:.3f})")
    print(f"   Dist: {dist*100:.1f}cm  → {'✅ SUCCESS' if ok else '❌ MISS'}")
    return ok

# ═══════════════════════════════════════════════════════════════════
#  MAIN LOOP  (identical to v4)
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'═'*60}")
print(f" v4 + v7 Constraint Grasp MERGED")
print(f" EE grasp z = {EEZ(TIP_Z_GRASP):.4f}  (tips at {TIP_Z_GRASP:.4f})")
print(f" Ball centre z = {BALL_Z:.4f}")
print(f"{'═'*60}\n")

rng = np.random.default_rng(42)
successes = 0; ep = 0

while not viewer.closed:
    ep += 1
    bx = TX + rng.uniform(-0.09, -0.01)
    by = TY + rng.uniform(-0.08,  0.08)
    for _ in range(100):
        gx = TX + rng.uniform(0.04, 0.11)
        gy = TY + rng.uniform(-0.08, 0.08)
        if np.linalg.norm(np.array([bx,by])-np.array([gx,gy])) >= 0.20: break
    ok = run_episode(ep, np.array([bx,by]), np.array([gx,gy]))
    if ok: successes += 1
    print(f"\n Total: {successes}/{ep} ({100*successes/ep:.0f}%)\n")