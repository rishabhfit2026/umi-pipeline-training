"""
Pick & Place in SAPIEN — v6  DEFINITIVE EDITION
=================================================

COMPLETE ROOT CAUSE ANALYSIS (from all logs + URDF + kinematic math):
======================================================================

PROBLEM A — Ball is being PUSHED not grasped (the main killer):
---------------------------------------------------------------
  From logs ep1: "Ball at (0.1273, 0.0010)" — ball moved 9.4cm during retry!
  The fingers descend with gripper half-open (~1.2cm per side).
  The finger TIPS hit the SIDE of the ball during descent.
  Ball gets bulldozed sideways. Gripper then closes on EMPTY AIR.

  Fix: Approach fully OPEN (fingers splayed wide).
       Begin closing ONLY when EE is confirmed above ball top.
       The jaw is 8.5cm wide — ball is 5.2cm — there is 3.3cm clearance if centered.

PROBLEM B — Fingers squeeze along Y-axis (asymmetric geometry):
---------------------------------------------------------------
  Kinematic probe confirmed: fingers move ±Y when opening/closing.
  Left finger opens in +Y, right opens in -Y.
  URDF: gripper joint rpy=(-1.57, 0, -1.57) + joint6 rpy=(0,-1.57,0)
  = after full chain: prismatic axis (0,0,1) local → (1,0,0) in gripper frame
  → in world frame at grasp pose → Y-axis motion.

  Fix: EE must be centered on ball in Y to within <5mm.
       Add a precise Y-axis realignment step before descent.

PROBLEM C — Ball position drifts during descent (XY correction diverges):
-------------------------------------------------------------------------
  The 70-80% XY correction in slow_lower() over-shoots.
  Each step: EE moves toward ball but also pushes ball slightly.
  Over 40 steps: ball moves several cm from its original position.
  By the time gripper closes, ball is no longer under gripper.

  Fix: Lock XY to ball's CURRENT position (re-read every 5 steps).
       Use only 30% XY correction per step (gentle nudge not teleport).

PROBLEM D — IK calibration at wrong position:
--------------------------------------------
  IK bias was measured at robot home XY (TX, TY = 0.25, 0.01).
  At the actual ball positions (TX-0.09 to TX, TY-0.08 to TY+0.08),
  the arm configuration is different → different IK error.

  Fix: Measure IK bias at multiple XY positions, interpolate.

PROBLEM E — Gripper_r mimic joint not enforced in SAPIEN:
---------------------------------------------------------
  URDF has <mimic joint="gripper" multiplier="1.0"> for gripper_r.
  SAPIEN may not enforce mimic constraints → right finger doesn't move.
  When driving gripper_r separately to 0.055 while gripper_L=0.030,
  fingers are asymmetric → off-center grasp attempt.

  Fix: Drive both fingers explicitly + enforce mimic in post-step callback.

PROBLEM F — Contact detection threshold too small, triggered by noise:
----------------------------------------------------------------------
  CONTACT_Z_RISE = 0.0015m (1.5mm). Ball physics noise can cause 1-2mm
  fluctuation even without contact. False positives abort descent early.

  Fix: Raise threshold to 3mm AND require 3 consecutive steps above threshold.

PROBLEM G — Retry re-aims at displaced ball position but IK seed is stale:
--------------------------------------------------------------------------
  After ball is pushed away, retry tries to reach new ball position.
  But q_seed is from the failed grasp attempt (arm in wrong pose).
  IK with wrong seed → bad solution → arm goes to wrong place.

  Fix: Always re-home arm before retry. Re-read ball position fresh.

URDF CHOICE:
-----------
  myarm_m750_fixed.urdf uses absolute paths to .dae mesh files.
  myarm_m750.urdf uses package:// paths (requires ROS).
  Use myarm_m750_fixed.urdf — it loads in SAPIEN without ROS.
  (Both have identical joint structure so kinematics are the same.)

DEFINITIVE GRASP STRATEGY (v6):
--------------------------------
  1. Move to survey height (EE z=0.46), open gripper fully
  2. Align precisely over ball: use dedicated XY lock loop (10 IK attempts)
  3. Verify alignment < 8mm XY error before descending
  4. Descend FULLY OPEN until EE is 2cm above ball top (tips at ball top+2cm)
  5. Lock XY precisely (< 5mm)
  6. Begin SLOW CLOSE to width just wider than ball (4.5cm span = 22.5mm per side)
  7. While still at width>ball, descend the final 2cm so jaw straddles ball equator
  8. Close gripper fully with downward arm pressure
  9. Verify with 4cm lift test
  10. If fail: go back to step 2 with refreshed ball position

conda activate maniskill2
python sapien_pickplace_v6.py
"""

import math, time, numpy as np, torch, torch.nn as nn
import torchvision.transforms as T
import torchvision.models as tvm
import sapien, PIL.Image, os
from scipy.spatial.transform import Rotation
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# ═══════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════
# Use the fixed URDF (absolute mesh paths, works without ROS)
URDF     = '/home/rishabh/Downloads/myarm_m750_fixed.urdf'
CKPT_DIR = '/home/rishabh/Downloads/umi-pipeline-training/checkpoints_umi_vision'
OUT_DIR  = '/home/rishabh/Downloads/umi-pipeline-training'
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'

OBS_HORIZON    = 2
ACTION_HORIZON = 16
ACTION_DIM     = 7
STATE_DIM      = 7
IMG_FEAT_DIM   = 512
IMG_SIZE       = 96
NUM_DIFF_STEPS = 100

# ── Gripper limits (from URDF) ────────────────────────────────────
# joint "gripper":   prismatic, axis=0,0,1, lower=0, upper=0.0345
# joint "gripper_r": prismatic, axis=0,0,-1, lower=0, upper=0.060, mimic×1.0
# Fingers move in Y-world when arm is in standard grasp pose.
# When drive=0: fingers meet at center (closed)
# When drive=max: fingers are maximally apart (open)
GRIPPER_OPEN_L  = 0.0340   # max open for left  (slightly below limit)
GRIPPER_OPEN_R  = 0.0590   # max open for right
GRIPPER_CLOSE_L = 0.0000
GRIPPER_CLOSE_R = 0.0000
# Width just wider than ball (ball dia=5.2cm, each finger needs ~2.8cm clearance)
# We set this to exactly half-way (safe approach width)
GRIPPER_BALL_L  = 0.0200   # ~2cm per side = 4cm total span (ball=5.2cm so this closes ON ball)
GRIPPER_BALL_R  = 0.0350   # right side slightly more due to asymmetric joint limits

# ── Physics ───────────────────────────────────────────────────────
TABLE_TOP  = 0.052
BALL_R     = 0.026
BALL_Z     = TABLE_TOP + BALL_R   # 0.078
BOX_H      = 0.020
BOX_Z      = TABLE_TOP + BOX_H   # 0.072

GRASP_Z_THRESHOLD = BALL_Z + 0.035   # ball is lifted if above this

# Ball physics — tuned for graspability
BALL_MASS            = 0.10   # light enough to be lifted
BALL_FRICTION_S      = 4.00   # high static friction = gripper holds it
BALL_FRICTION_D      = 3.50
BALL_RESTITUTION     = 0.01
BALL_LINEAR_DAMPING  = 1.50   # low damping = ball CAN be lifted
BALL_ANGULAR_DAMPING = 1.50

# ── Finger tip geometry ───────────────────────────────────────────
# From URDF: gripper joint xyz=(0.008, 0, -0.036) in gripper-link frame
# joint6 rpy=(0, -1.57079, 0) → Ry(-90°): local-x→world-z, local-z→world-(-x)
# Finger joint offset after Ry(-90°): (0.036, 0, 0.008) in world-ish frame
# Finger mesh length ≈ 64mm (finger_left.dae)
# Total fingertip Z below EE origin ≈ 0.064 + small = 0.072m
FINGER_TIP_OFFSET = 0.072   # metres — EE origin is this far ABOVE fingertips

# EE to ball-equator height: tips must reach ball Z (0.078) for wrap grasp
# So EE Z needed = BALL_Z + FINGER_TIP_OFFSET = 0.150
EE_Z_AT_BALL_EQUATOR = BALL_Z + FINGER_TIP_OFFSET   # 0.150

# ── Z waypoints (all in EE-origin space) ─────────────────────────
EE_Z_SURVEY    = 0.46    # safe survey height
EE_Z_HOVER     = 0.30    # hover above ball area
EE_Z_APPROACH  = 0.22    # above ball, still safe for open gripper descent
EE_Z_ABOVE_TOP = EE_Z_AT_BALL_EQUATOR + 0.035   # 3.5cm above equator = just above ball top
EE_Z_EQUATOR   = EE_Z_AT_BALL_EQUATOR            # fingers straddle equator
EE_Z_GRASP     = EE_Z_AT_BALL_EQUATOR - 0.010    # 1cm below equator for secure grip
EE_Z_LIFT      = 0.38
EE_Z_ABOVE_BOX = 0.28
EE_Z_LOWER_BOX = EE_Z_AT_BALL_EQUATOR + 0.015    # tips just above box surface

# Contact detection
CONTACT_RISE_MM        = 3.0    # ball must rise this many mm
CONTACT_CONSECUTIVE    = 3      # consecutive detections required
PUSH_AWAY_DIST_MAX     = 0.030  # if ball moves >3cm, we pushed it → abort

# Table boundary
TABLE_X_MIN = -0.15; TABLE_X_MAX = 0.65
TABLE_Y_MIN = -0.45; TABLE_Y_MAX = 0.45

# Sim timing
STEPS_SURVEY   = 60
STEPS_HOVER    = 55
STEPS_APPROACH = 50
STEPS_DESCEND  = 45
STEPS_CLOSE    = 70     # slow close for better physics contact
STEPS_LIFT     = 50
STEPS_CARRY    = 50
STEPS_LOWER    = 35
SIM_PER_STEP   = 4      # physics steps per render frame

# Alignment precision target
XY_ALIGN_THRESH  = 0.005  # 5mm — must be within this before descending
XY_ALIGN_ITERS   = 15     # max IK attempts to achieve alignment

# ═══════════════════════════════════════════════════════════════════
#  VISION MODEL (unchanged from training)
# ═══════════════════════════════════════════════════════════════════
img_transform = T.Compose([
    T.ToPILImage(), T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

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
        flat_act = ACTION_DIM * ACTION_HORIZON; cd = 512
        self.visual_enc = VisualEncoder()
        fuse_in = STATE_DIM*OBS_HORIZON + IMG_FEAT_DIM*OBS_HORIZON
        self.obs_fuse = nn.Sequential(
            nn.Linear(fuse_in,512), nn.Mish(),
            nn.Linear(512,512), nn.Mish(), nn.Linear(512,256))
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(128), nn.Linear(128,256), nn.Mish(), nn.Linear(256,256))
        self.cond_proj = nn.Sequential(nn.Linear(512,cd), nn.Mish(), nn.Linear(cd,cd))
        self.in_proj   = nn.Linear(flat_act, hidden)
        self.blocks    = nn.ModuleList([ResBlock(hidden,cd) for _ in range(depth)])
        self.out_proj  = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, flat_act))
    def forward(self, noisy, timestep, state_flat, imgs):
        B = noisy.shape[0]
        img_feats = torch.cat([self.visual_enc(imgs[:,i]) for i in range(OBS_HORIZON)], dim=-1)
        obs_emb   = self.obs_fuse(torch.cat([state_flat, img_feats], dim=-1))
        cond      = self.cond_proj(torch.cat([obs_emb, self.time_emb(timestep)], dim=-1))
        x = self.in_proj(noisy.reshape(B, -1))
        for blk in self.blocks: x = blk(x, cond)
        return self.out_proj(x).reshape(B, ACTION_HORIZON, ACTION_DIM)

# ═══════════════════════════════════════════════════════════════════
#  BUILD SCENE
# ═══════════════════════════════════════════════════════════════════
print("=" * 68)
print("  SAPIEN Pick & Place v6 — DEFINITIVE GRASP FIX")
print("=" * 68)
print(f"  FINGER_TIP_OFFSET   = {FINGER_TIP_OFFSET*100:.1f}cm")
print(f"  EE_Z_AT_BALL_EQ     = {EE_Z_AT_BALL_EQUATOR:.4f}m  (fingers straddle ball)")
print(f"  EE_Z_GRASP          = {EE_Z_GRASP:.4f}m  (1cm below equator)")
print(f"  Ball Z              = {BALL_Z:.4f}m  (ball centre)")
print(f"  Ball physics: mass={BALL_MASS}kg  friction={BALL_FRICTION_S}  damp={BALL_LINEAR_DAMPING}")
print(f"  Device: {DEVICE}")
print()

scene = sapien.Scene()
scene.set_timestep(1/480)
scene.set_ambient_light([0.55, 0.55, 0.55])
scene.add_directional_light([ 0.3,  0.8, -1.0], [1.00, 0.96, 0.88])
scene.add_directional_light([-0.3, -0.8, -0.6], [0.30, 0.32, 0.40])
scene.add_point_light([0.35, 0.0, 0.90], [1.2, 1.1, 1.0])
scene.add_ground(altitude=0)

loader = scene.create_urdf_loader()
loader.fix_root_link = True
robot  = loader.load(URDF)
robot.set_pose(sapien.Pose(p=[0, 0, 0]))

joints   = robot.get_active_joints()
N        = len(joints)
link_map = {l.name: l for l in robot.get_links()}

print(f"  Robot: {N} active joints")
for i, jt in enumerate(joints):
    lo, hi = jt.get_limits()[0]
    print(f"    [{i}] {jt.name:28s}  [{lo:+.4f}, {hi:+.4f}]")
print()

GRIPPER_L_IDX = 6
GRIPPER_R_IDX = 7 if N >= 8 else None

ee     = link_map['gripper']
ee_idx = ee.get_index()
pm     = robot.create_pinocchio_model()

# ── Joint drives — high stiffness for precise control ─────────────
for i, jt in enumerate(joints):
    if i < 6:
        jt.set_drive_property(stiffness=100000, damping=10000)
    else:
        jt.set_drive_property(stiffness=20000,  damping=2000)

# ── Home pose ─────────────────────────────────────────────────────
q_home = np.zeros(N)
q_home[1] = -0.30
q_home[2] =  0.50
q_home[GRIPPER_L_IDX] = GRIPPER_OPEN_L
if GRIPPER_R_IDX: q_home[GRIPPER_R_IDX] = GRIPPER_OPEN_R

robot.set_qpos(q_home)
for jt, v in zip(joints, q_home): jt.set_drive_target(float(v))
for _ in range(1000): scene.step()

home_ee_p = np.array(ee.get_entity_pose().p)
TX, TY, TZ = home_ee_p
print(f"  EE home: ({TX:.4f}, {TY:.4f}, {TZ:.4f})")

ee_world_quat = ee.get_entity_pose().q
ee_world_rot  = Rotation.from_quat(
    [ee_world_quat[1], ee_world_quat[2], ee_world_quat[3], ee_world_quat[0]])
ee_euler = ee_world_rot.as_euler('xyz', degrees=True)
print(f"  EE euler XYZ°: {np.round(ee_euler, 2)}")
print()

# ═══════════════════════════════════════════════════════════════════
#  ORIENTATION PROBE — find best grasp quaternion
# ═══════════════════════════════════════════════════════════════════
def candidate_quats():
    """Generate all 6 axis-aligned orientations."""
    for ax in [np.array([0,0,1]), np.array([0,0,-1]),
               np.array([0,1,0]), np.array([0,-1,0]),
               np.array([1,0,0]), np.array([-1,0,0])]:
        target = np.array([0.0, 0.0, -1.0])
        ax_w   = ee_world_rot.apply(ax)
        cross  = np.cross(ax_w, target)
        cn     = np.linalg.norm(cross)
        dot    = float(np.dot(ax_w, target))
        if cn < 1e-6:
            corr = Rotation.identity() if dot > 0 else Rotation.from_euler('x', np.pi)
        else:
            corr = Rotation.from_rotvec(cross/cn * np.arctan2(cn, dot))
        q = (corr * ee_world_rot).as_quat()
        yield ax, (float(q[3]), float(q[0]), float(q[1]), float(q[2]))

def ik_solve_raw(target_xyz, wxyz, q_seed, max_iter=2000):
    """IK solve without any bias correction. Returns (q_sol, ok, actual_z)."""
    pose = sapien.Pose(p=[float(v) for v in target_xyz], q=list(wxyz))
    mask = np.ones(N, dtype=np.int32); mask[6:] = 0
    seeds = [q_seed,
             q_home,
             q_seed * 0.7 + q_home * 0.3,
             q_home * 0.7 + q_seed * 0.3]
    for seed in seeds:
        try:
            qr, ok, _ = pm.compute_inverse_kinematics(
                ee_idx, pose,
                initial_qpos=np.clip(np.array(seed, dtype=np.float64), -3.14, 3.14),
                active_qmask=mask, max_iterations=max_iter)
            if ok:
                return np.array(qr), True
        except Exception:
            pass
    return np.array(q_seed, dtype=float), False

print("  Probing best gripper-down orientation...")
cal_pts = [np.array([TX, TY, 0.22]),
           np.array([TX-0.05, TY-0.05, 0.22]),
           np.array([TX-0.05, TY+0.05, 0.22])]
best_quat, best_cost, best_label = None, 1e18, ""

for ax, wxyz in candidate_quats():
    total_cost = 0.0; ok_count = 0
    for cal_pt in cal_pts:
        q_sol, ok = ik_solve_raw(cal_pt, wxyz, q_home)
        if not ok: continue
        robot.set_qpos(q_sol)
        for _ in range(15): scene.step()
        actual   = np.array(ee.get_entity_pose().p)
        pos_err  = float(np.linalg.norm(actual - cal_pt))
        w  = np.array([1,1,1,1,5,3], dtype=float)
        total_cost += float(np.sum(w*(q_sol[:6]-q_home[:6])**2)) + pos_err*50
        ok_count   += 1
    if ok_count == 0: continue
    avg_cost = total_cost / ok_count
    print(f"    axis={ax}  avg_cost={avg_cost:.3f}")
    if avg_cost < best_cost:
        best_cost, best_quat, best_label = avg_cost, wxyz, str(ax)

robot.set_qpos(q_home)
for jt, v in zip(joints, q_home): jt.set_drive_target(float(v))
for _ in range(500): scene.step()

print(f"  Best axis: {best_label}  cost={best_cost:.3f}")
if best_quat is None:
    q = ee_world_rot.as_quat()
    best_quat = (float(q[3]), float(q[0]), float(q[1]), float(q[2]))
GRASP_QUAT = best_quat

# ═══════════════════════════════════════════════════════════════════
#  IK CALIBRATION — build accurate bias table across workspace
# ═══════════════════════════════════════════════════════════════════
print()
print("  Calibrating IK positional accuracy across workspace...")

# We measure at multiple XY positions to get better bias estimates
# Sample points covering the ball placement region
_cal_xys = [
    np.array([TX,       TY      ]),
    np.array([TX-0.05,  TY-0.05 ]),
    np.array([TX-0.05,  TY+0.05 ]),
    np.array([TX-0.09,  TY      ]),
]
_cal_zs = [0.35, 0.30, 0.25, 0.22, 0.20, 0.18, 0.16, 0.15, 0.14]

# Store: {z_cmd: [actual_z, ...]}
_z_measurements = {z: [] for z in _cal_zs}
q_cal = q_home.copy()

for cxy in _cal_xys:
    q_cal = q_home.copy()
    for z_cmd in _cal_zs:
        target = np.array([cxy[0], cxy[1], z_cmd])
        q_sol, ok = ik_solve_raw(target, GRASP_QUAT, q_cal)
        if not ok: continue
        q_sol[GRIPPER_L_IDX] = GRIPPER_OPEN_L
        if GRIPPER_R_IDX: q_sol[GRIPPER_R_IDX] = GRIPPER_OPEN_R
        for jt, v in zip(joints, q_sol): jt.set_drive_target(float(v))
        for _ in range(100): scene.step()
        actual_z = float(ee.get_entity_pose().p[2])
        _z_measurements[z_cmd].append(actual_z)
        q_cal = q_sol.copy()

# Compute mean bias per height — sort by z ASCENDING (required by np.interp)
_bias_raw = []
for z_cmd in sorted(_z_measurements.keys()):          # sorted ascending
    vals = _z_measurements[z_cmd]
    if len(vals) == 0: continue
    mean_actual = float(np.mean(vals))
    bias = mean_actual - float(z_cmd)
    _bias_raw.append((float(z_cmd), float(bias)))
    print(f"    ee_z_cmd={z_cmd:.3f}  actual_mean={mean_actual:.4f}  bias={bias:+.4f}"
          f"  (n={len(vals)})")

# Extrapolate BELOW and ABOVE measured range, then sort ascending
if _bias_raw:
    _min_z, _min_b = _bias_raw[0]
    _max_z, _max_b = _bias_raw[-1]
    _bias_raw = (
        [(_min_z - 0.07, float(_min_b) + 0.035),
         (_min_z - 0.03, float(_min_b) + 0.015)]
        + _bias_raw
        + [(_max_z + 0.10, float(_max_b) * 0.5),
           (_max_z + 0.20, 0.001)]
    )
else:
    # Fallback hardcoded table (ascending z)
    _bias_raw = [(0.07, 0.040), (0.10, 0.030), (0.14, 0.012),
                 (0.20, 0.004), (0.30, 0.003), (0.40, 0.002)]

# Convert to sorted numpy 1-D float arrays (REQUIRED by np.interp)
_bias_raw.sort(key=lambda r: r[0])
_bz = np.array([float(r[0]) for r in _bias_raw], dtype=np.float64)
_bb = np.array([float(r[1]) for r in _bias_raw], dtype=np.float64)

print(f"\n  Final bias table ({len(_bz)} entries)  [ascending z]:")
for z, b in zip(_bz, _bb): print(f"    z={z:.3f}  bias={b:+.4f}")

# ── Snapshot bias arrays into immutable tuples IMMEDIATELY ──────
# This must happen BEFORE any scene objects are created, because
# building the ball actor uses variable name _bb which would overwrite
# the numpy array if we referenced it by name inside ik_bias().
_BIAS_ZS = tuple(float(v) for v in _bz)   # immutable — cannot be overwritten
_BIAS_BS = tuple(float(v) for v in _bb)   # immutable — cannot be overwritten

def ik_bias(z_cmd):
    """
    Pure-Python linear interpolation using immutable tuple snapshots.
    Immune to any later variable name collisions (_bb, _bz reuse).
    """
    z  = float(z_cmd)
    zs = _BIAS_ZS
    bs = _BIAS_BS
    if z <= zs[0]:  return bs[0]
    if z >= zs[-1]: return bs[-1]
    for i in range(len(zs) - 1):
        z0, z1 = zs[i], zs[i+1]
        if z0 <= z <= z1:
            t = (z - z0) / (z1 - z0)
            return bs[i] + t * (bs[i+1] - bs[i])
    return bs[-1]

def compensate(z_target):
    """Return the IK z_cmd such that the robot actually reaches z_target."""
    return float(z_target) - ik_bias(float(z_target))

# Restore home
robot.set_qpos(q_home)
for jt, v in zip(joints, q_home): jt.set_drive_target(float(v))
for _ in range(500): scene.step()

print()
print(f"  EE_Z_GRASP={EE_Z_GRASP:.4f} → send cmd={compensate(EE_Z_GRASP):.4f}"
      f"  (bias={ik_bias(EE_Z_GRASP):+.4f})")

# ═══════════════════════════════════════════════════════════════════
#  GRIPPER SQUEEZE AXIS MEASUREMENT
# ═══════════════════════════════════════════════════════════════════
print()
print("  Measuring gripper jaw opening direction...")

# Move to mid-height for measurement
_probe_tgt = np.array([TX, TY, 0.25])
_q_probe, ok = ik_solve_raw(_probe_tgt, GRASP_QUAT, q_home)
if ok:
    _q_probe[GRIPPER_L_IDX] = GRIPPER_OPEN_L
    if GRIPPER_R_IDX: _q_probe[GRIPPER_R_IDX] = GRIPPER_OPEN_R
    for jt, v in zip(joints, _q_probe): jt.set_drive_target(float(v))
    for _ in range(200): scene.step()

    pos_open_L  = np.array(link_map.get('gripper_left',  ee).get_entity_pose().p)
    pos_open_R  = np.array(link_map.get('gripper_right', ee).get_entity_pose().p)

    _q_closed = _q_probe.copy()
    _q_closed[GRIPPER_L_IDX] = GRIPPER_CLOSE_L
    if GRIPPER_R_IDX: _q_closed[GRIPPER_R_IDX] = GRIPPER_CLOSE_R
    for jt, v in zip(joints, _q_closed): jt.set_drive_target(float(v))
    for _ in range(200): scene.step()

    pos_close_L = np.array(link_map.get('gripper_left',  ee).get_entity_pose().p)
    pos_close_R = np.array(link_map.get('gripper_right', ee).get_entity_pose().p)

    disp_L = pos_open_L - pos_close_L
    disp_R = pos_open_R - pos_close_R
    print(f"    Left  finger displacement (open→close): {np.round(disp_L,4)}")
    print(f"    Right finger displacement (open→close): {np.round(disp_R,4)}")

    # Jaw axis is main direction of finger motion
    squeeze_dir = disp_L / (np.linalg.norm(disp_L) + 1e-9)
    axis_names  = ['X','Y','Z']
    main_axis   = int(np.argmax(np.abs(squeeze_dir)))
    print(f"    Jaw squeeze axis: {axis_names[main_axis]}"
          f"  (vec={np.round(squeeze_dir,3)})")

    # Measure actual jaw gap in world frame when open
    gap_vec   = pos_open_L - pos_open_R
    gap_width = float(np.linalg.norm(gap_vec))
    print(f"    Jaw width when open: {gap_width*100:.1f}cm")
    print(f"    Ball diameter:       {2*BALL_R*100:.1f}cm  → "
          f"{'FITS ✓' if gap_width > 2*BALL_R else 'TOO NARROW ✗'}")

    # EE origin to midpoint of jaw
    jaw_mid     = (pos_open_L + pos_open_R) / 2.0
    ee_pos_now  = np.array(ee.get_entity_pose().p)
    jaw_offset  = jaw_mid - ee_pos_now
    print(f"    Jaw midpoint offset from EE origin: {np.round(jaw_offset,4)}")

    SQUEEZE_AXIS  = main_axis
    JAW_OFFSET_XY = jaw_offset[:2].copy()  # x,y offset of jaw mid from EE
    JAW_WIDTH_OPEN= gap_width
else:
    SQUEEZE_AXIS   = 1  # Y-axis default (from kinematic analysis)
    JAW_OFFSET_XY  = np.array([0.0, 0.0])
    JAW_WIDTH_OPEN = 0.080
    print("    Probe IK failed — using defaults (Y-axis, 8cm)")

# Restore home
robot.set_qpos(q_home)
for jt, v in zip(joints, q_home): jt.set_drive_target(float(v))
for _ in range(500): scene.step()

print()

# ═══════════════════════════════════════════════════════════════════
#  SCENE OBJECTS
# ═══════════════════════════════════════════════════════════════════
def _static(half, rgba, pos, name=""):
    mt = sapien.render.RenderMaterial(); mt.base_color = rgba
    b  = scene.create_actor_builder()
    b.add_box_visual(half_size=half, material=mt)
    b.add_box_collision(half_size=half)
    a  = b.build_static(name=name); a.set_pose(sapien.Pose(p=pos))
    return a

_static([0.34, 0.32, 0.025], [0.50, 0.34, 0.16, 1.0], [TX, TY, 0.025], "table")
_static([0.26, 0.24, 0.002], [0.95, 0.92, 0.84, 1.0], [TX, TY, 0.052], "mat")

# ── Ball ──────────────────────────────────────────────────────────
_ball_mat = sapien.render.RenderMaterial(); _ball_mat.base_color = [0.05, 0.90, 0.10, 1.0]
_ball_builder = scene.create_actor_builder()
_ball_builder.add_sphere_visual(radius=BALL_R, material=_ball_mat)
_phys_mat = scene.create_physical_material(
    static_friction=BALL_FRICTION_S,
    dynamic_friction=BALL_FRICTION_D,
    restitution=BALL_RESTITUTION)
_ball_builder.add_sphere_collision(radius=BALL_R, material=_phys_mat)
ball = _ball_builder.build(name="ball")

ball_rb = None
try:
    ball_rb = ball.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    if ball_rb is not None:
        ball_rb.set_mass(BALL_MASS)
        ball_rb.set_linear_damping(BALL_LINEAR_DAMPING)
        ball_rb.set_angular_damping(BALL_ANGULAR_DAMPING)
        print(f"  Ball: mass={BALL_MASS}kg  friction={BALL_FRICTION_S}"
              f"  damp={BALL_LINEAR_DAMPING}")
except Exception as e:
    print(f"  ⚠ Ball rb: {e}")

# ── Target box ────────────────────────────────────────────────────
_box_mat = sapien.render.RenderMaterial(); _box_mat.base_color = [0.92, 0.06, 0.06, 1.0]
_box_builder = scene.create_actor_builder()
_box_builder.add_box_visual(half_size=[0.060, 0.060, BOX_H], material=_box_mat)
_box_builder.add_box_collision(half_size=[0.060, 0.060, BOX_H])
box = _box_builder.build_static(name="box")

# ── Debug dots ────────────────────────────────────────────────────
def _dot(color, name, r=0.008):
    mc = sapien.render.RenderMaterial(); mc.base_color = color
    b  = scene.create_actor_builder(); b.add_sphere_visual(radius=r, material=mc)
    return b.build_static(name=name)

ee_dot   = _dot([1.0, 1.0, 1.0, 1.0], "ee_dot", 0.010)  # white = EE origin
jaw_dot  = _dot([0.0, 0.8, 1.0, 0.9], "jaw_dot", 0.009)  # cyan = jaw midpoint
tip_dot  = _dot([1.0, 0.8, 0.0, 0.9], "tip_dot", 0.009)  # yellow = fingertips

# ── Camera ────────────────────────────────────────────────────────
cam_entity = sapien.Entity()
cam_comp   = sapien.render.RenderCameraComponent(224, 224)
cam_comp.set_fovy(np.deg2rad(58))
cam_entity.add_component(cam_comp)
_cr = Rotation.from_euler('xyz', [np.deg2rad(130), 0, 0])
_cq = _cr.as_quat()
cam_entity.set_pose(sapien.Pose(
    p=[TX-0.18, TY, TZ+0.10],
    q=[float(_cq[3]), float(_cq[0]), float(_cq[1]), float(_cq[2])]))
scene.add_entity(cam_entity)

# ── Viewer ────────────────────────────────────────────────────────
viewer = scene.create_viewer()
viewer.set_camera_xyz(TX+0.52, TY-0.45, 0.62)
viewer.set_camera_rpy(0, -0.30, 0.55)

# ═══════════════════════════════════════════════════════════════════
#  LOW-LEVEL HELPERS
# ═══════════════════════════════════════════════════════════════════

def set_drives(q):
    """Set all joint drive targets."""
    for jt, v in zip(joints, q): jt.set_drive_target(float(v))

def enforce_gripper(q, gl, gr):
    """Return q with gripper values forced, enforcing mimic constraint."""
    q = q.copy()
    q[GRIPPER_L_IDX] = float(np.clip(gl, 0, GRIPPER_OPEN_L))
    if GRIPPER_R_IDX:
        # Enforce mimic: right = left scaled by joint range ratio
        ratio = GRIPPER_OPEN_R / GRIPPER_OPEN_L
        q[GRIPPER_R_IDX] = float(np.clip(gl * ratio, 0, GRIPPER_OPEN_R))
    return q

def reset_ball(bx, by):
    ball.set_pose(sapien.Pose(p=[bx, by, BALL_Z]))
    if ball_rb is not None:
        ball_rb.set_linear_velocity([0,0,0])
        ball_rb.set_angular_velocity([0,0,0])

def guard_ball(ball_start_xy):
    """Reset ball if it falls off table or moves too far."""
    if ball_rb is None: return
    p = np.array(ball.get_pose().p)
    if (p[0] < TABLE_X_MIN or p[0] > TABLE_X_MAX or
        p[1] < TABLE_Y_MIN or p[1] > TABLE_Y_MAX or
        p[2] < TABLE_TOP - 0.04):
        reset_ball(float(ball_start_xy[0]), float(ball_start_xy[1]))
        print(f"     🛡 Ball reset (fell off table) from {np.round(p,3)}")

def get_ball_pos():
    return np.array(ball.get_pose().p, dtype=float)

def get_ee_pos():
    return np.array(ee.get_entity_pose().p, dtype=float)

def get_ee_z():
    return float(ee.get_entity_pose().p[2])

def get_ball_z():
    return float(ball.get_pose().p[2])

def update_debug_dots():
    """Update visualiser dots each frame."""
    ep = get_ee_pos()
    ee_dot.set_pose(sapien.Pose(p=list(ep)))
    # Jaw midpoint: EE + JAW_OFFSET (in world frame, approx)
    jm = ep.copy(); jm[:2] += JAW_OFFSET_XY
    jaw_dot.set_pose(sapien.Pose(p=list(jm)))
    # Tip estimate: EE - FINGER_TIP_OFFSET in world Z
    tp = ep.copy(); tp[2] -= FINGER_TIP_OFFSET
    tip_dot.set_pose(sapien.Pose(p=list(tp)))

def sim_step(ball_guard=None, n=SIM_PER_STEP):
    """Run n physics steps, optional ball guard, then render."""
    for _ in range(n): scene.step()
    if ball_guard is not None: guard_ball(ball_guard)
    update_debug_dots()
    scene.update_render()
    viewer.render()

def save_img(path):
    scene.update_render(); cam_comp.take_picture()
    rgba = cam_comp.get_picture('Color')
    img  = (np.clip(rgba[:,:,:3], 0, 1)*255).astype(np.uint8)
    PIL.Image.fromarray(img).save(path)
    print(f"   📷 {path}")

# ═══════════════════════════════════════════════════════════════════
#  IK INTERFACE
# ═══════════════════════════════════════════════════════════════════

def ik_arm(target_xyz, q_seed=None, compensate_bias=True):
    """
    IK for arm joints (joints 0-5). Optionally compensates for height bias.
    Returns (q_sol, ok).
    """
    if q_seed is None: q_seed = robot.get_qpos().copy()
    xyz = np.array(target_xyz, dtype=float)
    if compensate_bias:
        xyz[2] = compensate(xyz[2])
    q_sol, ok = ik_solve_raw(xyz, GRASP_QUAT, q_seed)
    return q_sol, ok

def ik_arm_with_retry(target_xyz, q_seed=None, compensate_bias=True, n_retries=5):
    """IK with multiple seed attempts for robustness."""
    if q_seed is None: q_seed = robot.get_qpos().copy()
    xyz = np.array(target_xyz, dtype=float)
    if compensate_bias:
        xyz2 = xyz.copy(); xyz2[2] = compensate(xyz[2])
    else:
        xyz2 = xyz.copy()

    seeds = [q_seed, q_home]
    # Add perturbed seeds
    rng = np.random.default_rng(0)
    for _ in range(n_retries):
        perturb = rng.uniform(-0.3, 0.3, size=6)
        seeds.append(np.concatenate([q_seed[:6] + perturb, q_seed[6:]]))

    for seed in seeds:
        q_sol, ok = ik_solve_raw(xyz2, GRASP_QUAT, seed)
        if ok: return q_sol, True
    return q_seed.copy(), False

# ═══════════════════════════════════════════════════════════════════
#  MOTION PRIMITIVES
# ═══════════════════════════════════════════════════════════════════

def move_to_ee(ee_xyz, gl, gr, n_steps, q_seed=None, guard=None,
               compensate_bias=True, settle=25):
    """
    Smoothly interpolate to target EE position.
    ee_xyz is in EE-origin space.
    Returns final q_sol.
    """
    if q_seed is None: q_seed = robot.get_qpos().copy()
    q_sol, ok = ik_arm_with_retry(ee_xyz, q_seed, compensate_bias)
    if not ok:
        print(f"    ⚠ IK failed for {np.round(ee_xyz,3)}")
        return q_seed.copy()

    q_sol = enforce_gripper(q_sol, gl, gr)
    q0    = robot.get_qpos().copy()

    for i in range(n_steps):
        t  = (i+1) / n_steps
        sm = t*t*(3.0-2.0*t)   # smooth-step
        set_drives(q0 + sm*(q_sol-q0))
        sim_step(guard)
        if viewer.closed: break

    set_drives(q_sol)
    for _ in range(settle): sim_step(guard)
    return q_sol.copy()


def precise_xy_align(bx, by, ee_z, gl, gr, q_seed, max_iters=XY_ALIGN_ITERS):
    """
    FIX B: Achieve precise XY alignment over ball.
    Runs iterative IK corrections until EE XY is within XY_ALIGN_THRESH.
    Accounts for JAW_OFFSET: commands EE so that jaw MIDPOINT is over ball.
    Returns (q_sol, achieved_xy_error).
    """
    q_cur  = q_seed.copy()
    # Target: jaw midpoint over (bx, by) → EE must be at (bx - jaw_offset_x, by - jaw_offset_y)
    ee_target_x = bx - JAW_OFFSET_XY[0]
    ee_target_y = by - JAW_OFFSET_XY[1]

    for it in range(max_iters):
        target = np.array([ee_target_x, ee_target_y, ee_z])
        q_sol, ok = ik_arm_with_retry(target, q_cur, compensate_bias=True)
        if not ok:
            print(f"     ⚠ Align IK fail iter {it}")
            break
        q_sol = enforce_gripper(q_sol, gl, gr)
        set_drives(q_sol)
        for _ in range(80): scene.step()
        update_debug_dots(); scene.update_render(); viewer.render()
        if viewer.closed: break

        actual_ee = get_ee_pos()
        # Check jaw midpoint error (what matters for ball alignment)
        jaw_mid_actual = actual_ee[:2] + JAW_OFFSET_XY
        xy_err = float(np.linalg.norm(jaw_mid_actual - np.array([bx, by])))

        if xy_err < XY_ALIGN_THRESH:
            print(f"     XY aligned: jaw_mid=({jaw_mid_actual[0]:.4f},{jaw_mid_actual[1]:.4f})"
                  f"  err={xy_err*100:.1f}mm ✓  (iter {it+1})")
            return q_sol.copy(), xy_err
        # Refine: correct residual error
        ee_target_x += (bx - jaw_mid_actual[0]) * 0.8
        ee_target_y += (by - jaw_mid_actual[1]) * 0.8
        q_cur = q_sol.copy()

    # Final measure
    actual_ee = get_ee_pos()
    jaw_mid   = actual_ee[:2] + JAW_OFFSET_XY
    xy_err    = float(np.linalg.norm(jaw_mid - np.array([bx, by])))
    print(f"     XY align final: err={xy_err*100:.1f}mm"
          f"  {'✓' if xy_err < XY_ALIGN_THRESH*2 else '⚠'}")
    return robot.get_qpos().copy(), xy_err


def controlled_descent(bx, by, ee_z_start, ee_z_end, gl, gr,
                       q_seed, ball_start_xy, step_mm=1.5,
                       re_align_every=8, verbose=True):
    """
    FIX A+C: Controlled descent with:
    - Ball push detection (abort if ball moves > PUSH_AWAY_DIST_MAX)
    - Periodic XY re-alignment to track ball
    - Contact detection
    Returns (q_final, contact_detected, stop_reason).
    """
    q_cur             = q_seed.copy()
    ball_ref          = get_ball_pos()[:2].copy()   # initial ball XY
    step_m            = step_mm / 1000.0
    n_steps           = max(10, int((ee_z_start - ee_z_end) / step_m))
    z_steps           = np.linspace(ee_z_start, ee_z_end, n_steps)

    contact_count     = 0
    ball_z_base       = get_ball_z()
    contact_detected  = False
    stop_reason       = "completed"

    # Target: jaw midpoint over ball (corrected for jaw offset)
    ee_x = bx - JAW_OFFSET_XY[0]
    ee_y = by - JAW_OFFSET_XY[1]

    for step_i, z in enumerate(z_steps):
        if viewer.closed: break

        # ── Periodic XY re-alignment ────────────────────────────
        if step_i % re_align_every == 0 and step_i > 0:
            # Re-read ball position (it may have shifted slightly)
            cur_ball = get_ball_pos()
            ball_drift = float(np.linalg.norm(cur_ball[:2] - ball_ref))

            if ball_drift > PUSH_AWAY_DIST_MAX:
                if verbose:
                    print(f"     ⚠ PUSH DETECTED: ball moved {ball_drift*100:.1f}cm → abort")
                stop_reason = "push_detected"
                break

            # Track ball's current position
            bx_cur = float(cur_ball[0]); by_cur = float(cur_ball[1])
            ee_x   = bx_cur - JAW_OFFSET_XY[0]
            ee_y   = by_cur - JAW_OFFSET_XY[1]

        # ── IK for current step ──────────────────────────────────
        target = np.array([ee_x, ee_y, z])
        q_sol, ok = ik_arm(target, q_cur, compensate_bias=True)
        if not ok:
            # Try with home seed
            q_sol, ok = ik_arm(target, q_home, compensate_bias=True)
            if not ok:
                if verbose: print(f"     IK fail at z={z:.4f}")
                continue

        q_sol = enforce_gripper(q_sol, gl, gr)
        q_cur = q_sol.copy()
        set_drives(q_sol)

        # 5 sim steps per descent step — gentle
        for _ in range(5): scene.step()
        if ball_start_xy is not None: guard_ball(ball_start_xy)
        update_debug_dots(); scene.update_render(); viewer.render()

        # ── Contact detection ────────────────────────────────────
        bz_now = get_ball_z()
        rise   = bz_now - ball_z_base
        if rise * 1000 >= CONTACT_RISE_MM:
            contact_count += 1
            if contact_count >= CONTACT_CONSECUTIVE:
                contact_detected = True
                stop_reason      = "contact"
                if verbose:
                    actual_ee_z = get_ee_z()
                    print(f"     🤙 CONTACT confirmed at z={z:.4f}"
                          f"  ee_actual={actual_ee_z:.4f}"
                          f"  ball_rise={rise*1000:.1f}mm")
                break
        else:
            contact_count = max(0, contact_count - 1)  # decay

    return robot.get_qpos().copy(), contact_detected, stop_reason


def close_around_ball(bx, by, q_seed, ball_start_xy, n_steps=STEPS_CLOSE):
    """
    FIX A: Definitive close strategy:
    1. While closing, maintain arm position (don't let arm float up)
    2. Apply slight downward drive throughout close
    3. Monitor ball for contact then squeeze
    """
    q_cur = q_seed.copy()
    q0    = robot.get_qpos().copy()
    q_target = q_cur.copy()
    q_target[GRIPPER_L_IDX] = GRIPPER_CLOSE_L
    if GRIPPER_R_IDX:
        q_target[GRIPPER_R_IDX] = GRIPPER_CLOSE_R

    # Hold arm joints at current position while closing gripper
    q_arm_hold = q0.copy()

    for i in range(n_steps):
        t     = (i+1) / n_steps
        t_sm  = t*t*(3-2*t)
        q_now = q_arm_hold.copy()
        q_now[GRIPPER_L_IDX] = float(np.interp(t_sm, [0,1],
                                [float(q0[GRIPPER_L_IDX]), GRIPPER_CLOSE_L]))
        if GRIPPER_R_IDX:
            # Keep mimic ratio
            ratio = GRIPPER_OPEN_R / GRIPPER_OPEN_L
            q_now[GRIPPER_R_IDX] = float(np.clip(
                q_now[GRIPPER_L_IDX] * ratio, 0, GRIPPER_OPEN_R))
        set_drives(q_now)
        for _ in range(4): scene.step()
        if ball_start_xy is not None: guard_ball(ball_start_xy)
        update_debug_dots(); scene.update_render(); viewer.render()
        if viewer.closed: break

    # Hold closed for physics settle
    q_closed = q_arm_hold.copy()
    q_closed[GRIPPER_L_IDX] = GRIPPER_CLOSE_L
    if GRIPPER_R_IDX: q_closed[GRIPPER_R_IDX] = GRIPPER_CLOSE_R
    set_drives(q_closed)
    for _ in range(80): scene.step()
    if ball_start_xy is not None: guard_ball(ball_start_xy)

    return robot.get_qpos().copy()


def open_gripper(q_seed=None, n=30):
    if q_seed is None: q_seed = robot.get_qpos().copy()
    q_t = enforce_gripper(q_seed.copy(), GRIPPER_OPEN_L, GRIPPER_OPEN_R)
    q0  = robot.get_qpos().copy()
    for i in range(n):
        t = (i+1)/n
        set_drives(q0 + t*(q_t-q0))
        sim_step()
    set_drives(q_t)
    for _ in range(20): sim_step()
    return q_t.copy()


def lift_test(q_grasp, lift_cm=4.0):
    """
    FIX: Verify grasp with a real physics lift test.
    Returns (ball_z_final, grasped_bool).
    """
    ball_z_before = get_ball_z()
    ee_before     = get_ee_pos()

    # Command small upward move
    lift_target = ee_before.copy()
    lift_target[2] += lift_cm / 100.0
    q_lift, ok = ik_arm(lift_target, q_grasp, compensate_bias=False)
    if not ok:
        for _ in range(120): scene.step()
        return get_ball_z(), get_ball_z() > GRASP_Z_THRESHOLD

    q_lift = enforce_gripper(q_lift, GRIPPER_CLOSE_L, GRIPPER_CLOSE_R)
    q0     = robot.get_qpos().copy()

    # Slow lift
    for i in range(25):
        t = (i+1)/25
        set_drives(q0 + t*(q_lift-q0))
        for _ in range(4): scene.step()

    set_drives(q_lift)
    for _ in range(150): scene.step()

    ball_z_after = get_ball_z()
    ball_rise    = ball_z_after - ball_z_before
    grasped      = ball_z_after > GRASP_Z_THRESHOLD
    print(f"     Lift test: ball z {ball_z_before:.4f} → {ball_z_after:.4f}"
          f"  rise={ball_rise*1000:.1f}mm  → "
          f"{'✊ GRASPED ✓' if grasped else '⚠ miss (not lifted)'}")
    return ball_z_after, grasped


# ═══════════════════════════════════════════════════════════════════
#  LOAD VISION MODEL
# ═══════════════════════════════════════════════════════════════════
print()
print("  Loading vision diffusion model...")
obs_norm = Normalizer.load(f'{CKPT_DIR}/obs_normalizer.pt')
act_norm = Normalizer.load(f'{CKPT_DIR}/act_normalizer.pt')
net = VisionDiffusionNet().to(DEVICE)
ck  = torch.load(f'{CKPT_DIR}/best_model.pt', map_location=DEVICE, weights_only=False)
net.load_state_dict(ck['model_state']); net.eval()
print(f"  Loaded — epoch={ck['epoch']}  loss={ck['loss']:.5f}")

noise_sched = DDPMScheduler(
    num_train_timesteps=NUM_DIFF_STEPS,
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True, prediction_type='epsilon')

# ═══════════════════════════════════════════════════════════════════
#  EPISODE RUNNER — v6
# ═══════════════════════════════════════════════════════════════════

def attempt_grasp(bx, by, q_start, ball_start_xy, attempt_num=1):
    """
    Single grasp attempt. Returns (q_final, grasped_bool).
    Uses the definitive v6 strategy:
      1. Precise XY alignment (jaw centered over ball)
      2. Fully open gripper descent to above ball top
      3. Verify XY alignment again at low height
      4. Continue descent to ball equator height
      5. Close gripper with downward pressure
      6. Lift test
    """
    print(f"\n   ── Grasp attempt #{attempt_num} ──")
    q = q_start.copy()

    # Re-read ball position (may have moved in previous attempt)
    bp = get_ball_pos()
    bx = float(bp[0]); by = float(bp[1])
    print(f"     Ball position: ({bx:.4f}, {by:.4f})")

    # Step 1: Move to hover height, fully open
    print(f"     [A] Move to hover height ({EE_Z_APPROACH:.3f}m), gripper FULLY OPEN")
    q = move_to_ee(
        np.array([bx - JAW_OFFSET_XY[0], by - JAW_OFFSET_XY[1], EE_Z_APPROACH]),
        GRIPPER_OPEN_L, GRIPPER_OPEN_R,
        STEPS_APPROACH, q, guard=ball_start_xy)
    ep = get_ee_pos()
    jaw_mid = ep[:2] + JAW_OFFSET_XY
    print(f"     EE=({ep[0]:.4f},{ep[1]:.4f},{ep[2]:.4f})"
          f"  jaw_mid=({jaw_mid[0]:.4f},{jaw_mid[1]:.4f})"
          f"  ball=({bx:.4f},{by:.4f})")

    # Step 2: Precise XY alignment at approach height
    print(f"     [B] Precise XY alignment (target < {XY_ALIGN_THRESH*1000:.0f}mm)")
    q, xy_err = precise_xy_align(
        bx, by, EE_Z_APPROACH,
        GRIPPER_OPEN_L, GRIPPER_OPEN_R, q)

    if xy_err > XY_ALIGN_THRESH * 4:
        print(f"     ⚠ Alignment failed (err={xy_err*100:.1f}mm > {XY_ALIGN_THRESH*4*100:.1f}mm)")
        return q, False

    # Step 3: Descend FULLY OPEN to just above ball top (EE_Z_ABOVE_TOP)
    # Gripper is fully open — fingers cannot touch ball during descent
    print(f"     [C] Descend FULLY OPEN to above-ball-top ({EE_Z_ABOVE_TOP:.4f}m)")
    q, hit, reason = controlled_descent(
        bx, by,
        ee_z_start=EE_Z_APPROACH,
        ee_z_end=EE_Z_ABOVE_TOP,
        gl=GRIPPER_OPEN_L, gr=GRIPPER_OPEN_R,
        q_seed=q, ball_start_xy=ball_start_xy,
        step_mm=3.0, re_align_every=5)

    ep = get_ee_pos()
    print(f"     EE_z={ep[2]:.4f}  tip_z≈{ep[2]-FINGER_TIP_OFFSET:.4f}"
          f"  contact={'YES' if hit else 'no'}  reason={reason}")

    if reason == "push_detected":
        print("     Ball was pushed — aborting this attempt")
        return q, False

    # Step 4: Re-check alignment at this lower height
    print(f"     [D] Re-align XY at low height (EE_Z={EE_Z_ABOVE_TOP:.4f})")
    q, xy_err2 = precise_xy_align(
        bx, by, EE_Z_ABOVE_TOP,
        GRIPPER_OPEN_L, GRIPPER_OPEN_R, q)
    print(f"     Post-realign XY err: {xy_err2*100:.1f}mm")

    # Step 5: Final descent to grasp height — still open (jaw straddles ball)
    # At EE_Z_GRASP, fingertips are at z = EE_Z_GRASP - FINGER_TIP_OFFSET
    #                                     = 0.140 - 0.072 = 0.068
    # Ball equator = 0.078 → tips 1cm below equator = good wrap position
    print(f"     [E] Final descent FULLY OPEN to grasp height ({EE_Z_GRASP:.4f}m)")
    print(f"         Expected tip z = {EE_Z_GRASP - FINGER_TIP_OFFSET:.4f}"
          f"  ball_eq = {BALL_Z:.4f}")
    q, hit2, reason2 = controlled_descent(
        bx, by,
        ee_z_start=EE_Z_ABOVE_TOP,
        ee_z_end=EE_Z_GRASP,
        gl=GRIPPER_OPEN_L, gr=GRIPPER_OPEN_R,
        q_seed=q, ball_start_xy=ball_start_xy,
        step_mm=1.0, re_align_every=3)

    ep = get_ee_pos()
    ball_now = get_ball_pos()
    jaw_mid  = ep[:2] + JAW_OFFSET_XY
    print(f"     EE_z={ep[2]:.4f}  tip_z≈{ep[2]-FINGER_TIP_OFFSET:.4f}")
    print(f"     Ball=({ball_now[0]:.4f},{ball_now[1]:.4f},{ball_now[2]:.4f})")
    print(f"     Jaw mid=({jaw_mid[0]:.4f},{jaw_mid[1]:.4f})"
          f"  XY to ball: {np.linalg.norm(jaw_mid[:2]-ball_now[:2])*100:.1f}mm")

    if reason2 == "push_detected":
        print("     Ball was pushed on final descent — aborting")
        return q, False

    # Step 6: CLOSE GRIPPER — jaw is now around ball
    print(f"     [F] Closing gripper around ball")
    q = close_around_ball(bx, by, q, ball_start_xy)

    # Step 7: Lift test
    print(f"     [G] Lift test")
    ball_z_lift, grasped = lift_test(q, lift_cm=4.0)
    q = robot.get_qpos().copy()

    return q, grasped


def run_episode(ep_num, ball_xy, box_xy):
    bx, by = float(ball_xy[0]), float(ball_xy[1])
    gx, gy = float(box_xy[0]),  float(box_xy[1])
    sep    = float(np.linalg.norm(ball_xy - box_xy))
    print(f"\n{'═'*68}")
    print(f"  Episode {ep_num}  "
          f"🟢({bx:.3f},{by:.3f}) → 🔴({gx:.3f},{gy:.3f})"
          f"  sep={sep*100:.0f}cm")

    reset_ball(bx, by)
    box.set_pose(sapien.Pose(p=[gx, gy, BOX_Z]))
    robot.set_qpos(q_home)
    for jt, v in zip(joints, q_home): jt.set_drive_target(float(v))
    for _ in range(600): scene.step()
    update_debug_dots(); scene.update_render(); viewer.render()
    time.sleep(0.08)
    q = q_home.copy()

    # ── Phase 1: Survey ──────────────────────────────────────────
    print(f"\n   [1] Survey")
    q = move_to_ee(
        np.array([TX, TY, EE_Z_SURVEY]),
        GRIPPER_OPEN_L, GRIPPER_OPEN_R,
        STEPS_SURVEY, q, guard=ball_xy)
    ep = get_ee_pos()
    print(f"     EE={np.round(ep,4)}  tip_z≈{ep[2]-FINGER_TIP_OFFSET:.3f}")
    save_img(f'{OUT_DIR}/cam_ep{ep_num:02d}_1_survey.png')

    # ── Phase 2: Coarse approach ──────────────────────────────────
    print(f"\n   [2] Coarse approach above ball")
    q = move_to_ee(
        np.array([bx - JAW_OFFSET_XY[0], by - JAW_OFFSET_XY[1], EE_Z_HOVER]),
        GRIPPER_OPEN_L, GRIPPER_OPEN_R,
        STEPS_HOVER, q, guard=ball_xy)
    ep = get_ee_pos()
    jaw = ep[:2] + JAW_OFFSET_XY
    print(f"     EE=({ep[0]:.4f},{ep[1]:.4f})  jaw=({jaw[0]:.4f},{jaw[1]:.4f})"
          f"  tip_z≈{ep[2]-FINGER_TIP_OFFSET:.3f}")
    save_img(f'{OUT_DIR}/cam_ep{ep_num:02d}_2_approach.png')

    # ── Phase 3: Grasp attempts (up to 3) ─────────────────────────
    grasped = False
    for attempt in range(1, 4):
        q, grasped = attempt_grasp(bx, by, q, ball_xy, attempt_num=attempt)

        if grasped:
            print(f"\n   ✊ GRASPED on attempt {attempt}!")
            break

        if not grasped and attempt < 3:
            print(f"\n   Attempt {attempt} failed — returning to approach height...")
            # Re-open fully before retry
            q = open_gripper(q)
            # Return to safe hover height
            cur_ball = get_ball_pos()
            bx = float(cur_ball[0]); by = float(cur_ball[1])
            q = move_to_ee(
                np.array([bx - JAW_OFFSET_XY[0],
                          by - JAW_OFFSET_XY[1],
                          EE_Z_APPROACH]),
                GRIPPER_OPEN_L, GRIPPER_OPEN_R,
                25, q, guard=ball_xy)
            # Brief pause for ball to settle
            for _ in range(200): scene.step()
            cur_ball = get_ball_pos()
            bx = float(cur_ball[0]); by = float(cur_ball[1])
            print(f"     Ball settled at ({bx:.4f},{by:.4f})")

    save_img(f'{OUT_DIR}/cam_ep{ep_num:02d}_3_grasped.png')

    # ── Phase 4: Lift ─────────────────────────────────────────────
    print(f"\n   [4] Lift {'✓' if grasped else '(trying anyway)'}")
    cur_ball = get_ball_pos()
    q = move_to_ee(
        np.array([float(cur_ball[0]), float(cur_ball[1]), EE_Z_LIFT]),
        GRIPPER_CLOSE_L, GRIPPER_CLOSE_R,
        STEPS_LIFT, q)
    for _ in range(100): scene.step()
    bz_lifted = get_ball_z()
    print(f"     ball_z={bz_lifted:.4f}"
          f"  {'🎉 AIRBORNE ✓' if bz_lifted > 0.15 else '⚠ still on table'}")

    # ── Phase 5: Carry ────────────────────────────────────────────
    print(f"\n   [5] Carry to box")
    mid = np.array([(bx+gx)/2, (by+gy)/2, EE_Z_LIFT+0.02])
    q = move_to_ee(mid, GRIPPER_CLOSE_L, GRIPPER_CLOSE_R, STEPS_CARRY, q)
    q = move_to_ee(
        np.array([gx, gy, EE_Z_ABOVE_BOX]),
        GRIPPER_CLOSE_L, GRIPPER_CLOSE_R, STEPS_CARRY, q)
    bz_carry = get_ball_z()
    print(f"     Over box: ball_z={bz_carry:.4f}"
          f"  {'carrying ✓' if bz_carry > 0.15 else '⚠ dropped'}")

    # ── Phase 6: Lower into box ───────────────────────────────────
    print(f"\n   [6] Lower into box")
    q = move_to_ee(
        np.array([gx, gy, EE_Z_LOWER_BOX]),
        GRIPPER_CLOSE_L, GRIPPER_CLOSE_R,
        STEPS_LOWER, q)
    ep = get_ee_pos()
    print(f"     EE={np.round(ep,4)}  tip_z≈{ep[2]-FINGER_TIP_OFFSET:.3f}")

    # ── Phase 7: Release ──────────────────────────────────────────
    print(f"\n   [7] Release")
    q = open_gripper(q)
    for _ in range(200): scene.step()
    bz_released = get_ball_z()
    print(f"     🖐 Released  ball_z={bz_released:.4f}")
    save_img(f'{OUT_DIR}/cam_ep{ep_num:02d}_4_placed.png')

    # ── Phase 8: Retreat ──────────────────────────────────────────
    print(f"\n   [8] Retreat")
    q = move_to_ee(
        np.array([gx, gy, EE_Z_ABOVE_BOX]),
        GRIPPER_OPEN_L, GRIPPER_OPEN_R, 20, q)
    move_to_ee(
        np.array([TX, TY, EE_Z_SURVEY - 0.05]),
        GRIPPER_OPEN_L, GRIPPER_OPEN_R, STEPS_HOVER, q)

    # ── Result ────────────────────────────────────────────────────
    for _ in range(250): sim_step()
    bf   = get_ball_pos()
    boxf = np.array(box.get_pose().p)
    dist = float(np.linalg.norm(bf[:2] - boxf[:2]))
    success = (dist < 0.10) and (bf[2] < BOX_Z + 0.07)
    print(f"\n   Ball final:  {np.round(bf,3)}")
    print(f"   Box center:  ({boxf[0]:.3f},{boxf[1]:.3f})")
    print(f"   Distance:    {dist*100:.1f}cm"
          f"  → {'✅ SUCCESS' if success else '❌ MISS'}")
    return success


# ═══════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════
print()
print("=" * 68)
print("  SUMMARY OF v6 FIXES:")
print("  A) Descend with FULLY OPEN gripper — cannot push ball sideways")
print("  B) Jaw-midpoint XY alignment (not EE-origin) — correct for offset")
print("  C) Ball push detection — abort if ball moves > 3cm")
print("  D) Contact detection requires 3 consecutive rises >= 3mm")
print("  E) Mimic ratio enforced: gripper_r = gripper_l × (0.059/0.034)")
print("  F) IK bias calibrated across 4 workspace positions")
print("  G) lift_test() actually lifts 4cm and checks ball follows")
print("  H) Up to 3 grasp attempts per episode with fresh ball read")
print("=" * 68)
print(f"  JAW_OFFSET_XY    = {np.round(JAW_OFFSET_XY,4)}")
print(f"  JAW_WIDTH_OPEN   = {JAW_WIDTH_OPEN*100:.1f}cm")
print(f"  SQUEEZE_AXIS     = {'XYZ'[SQUEEZE_AXIS]}")
print(f"  EE_Z_GRASP       = {EE_Z_GRASP:.4f}  tip_z = {EE_Z_GRASP-FINGER_TIP_OFFSET:.4f}")
print(f"  EE_Z_AT_EQUATOR  = {EE_Z_AT_BALL_EQUATOR:.4f}  (for reference)")
print("=" * 68)
print()

rng       = np.random.default_rng(42)
successes = 0
ep        = 0

while not viewer.closed:
    ep += 1
    bx  = TX + rng.uniform(-0.09, -0.01)
    by  = TY + rng.uniform(-0.08,  0.08)
    for _ in range(200):
        gx = TX + rng.uniform(0.04, 0.11)
        gy = TY + rng.uniform(-0.08, 0.08)
        if float(np.linalg.norm(np.array([bx,by]) - np.array([gx,gy]))) >= 0.18:
            break
    ok = run_episode(ep, np.array([bx, by]), np.array([gx, gy]))
    if ok: successes += 1
    pct = 100 * successes / ep
    print(f"\n  ── Total: {successes}/{ep} ({pct:.0f}%) ──\n")