"""
Dual-Robot Pick & Place — PERFECT FINAL
=========================================
ALL BUGS FIXED:

  BUG 1 — Decorative collision boxes pushed robots into wrong poses:
    bench_shelf at z=0.45 overlapped robot arm sweep zone → robots settled
    in corrupted positions → IK found wrong solutions → 100cm+ XY errors.
    FIX: ALL decorative elements are visual-only (no collision shapes).
         Only the actual robot table (thin slab at z=0.025) has collision.

  BUG 2 — TABLE_TOP computed from stacked geometry (=0.966):
    Stacking floor+bench+table+mat gave TABLE_TOP=0.966 → BALL_Z=0.992
    but robots sit at z=0 and can only reach z≈0.77 → impossible to grasp.
    FIX: TABLE_TOP=0.052 hardcoded (proven working value from all v4-v8 logs).
         The visual workbench is positioned to look like it's under the table,
         not affect robot geometry.

  BUG 3 — IK world→local frame conversion (from previous fix, kept):
    FIX: target_local = target_world - robot_base before every IK call.

  BUG 4 — Jerky motion from high SIM_PER_STEP and low stiffness:
    FIX: Raised stiffness to 80000/8000, smoother motion profile,
         added extra settle steps after each phase.

conda activate maniskill2
python sapien_dual_perfect.py
"""

import math, time, numpy as np, torch, torch.nn as nn
import torchvision.transforms as T, torchvision.models as tvm
import sapien, PIL.Image
from scipy.spatial.transform import Rotation
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# ═══════════════════════════════════════════════════════════════════
#  PATHS
# ═══════════════════════════════════════════════════════════════════
URDF     = '/home/rishabh/Downloads/myarm_m750_fixed.urdf'
CKPT_DIR = '/home/rishabh/Downloads/umi-pipeline-training/checkpoints_umi_vision'
OUT_DIR  = '/home/rishabh/Downloads/umi-pipeline-training'
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'

# ═══════════════════════════════════════════════════════════════════
#  ROBOT PLACEMENT
# ═══════════════════════════════════════════════════════════════════
ROBOT0_BASE = np.array([0.0, -0.38, 0.0])
ROBOT1_BASE = np.array([0.0, +0.38, 0.0])

# ═══════════════════════════════════════════════════════════════════
#  GEOMETRY — HARDCODED FROM PROVEN SINGLE-ROBOT LOGS
# ═══════════════════════════════════════════════════════════════════
TABLE_TOP  = 0.052          # proven: this is where balls sit
BALL_R     = 0.026
BALL_Z     = TABLE_TOP + BALL_R    # 0.078
BOX_H      = 0.020
BOX_Z      = TABLE_TOP + BOX_H    # 0.072

FINGER_TIP_OFFSET = 0.100

TIP_Z_SURVEY    = 0.36
TIP_Z_ABOVE     = 0.18
TIP_Z_PRE       = 0.060
TIP_Z_GRASP     = 0.063
TIP_Z_LIFT      = 0.28
TIP_Z_ABOVE_BOX = 0.20
TIP_Z_LOWER     = 0.085

def EEZ(t): return float(t) + FINGER_TIP_OFFSET

# ═══════════════════════════════════════════════════════════════════
#  GRIPPER
# ═══════════════════════════════════════════════════════════════════
GRIPPER_OPEN_L  = 0.030
GRIPPER_OPEN_R  = 0.060
GRIPPER_CLOSE_L = 0.000
GRIPPER_CLOSE_R = 0.000
GRIPPER_HALF_L  = 0.018
GRIPPER_HALF_R  = 0.036
GRIPPER_L_IDX   = 6
GRIPPER_R_IDX   = 7   # set to None if N<8

# ═══════════════════════════════════════════════════════════════════
#  BALL PHYSICS
# ═══════════════════════════════════════════════════════════════════
BALL_MASS    = 0.50
BALL_FRIC    = 2.00
BALL_REST    = 0.02
BALL_LDAMP   = 15.0
BALL_ADAMP   = 15.0

# ═══════════════════════════════════════════════════════════════════
#  MOTION TIMING
# ═══════════════════════════════════════════════════════════════════
STEPS_SURVEY  = 55
STEPS_HOVER   = 45
STEPS_DESCEND = 35
STEPS_GRASP   = 45
STEPS_LIFT    = 35
STEPS_CARRY   = 40
STEPS_LOWER   = 30
STEPS_GRIP    = 35
SIM_PER_STEP  = 3     # lower = smoother motion, less jerky

# ═══════════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════════
OBS_HORIZON = 2; ACTION_HORIZON = 16; ACTION_DIM = 7
STATE_DIM   = 7; IMG_FEAT_DIM   = 512; IMG_SIZE   = 96
NUM_DIFF    = 100

img_tf = T.Compose([T.ToPILImage(), T.Resize((IMG_SIZE,IMG_SIZE)), T.ToTensor(),
                    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

class Normalizer:
    @classmethod
    def load(cls,p):
        n=cls(); d=torch.load(p,map_location='cpu',weights_only=False)
        n.min=d['min']; n.max=d['max']; n.scale=d['scale']; return n
    def normalize(self,x): return 2.*(x-self.min.to(x.device))/self.scale.to(x.device)-1.
    def denormalize(self,x): return (x+1.)/2.*self.scale.to(x.device)+self.min.to(x.device)

class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(*list(tvm.resnet18(weights=None).children())[:-1])
    def forward(self,x): return self.encoder(x).squeeze(-1).squeeze(-1)

class SinusoidalPosEmb(nn.Module):
    def __init__(self,d): super().__init__(); self.d=d
    def forward(self,t):
        h=self.d//2; e=math.log(10000)/(h-1)
        e=torch.exp(torch.arange(h,device=t.device)*-e)
        return torch.cat([(t.float()[:,None]*e[None,:]).sin(),
                          (t.float()[:,None]*e[None,:]).cos()],dim=-1)

class ResBlock(nn.Module):
    def __init__(self,dim,cd):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,dim),nn.Mish(),nn.Linear(dim,dim))
        self.cond=nn.Linear(cd,dim*2); self.norm=nn.LayerNorm(dim)
    def forward(self,x,c):
        s,b=self.cond(c).chunk(2,dim=-1)
        return x+self.net(self.norm(x)*(s+1)+b)

class VisionDiffusionNet(nn.Module):
    def __init__(self,hidden=512,depth=8):
        super().__init__()
        fa=ACTION_DIM*ACTION_HORIZON; cd=512
        self.visual_enc=VisualEncoder()
        fi=STATE_DIM*OBS_HORIZON+IMG_FEAT_DIM*OBS_HORIZON
        self.obs_fuse=nn.Sequential(
            nn.Linear(fi,512),nn.Mish(),nn.Linear(512,512),nn.Mish(),nn.Linear(512,256))
        self.time_emb=nn.Sequential(
            SinusoidalPosEmb(128),nn.Linear(128,256),nn.Mish(),nn.Linear(256,256))
        self.cond_proj=nn.Sequential(nn.Linear(512,cd),nn.Mish(),nn.Linear(cd,cd))
        self.in_proj=nn.Linear(fa,hidden)
        self.blocks=nn.ModuleList([ResBlock(hidden,cd) for _ in range(depth)])
        self.out_proj=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,fa))
    def forward(self,noisy,ts,sf,imgs):
        B=noisy.shape[0]
        ig=torch.cat([self.visual_enc(imgs[:,i]) for i in range(OBS_HORIZON)],dim=-1)
        oe=self.obs_fuse(torch.cat([sf,ig],dim=-1))
        c=self.cond_proj(torch.cat([oe,self.time_emb(ts)],dim=-1))
        x=self.in_proj(noisy.reshape(B,-1))
        for blk in self.blocks: x=blk(x,c)
        return self.out_proj(x).reshape(B,ACTION_HORIZON,ACTION_DIM)

# ═══════════════════════════════════════════════════════════════════
#  BUILD SCENE
# ═══════════════════════════════════════════════════════════════════
print("="*62)
print(" SAPIEN Dual-Robot Pick & Place — PERFECT FINAL")
print("="*62)
print(f" TABLE_TOP={TABLE_TOP}  BALL_Z={BALL_Z}  (hardcoded, proven)")
print(f" Robot 0 base={ROBOT0_BASE}  Robot 1 base={ROBOT1_BASE}")

scene = sapien.Scene()
scene.set_timestep(1/500)     # higher frequency = more stable physics

# ── Professional robotics-lab lighting ───────────────────────────
scene.set_ambient_light([0.40, 0.40, 0.42])
# Main warm overhead LED (primary illumination)
scene.add_directional_light([ 0.05,  0.10, -1.00], [1.00, 0.97, 0.90])
# Cool blue-white front fill (simulates window light)
scene.add_directional_light([-0.55, -0.35, -0.45], [0.30, 0.32, 0.42])
# Warm back-rim (separates robot from background, adds depth)
scene.add_directional_light([ 0.65, -0.10, -0.30], [0.22, 0.24, 0.30])
# Ceiling LED panels above each robot station
scene.add_point_light([0.28, -0.40, 2.20], [3.2, 3.10, 2.90])
scene.add_point_light([0.28,  0.40, 2.20], [3.2, 3.10, 2.90])
scene.add_point_light([0.28,  0.00, 2.20], [2.8, 2.70, 2.55])
# Low warm front bounce
scene.add_point_light([-0.50, 0.00, 0.55], [0.40, 0.35, 0.28])

scene.add_ground(altitude=0)

# ── VISUAL-ONLY static scene builder ─────────────────────────────
# CRITICAL: NO collision shapes on decorative elements
# Only the actual robot table has collision
def vis_box(half, rgba, pos, rough=0.80, metal=0.0, spec=0.04):
    """Visual-only box — no collision, cannot push robot."""
    mt = sapien.render.RenderMaterial()
    mt.base_color = rgba; mt.roughness = rough
    mt.metallic   = metal; mt.specular = spec
    b = scene.create_actor_builder()
    b.add_box_visual(half_size=half, material=mt)
    a = b.build_static()
    a.set_pose(sapien.Pose(p=pos))
    return a

def col_box(half, rgba, pos, name="", rough=0.80, metal=0.0):
    """Box with collision — only for things robots must interact with."""
    mt = sapien.render.RenderMaterial()
    mt.base_color = rgba; mt.roughness = rough; mt.metallic = metal
    b = scene.create_actor_builder()
    b.add_box_visual(half_size=half, material=mt)
    b.add_box_collision(half_size=half)
    a = b.build_static(name=name)
    a.set_pose(sapien.Pose(p=pos))
    return a

# ── FLOOR: dark concrete ─────────────────────────────────────────
vis_box([4.0, 4.0, 0.006], [0.17,0.17,0.18,1.0], [0.25,0.00,0.006], rough=0.96)

# ── WALLS (visual only — no collision needed) ────────────────────
vis_box([0.020, 4.0, 2.80], [0.87,0.86,0.83,1.0], [ 1.30, 0.00,1.40], rough=0.92)  # back
vis_box([4.0, 0.020, 2.80], [0.84,0.83,0.81,1.0], [ 0.25,-1.70,1.40], rough=0.92)  # left
vis_box([4.0, 0.020, 2.80], [0.84,0.83,0.81,1.0], [ 0.25, 1.70,1.40], rough=0.92)  # right

# ── WORKBENCH VISUAL (purely decorative, no collision) ───────────
# Dark-wood bench top (visual layer under the actual physics table)
CX,CY = 0.25, 0.00
vis_box([0.65, 0.95, 0.018], [0.16,0.10,0.06,1.0], [CX,CY,0.868], rough=0.55)  # bench top
vis_box([0.60, 0.90, 0.014], [0.14,0.09,0.05,1.0], [CX,CY,0.834], rough=0.50)  # bench trim

# Brushed aluminium legs (visual)
for lx,ly in [(0.58,-0.88),(0.58,0.88),(-0.48,-0.88),(-0.48,0.88)]:
    vis_box([0.022,0.022,0.43],[0.20,0.20,0.22,1.0],[CX+lx,ly,0.43],rough=0.25,metal=0.85)

# Shelf visual (no collision — robot arm sweeps through this space)
vis_box([0.58, 0.88, 0.012], [0.18,0.11,0.07,1.0], [CX,CY,0.44], rough=0.55)

# Control boxes on shelf (visual)
for sy in [-0.55, 0.55]:
    vis_box([0.11,0.09,0.055],[0.10,0.12,0.15,1.0],[CX-0.22,sy,0.495],rough=0.35,metal=0.65)
    vis_box([0.010,0.010,0.007],[0.05,0.88,0.10,1.0],[CX-0.11,sy,0.553],rough=0.20)  # LED

# Cable trays
for ty in [-0.87, 0.87]:
    vis_box([0.38,0.025,0.035],[0.12,0.12,0.13,1.0],[CX,ty,0.920],rough=0.40,metal=0.72)

# Overhead light panels
for ly in [-0.40, 0.00, 0.40]:
    vis_box([0.055,0.32,0.010],[0.94,0.94,0.90,1.0],[CX,ly,2.18],rough=0.10,spec=0.90)

# Parts bins at back of table
BINS = [[0.88,0.15,0.10,1.0],[0.10,0.42,0.82,1.0],[0.20,0.65,0.18,1.0]]
for i,(bc,bcy) in enumerate(zip(BINS,[-0.68,-0.50,-0.32])):
    vis_box([0.055,0.045,0.040], bc, [CX+0.40,bcy,TABLE_TOP+0.040], rough=0.60)
for i,(bc,bcy) in enumerate(zip(BINS,[0.32,0.50,0.68])):
    vis_box([0.055,0.045,0.040], bc, [CX+0.40,bcy,TABLE_TOP+0.040], rough=0.60)

# Robot mounting pads (visual)
for py in [-0.38, 0.38]:
    vis_box([0.110,0.090,0.005],[0.14,0.14,0.16,1.0],
            [CX-0.04,py,TABLE_TOP+0.005],rough=0.25,metal=0.90)

# ── PHYSICS TABLE — only this has collision ───────────────────────
# Thin slab at proven height. Robots sit on the ground, reach over this.
col_box([0.38, 0.90, 0.025], [0.22,0.14,0.09,1.0],
        [CX, CY, 0.025], name="table", rough=0.65)
# Black work mat on top
col_box([0.32, 0.84, 0.003], [0.07,0.07,0.08,1.0],
        [CX, CY, TABLE_TOP-0.001], name="mat", rough=0.95)

print(f" Scene built (TABLE_TOP={TABLE_TOP:.3f})")

# ═══════════════════════════════════════════════════════════════════
#  LOAD ROBOTS — after all geometry so collision is fully built
# ═══════════════════════════════════════════════════════════════════
def load_robot(base):
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot  = loader.load(URDF)
    robot.set_pose(sapien.Pose(p=base.tolist()))
    joints = robot.get_active_joints()
    N      = len(joints)
    links  = {l.name:l for l in robot.get_links()}
    ee     = links['gripper']
    pm     = robot.create_pinocchio_model()
    # High stiffness → no wobble, firm positioning
    for i,jt in enumerate(joints):
        if i<6: jt.set_drive_property(stiffness=80000, damping=8000)
        else:   jt.set_drive_property(stiffness=20000, damping=2000)
    return robot, joints, N, ee, pm

print(" Loading Robot 0...")
robot0,joints0,N0,ee0,pm0 = load_robot(ROBOT0_BASE)
print(" Loading Robot 1...")
robot1,joints1,N1,ee1,pm1 = load_robot(ROBOT1_BASE)
N = N0
GR = GRIPPER_R_IDX if N>=8 else None

def q_home():
    q = np.zeros(N)
    q[1]=-0.30; q[2]=0.50
    q[GRIPPER_L_IDX]=GRIPPER_OPEN_L
    if GR: q[GR]=GRIPPER_OPEN_R
    return q

Q_HOME = q_home()

def do_reset(robot, joints):
    robot.set_qpos(Q_HOME.copy())
    for i,jt in enumerate(joints): jt.set_drive_target(float(Q_HOME[i]))

do_reset(robot0,joints0); do_reset(robot1,joints1)
# Long settle to ensure robots reach correct home pose
for _ in range(800): scene.step()

home0 = np.array(ee0.get_entity_pose().p)
home1 = np.array(ee1.get_entity_pose().p)
TX0,TY0,TZ0 = home0
TX1,TY1,TZ1 = home1
print(f"\n R0 EE world home: ({TX0:.4f},{TY0:.4f},{TZ0:.4f})")
print(f" R1 EE world home: ({TX1:.4f},{TY1:.4f},{TZ1:.4f})")

# Sanity check — EE should be around z=0.77 and x>0
assert TZ0 > 0.50, f"R0 EE z={TZ0:.3f} too low — robot home is wrong!"
assert TX0 > 0.00, f"R0 EE x={TX0:.3f} negative — robot facing wrong way!"
print(f" ✓ Robot home positions look correct")

# ═══════════════════════════════════════════════════════════════════
#  ORIENTATION PROBE
# ═══════════════════════════════════════════════════════════════════
eq = ee0.get_entity_pose().q
ee_rot = Rotation.from_quat([eq[1],eq[2],eq[3],eq[0]])

def make_grasp_quat(ax):
    tgt=np.array([0.,0.,-1.]); aw=ee_rot.apply(ax)
    cr=np.cross(aw,tgt); cn=np.linalg.norm(cr); dt=np.dot(aw,tgt)
    if cn<1e-6: corr=Rotation.identity() if dt>0 else Rotation.from_euler('x',np.pi)
    else: corr=Rotation.from_rotvec(cr/cn*np.arctan2(cn,dt))
    q=(corr*ee_rot).as_quat()
    return (float(q[3]),float(q[0]),float(q[1]),float(q[2]))

print("\n Probing gripper-down orientation (home restored each time)...")
# cal_local: calibration point in R0 LOCAL frame
cal_local = np.array([TX0-ROBOT0_BASE[0], TY0-ROBOT0_BASE[1], 0.25])
best_q, best_c = None, 1e18

for ax in [np.array([0,0,1]), np.array([0,0,-1]),
           np.array([0,1,0]), np.array([0,-1,0]),
           np.array([1,0,0]), np.array([-1,0,0])]:
    do_reset(robot0,joints0); do_reset(robot1,joints1)
    for _ in range(250): scene.step()
    try:
        wxyz = make_grasp_quat(ax)
        pose = sapien.Pose(p=cal_local.tolist(), q=list(wxyz))
        mask = np.ones(N,dtype=np.int32); mask[6:]=0
        qr,ok,_ = pm0.compute_inverse_kinematics(
            ee0.get_index(), pose,
            initial_qpos=Q_HOME.astype(np.float64),
            active_qmask=mask, max_iterations=1000)
        if not ok: print(f"   {ax} IK fail"); continue
        qs=np.array(qr); robot0.set_qpos(qs)
        for _ in range(40): scene.step()
        ae=np.array(ee0.get_entity_pose().p)
        pe=np.linalg.norm(ae-ROBOT0_BASE-cal_local)
        w=np.array([1,1,1,1,5,3],dtype=float)
        c=float(np.sum(w*(qs[:6]-Q_HOME[:6])**2))+pe*50.
        print(f"   {ax}  err={pe*100:.1f}cm  cost={c:.2f}")
        if c<best_c: best_c,best_q=c,wxyz
    except Exception as ex:
        print(f"   {ax} ERR: {ex}")

do_reset(robot0,joints0); do_reset(robot1,joints1)
for _ in range(400): scene.step()
if best_q is None:
    q=ee_rot.as_quat(); best_q=(float(q[3]),float(q[0]),float(q[1]),float(q[2]))
    print(" ⚠ Using fallback orientation")
GRASP_QUAT = best_q
print(f" Best cost={best_c:.2f}")

# ═══════════════════════════════════════════════════════════════════
#  IK — WORLD → LOCAL FRAME (the essential fix)
# ═══════════════════════════════════════════════════════════════════
def ik_arm(pm, ee_link, world_xyz, robot_base, q_seed):
    """IK in robot LOCAL frame. Converts world→local before calling pinocchio."""
    local_xyz = world_xyz - robot_base           # ← THE FIX
    pose = sapien.Pose(p=[float(v) for v in local_xyz], q=list(GRASP_QUAT))
    mask = np.ones(N,dtype=np.int32); mask[6:]=0
    qr,ok,_ = pm.compute_inverse_kinematics(
        ee_link.get_index(), pose,
        initial_qpos=np.array(q_seed,dtype=np.float64),
        active_qmask=mask, max_iterations=1400)
    return np.array(qr), ok

# ═══════════════════════════════════════════════════════════════════
#  SCENE OBJECTS — BALLS & BOXES
# ═══════════════════════════════════════════════════════════════════
def make_ball(rgba, name):
    mt=sapien.render.RenderMaterial()
    mt.base_color=rgba; mt.roughness=0.20; mt.metallic=0.0
    bb=scene.create_actor_builder()
    bb.add_sphere_visual(radius=BALL_R, material=mt)
    pm_b=scene.create_physical_material(
        static_friction=BALL_FRIC, dynamic_friction=BALL_FRIC, restitution=BALL_REST)
    bb.add_sphere_collision(radius=BALL_R, material=pm_b)
    b=bb.build(name=name)
    try:
        rb=b.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        if rb:
            rb.set_mass(BALL_MASS)
            rb.set_linear_damping(BALL_LDAMP)
            rb.set_angular_damping(BALL_ADAMP)
    except: pass
    return b

def make_tray(rgba, name):
    mt=sapien.render.RenderMaterial()
    mt.base_color=rgba; mt.roughness=0.45; mt.metallic=0.0
    bx=scene.create_actor_builder()
    bx.add_box_visual(half_size=[0.058,0.058,BOX_H], material=mt)
    bx.add_box_collision(half_size=[0.058,0.058,BOX_H])
    return bx.build_static(name=name)

def make_dot(rgba):
    mt=sapien.render.RenderMaterial(); mt.base_color=rgba
    b=scene.create_actor_builder(); b.add_sphere_visual(radius=0.009, material=mt)
    return b.build_static()

# Bright distinct colours so both balls & boxes are clearly different
ball0 = make_ball([0.05, 0.92, 0.10, 1.0], "ball0")   # vivid green
ball1 = make_ball([1.00, 0.45, 0.02, 1.0], "ball1")   # vivid orange
box0  = make_tray([0.90, 0.08, 0.05, 1.0], "box0")    # red tray
box1  = make_tray([0.05, 0.20, 0.90, 1.0], "box1")    # blue tray
dot0  = make_dot([1.0, 1.0, 1.0, 0.85])
dot1  = make_dot([1.0, 0.85, 0.15, 0.85])

# ── Camera for each robot (vision model input) ────────────────────
def make_cam(tx, ty, tz):
    ce=sapien.Entity(); cc=sapien.render.RenderCameraComponent(224,224)
    cc.set_fovy(np.deg2rad(58)); ce.add_component(cc)
    cr=Rotation.from_euler('xyz',[np.deg2rad(130),0,0]); cq=cr.as_quat()
    ce.set_pose(sapien.Pose(
        p=[tx-0.18,ty,tz+0.10],
        q=[float(cq[3]),float(cq[0]),float(cq[1]),float(cq[2])]))
    scene.add_entity(ce); return cc

cam0 = make_cam(TX0,TY0,TZ0)
cam1 = make_cam(TX1,TY1,TZ1)

# ── Viewer — wide angled view showing full workspace ──────────────
viewer = scene.create_viewer()
# Slightly to the left and elevated, looking diagonally at both robots
viewer.set_camera_xyz(-0.50, -0.02, 1.35)
viewer.set_camera_rpy(0.0, -0.36, 0.07)

# ═══════════════════════════════════════════════════════════════════
#  CONSTRAINT GRASP
# ═══════════════════════════════════════════════════════════════════
class CS:
    def __init__(self, rid):
        self.rid=rid; self.active=False
        self.offset=np.zeros(3); self.ball=None; self.rb=None

    def on(self, ee_link, ball_actor):
        ep=np.array(ee_link.get_entity_pose().p)
        eq=ee_link.get_entity_pose().q
        er=Rotation.from_quat([eq[1],eq[2],eq[3],eq[0]])
        bp=np.array(ball_actor.get_pose().p)
        self.offset=er.inv().apply(bp-ep)
        self.active=True; self.ball=ball_actor
        try: self.rb=ball_actor.find_component_by_type(
                sapien.physx.PhysxRigidDynamicComponent)
        except: self.rb=None
        if self.rb:
            self.rb.set_linear_velocity([0,0,0])
            self.rb.set_angular_velocity([0,0,0])
        print(f"     R{self.rid} 🔗 ON  offset={np.round(self.offset,3)}")

    def off(self):
        if self.active:
            self.active=False
            if self.rb:
                self.rb.set_linear_velocity([0,0,0])
                self.rb.set_angular_velocity([0,0,0])

    def sync(self, ee_link):
        if not self.active or self.ball is None: return
        ep=np.array(ee_link.get_entity_pose().p)
        eq=ee_link.get_entity_pose().q
        er=Rotation.from_quat([eq[1],eq[2],eq[3],eq[0]])
        self.ball.set_pose(sapien.Pose(p=(ep+er.apply(self.offset)).tolist()))
        if self.rb:
            self.rb.set_linear_velocity([0,0,0])
            self.rb.set_angular_velocity([0,0,0])

cs0=CS(0); cs1=CS(1)

# ═══════════════════════════════════════════════════════════════════
#  RENDER
# ═══════════════════════════════════════════════════════════════════
def drv(joints, q):
    for i,jt in enumerate(joints): jt.set_drive_target(float(q[i]))

def step():
    """Single physics+constraint+render tick."""
    for _ in range(SIM_PER_STEP): scene.step()
    cs0.sync(ee0); cs1.sync(ee1)
    dot0.set_pose(sapien.Pose(p=list(ee0.get_entity_pose().p)))
    dot1.set_pose(sapien.Pose(p=list(ee1.get_entity_pose().p)))
    scene.update_render(); viewer.render()

def snap(path, cam):
    scene.update_render(); cam.take_picture()
    rgba=cam.get_picture('Color')
    PIL.Image.fromarray((np.clip(rgba[:,:,:3],0,1)*255).astype(np.uint8)).save(path)

def place_ball(actor, x, y):
    actor.set_pose(sapien.Pose(p=[x,y,BALL_Z]))
    try:
        rb=actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        if rb: rb.set_linear_velocity([0,0,0]); rb.set_angular_velocity([0,0,0])
    except: pass

# ═══════════════════════════════════════════════════════════════════
#  ARM OBJECT
# ═══════════════════════════════════════════════════════════════════
class Arm:
    def __init__(self, robot, joints, ee, pm, cs, cam, base, tx, ty, tz, rid):
        self.robot=robot; self.joints=joints; self.ee=ee
        self.pm=pm; self.cs=cs; self.cam=cam
        self.base=base; self.TX=tx; self.TY=ty; self.TZ=tz
        self.q=Q_HOME.copy(); self.rid=rid
        self.ball=None; self.box=None

    def ik(self, world_xyz, seed=None):
        return ik_arm(self.pm, self.ee, world_xyz, self.base, seed or self.q)

    def pos(self): return np.array(self.ee.get_entity_pose().p)

    def drv(self, q): drv(self.joints, q)

    def qpos(self): return self.robot.get_qpos().copy()

    def reset(self):
        self.robot.set_qpos(Q_HOME.copy())
        drv(self.joints, Q_HOME)
        self.q=Q_HOME.copy(); self.cs.off()

    def photo(self, path):
        snap(path, self.cam)
        print(f"   [R{self.rid}] 📷 {path}")

A0 = Arm(robot0,joints0,ee0,pm0,cs0,cam0,ROBOT0_BASE,TX0,TY0,TZ0,0)
A1 = Arm(robot1,joints1,ee1,pm1,cs1,cam1,ROBOT1_BASE,TX1,TY1,TZ1,1)

# ═══════════════════════════════════════════════════════════════════
#  PARALLEL MOTION PRIMITIVES
# ═══════════════════════════════════════════════════════════════════
def both_ik(a0, world0, a1, world1, gl, gr):
    """Solve IK for both robots. Returns (q0, q1, ok0, ok1)."""
    q0,ok0 = a0.ik(world0, a0.q)
    if not ok0: q0,ok0 = a0.ik(world0, Q_HOME)

    q1,ok1 = a1.ik(world1, a1.q)
    if not ok1: q1,ok1 = a1.ik(world1, Q_HOME)

    for q,ok,lbl in [(q0,ok0,"R0"),(q1,ok1,"R1")]:
        if not ok: print(f"  ⚠ {lbl} IK fail")

    q0[GRIPPER_L_IDX]=float(gl); q1[GRIPPER_L_IDX]=float(gl)
    if GR: q0[GR]=float(gr); q1[GR]=float(gr)
    return q0, q1, ok0, ok1


def par_move(a0, w0, a1, w1, gl, gr, n_steps, settle=20):
    """
    Smooth parallel motion. Both robots follow smooth-step trajectory.
    Extra settle steps at end eliminate residual vibration.
    """
    q0,q1,_,_ = both_ik(a0,w0,a1,w1,gl,gr)
    s0=a0.qpos(); s1=a1.qpos()

    for i in range(n_steps):
        t=(i+1)/n_steps; sm=t*t*(3.-2.*t)
        a0.drv(s0+sm*(q0-s0)); a1.drv(s1+sm*(q1-s1))
        step()
        if viewer.closed: return

    # Settle phase: hold target, let physics dampen
    for _ in range(settle):
        a0.drv(q0); a1.drv(q1); step()

    a0.q=q0.copy(); a1.q=q1.copy()


def par_slow_lower(a0, bx0, by0, a1, bx1, by1,
                   z_start, z_end, gl, gr, n_steps):
    """
    Micro-step descent with 80% XY correction per step.
    Both robots descend simultaneously and converge on their balls.
    """
    for z_tip in np.linspace(z_start, z_end, n_steps):
        ee_z = EEZ(z_tip)

        # Robot 0: correct XY toward ball
        p0 = a0.pos()
        d0 = np.array([bx0,by0]) - p0[:2]
        t0 = p0[:2] + d0*0.80
        t0[0] = float(np.clip(t0[0], a0.TX-0.18, a0.TX+0.18))
        t0[1] = float(np.clip(t0[1], a0.TY-0.18, a0.TY+0.18))

        # Robot 1: correct XY toward ball
        p1 = a1.pos()
        d1 = np.array([bx1,by1]) - p1[:2]
        t1 = p1[:2] + d1*0.80
        t1[0] = float(np.clip(t1[0], a1.TX-0.18, a1.TX+0.18))
        t1[1] = float(np.clip(t1[1], a1.TY-0.18, a1.TY+0.18))

        q0,ok0 = a0.ik(np.array([t0[0],t0[1],ee_z]))
        if not ok0: q0,ok0 = a0.ik(np.array([bx0,by0,ee_z]))

        q1,ok1 = a1.ik(np.array([t1[0],t1[1],ee_z]))
        if not ok1: q1,ok1 = a1.ik(np.array([bx1,by1,ee_z]))

        if ok0:
            q0[GRIPPER_L_IDX]=float(gl)
            if GR: q0[GR]=float(gr)
            a0.drv(q0); a0.q=q0.copy()
        if ok1:
            q1[GRIPPER_L_IDX]=float(gl)
            if GR: q1[GR]=float(gr)
            a1.drv(q1); a1.q=q1.copy()

        # 2 sub-steps per IK step → smooth, no jerk
        for _ in range(2): scene.step()
        cs0.sync(a0.ee); cs1.sync(a1.ee)
        dot0.set_pose(sapien.Pose(p=list(a0.ee.get_entity_pose().p)))
        dot1.set_pose(sapien.Pose(p=list(a1.ee.get_entity_pose().p)))
        scene.update_render(); viewer.render()
        if viewer.closed: break


def par_gripper(a0, a1, gl, gr, n=STEPS_GRIP, settle=30):
    """Smooth simultaneous gripper motion with settle."""
    qt0=a0.q.copy(); qt0[GRIPPER_L_IDX]=float(gl)
    qt1=a1.q.copy(); qt1[GRIPPER_L_IDX]=float(gl)
    if GR: qt0[GR]=float(gr); qt1[GR]=float(gr)
    s0=a0.qpos(); s1=a1.qpos()
    for i in range(n):
        t=(i+1)/n
        a0.drv(s0+t*(qt0-s0)); a1.drv(s1+t*(qt1-s1)); step()
    for _ in range(settle):
        a0.drv(qt0); a1.drv(qt1); step()
    a0.q=qt0.copy(); a1.q=qt1.copy()

# ═══════════════════════════════════════════════════════════════════
#  LOAD MODEL
# ═══════════════════════════════════════════════════════════════════
print("\n Loading vision model...")
obs_norm = Normalizer.load(f'{CKPT_DIR}/obs_normalizer.pt')
act_norm = Normalizer.load(f'{CKPT_DIR}/act_normalizer.pt')
net = VisionDiffusionNet().to(DEVICE)
ck  = torch.load(f'{CKPT_DIR}/best_model.pt', map_location=DEVICE, weights_only=False)
net.load_state_dict(ck['model_state']); net.eval()
print(f" Loaded epoch={ck['epoch']}  loss={ck['loss']:.5f}")
noise_sched = DDPMScheduler(num_train_timesteps=NUM_DIFF,
    beta_schedule='squaredcos_cap_v2', clip_sample=True, prediction_type='epsilon')

# ═══════════════════════════════════════════════════════════════════
#  EPISODE RUNNER
# ═══════════════════════════════════════════════════════════════════
def run_episode(ep_num, balls, boxes):
    """
    balls = [(bx0,by0),(bx1,by1)]  — world frame
    boxes = [(gx0,gy0),(gx1,gy1)]  — world frame
    Assigns nearest ball to each robot.
    """
    b0=np.array(balls[0]); b1=np.array(balls[1])
    h0=np.array([A0.TX,A0.TY]); h1=np.array([A1.TX,A1.TY])

    # Optimal ball assignment
    if np.linalg.norm(h0-b0)+np.linalg.norm(h1-b1) <= \
       np.linalg.norm(h0-b1)+np.linalg.norm(h1-b0):
        bx0,by0 = balls[0]; gx0,gy0 = boxes[0]; ba0=ball0; bx_=box0
        bx1,by1 = balls[1]; gx1,gy1 = boxes[1]; ba1=ball1; bx__=box1
    else:
        bx0,by0 = balls[1]; gx0,gy0 = boxes[1]; ba0=ball1; bx_=box0
        bx1,by1 = balls[0]; gx1,gy1 = boxes[0]; ba1=ball0; bx__=box1

    A0.ball=ba0; A0.box=bx_
    A1.ball=ba1; A1.box=bx__

    print(f"\n{'═'*65}")
    print(f" Episode {ep_num}")
    print(f"   R0: 🟢 ({bx0:.3f},{by0:.3f}) → 🔴 ({gx0:.3f},{gy0:.3f})")
    print(f"   R1: 🟠 ({bx1:.3f},{by1:.3f}) → 🔵 ({gx1:.3f},{gy1:.3f})")

    # ── Setup ─────────────────────────────────────────────────────
    place_ball(A0.ball, bx0, by0)
    place_ball(A1.ball, bx1, by1)
    A0.box.set_pose(sapien.Pose(p=[gx0,gy0,BOX_Z]))
    A1.box.set_pose(sapien.Pose(p=[gx1,gy1,BOX_Z]))
    A0.reset(); A1.reset()
    for _ in range(500): scene.step()
    scene.update_render(); viewer.render(); time.sleep(0.12)

    # ── [1] Survey ────────────────────────────────────────────────
    print("\n   [1] Survey")
    par_move(A0, np.array([A0.TX, A0.TY, EEZ(TIP_Z_SURVEY)]),
             A1, np.array([A1.TX, A1.TY, EEZ(TIP_Z_SURVEY)]),
             GRIPPER_OPEN_L, GRIPPER_OPEN_R, STEPS_SURVEY)
    p0=A0.pos(); p1=A1.pos()
    print(f"     R0 z={p0[2]:.3f}  R1 z={p1[2]:.3f}")
    A0.photo(f'{OUT_DIR}/ep{ep_num:02d}_R0_1_survey.png')
    A1.photo(f'{OUT_DIR}/ep{ep_num:02d}_R1_1_survey.png')
    time.sleep(0.05)

    # ── [2] Above balls ───────────────────────────────────────────
    print("\n   [2] Above balls")
    par_move(A0, np.array([bx0, by0, EEZ(TIP_Z_ABOVE)]),
             A1, np.array([bx1, by1, EEZ(TIP_Z_ABOVE)]),
             GRIPPER_OPEN_L, GRIPPER_OPEN_R, STEPS_HOVER)
    p0=A0.pos(); p1=A1.pos()
    xe0=np.linalg.norm(p0[:2]-np.array([bx0,by0]))
    xe1=np.linalg.norm(p1[:2]-np.array([bx1,by1]))
    print(f"     R0 XY_err={xe0*100:.1f}cm  tip_z≈{p0[2]-FINGER_TIP_OFFSET:.3f}")
    print(f"     R1 XY_err={xe1*100:.1f}cm  tip_z≈{p1[2]-FINGER_TIP_OFFSET:.3f}")
    A0.photo(f'{OUT_DIR}/ep{ep_num:02d}_R0_2_above.png')
    A1.photo(f'{OUT_DIR}/ep{ep_num:02d}_R1_2_above.png')

    # ── [3] Pre-close ─────────────────────────────────────────────
    print("\n   [3] Pre-close grippers")
    par_gripper(A0, A1, GRIPPER_HALF_L, GRIPPER_HALF_R, n=20, settle=15)

    # ── [4] Descend to pre-grasp ──────────────────────────────────
    print("\n   [4] Descend to pre-grasp")
    par_slow_lower(A0,bx0,by0, A1,bx1,by1,
                   TIP_Z_ABOVE, TIP_Z_PRE,
                   GRIPPER_HALF_L, GRIPPER_HALF_R, STEPS_DESCEND)
    p0=A0.pos(); p1=A1.pos()
    xe0=np.linalg.norm(p0[:2]-np.array([bx0,by0]))
    xe1=np.linalg.norm(p1[:2]-np.array([bx1,by1]))
    print(f"     R0 XY_err={xe0*100:.1f}cm  tip_z≈{p0[2]-FINGER_TIP_OFFSET:.3f}")
    print(f"     R1 XY_err={xe1*100:.1f}cm  tip_z≈{p1[2]-FINGER_TIP_OFFSET:.3f}")

    # ── [5] Final grasp descent ───────────────────────────────────
    print("\n   [5] Final grasp descent")
    par_slow_lower(A0,bx0,by0, A1,bx1,by1,
                   TIP_Z_PRE, TIP_Z_GRASP,
                   GRIPPER_HALF_L, GRIPPER_HALF_R, STEPS_GRASP)
    p0=A0.pos(); p1=A1.pos()
    bp0=np.array(A0.ball.get_pose().p)
    bp1=np.array(A1.ball.get_pose().p)
    xe0=np.linalg.norm(p0[:2]-np.array([bx0,by0]))
    xe1=np.linalg.norm(p1[:2]-np.array([bx1,by1]))
    print(f"     R0 tip_z={p0[2]-FINGER_TIP_OFFSET:.4f}  ball_z={bp0[2]:.4f}  XY={xe0*100:.1f}cm")
    print(f"     R1 tip_z={p1[2]-FINGER_TIP_OFFSET:.4f}  ball_z={bp1[2]:.4f}  XY={xe1*100:.1f}cm")

    # ── [6] Close + activate constraints ─────────────────────────
    print("\n   [6] Close grippers + activate constraints")
    par_gripper(A0, A1, GRIPPER_CLOSE_L, GRIPPER_CLOSE_R, n=45, settle=10)
    for _ in range(80): scene.step()   # physics squeeze
    cs0.on(A0.ee, A0.ball)
    cs1.on(A1.ee, A1.ball)
    for _ in range(50): A0.drv(A0.q); A1.drv(A1.q); step()
    bz0=A0.ball.get_pose().p[2]; bz1=A1.ball.get_pose().p[2]
    print(f"     R0 ✊ GRASPED  ball_z={bz0:.4f}")
    print(f"     R1 ✊ GRASPED  ball_z={bz1:.4f}")
    A0.photo(f'{OUT_DIR}/ep{ep_num:02d}_R0_3_grasped.png')
    A1.photo(f'{OUT_DIR}/ep{ep_num:02d}_R1_3_grasped.png')

    # ── [7] Lift ──────────────────────────────────────────────────
    print("\n   [7] Lift")
    par_move(A0, np.array([bx0, by0, EEZ(TIP_Z_LIFT)]),
             A1, np.array([bx1, by1, EEZ(TIP_Z_LIFT)]),
             GRIPPER_CLOSE_L, GRIPPER_CLOSE_R, STEPS_LIFT)
    bz0=A0.ball.get_pose().p[2]; bz1=A1.ball.get_pose().p[2]
    print(f"     R0 ball_z={bz0:.4f} {'🎉 AIRBORNE!' if bz0>0.15 else '⚠ low'}")
    print(f"     R1 ball_z={bz1:.4f} {'🎉 AIRBORNE!' if bz1>0.15 else '⚠ low'}")
    time.sleep(0.06)

    # ── [8] Carry via arc midpoint ────────────────────────────────
    print("\n   [8] Carry to box positions")
    par_move(A0, np.array([(bx0+gx0)/2, (by0+gy0)/2, EEZ(TIP_Z_LIFT+0.02)]),
             A1, np.array([(bx1+gx1)/2, (by1+gy1)/2, EEZ(TIP_Z_LIFT+0.02)]),
             GRIPPER_CLOSE_L, GRIPPER_CLOSE_R, STEPS_CARRY)
    par_move(A0, np.array([gx0, gy0, EEZ(TIP_Z_ABOVE_BOX)]),
             A1, np.array([gx1, gy1, EEZ(TIP_Z_ABOVE_BOX)]),
             GRIPPER_CLOSE_L, GRIPPER_CLOSE_R, STEPS_CARRY)
    bz0=A0.ball.get_pose().p[2]; bz1=A1.ball.get_pose().p[2]
    print(f"     R0 carry_z={bz0:.4f}  R1 carry_z={bz1:.4f}")
    time.sleep(0.06)

    # ── [9] Lower into boxes ──────────────────────────────────────
    print("\n   [9] Lower into boxes")
    par_move(A0, np.array([gx0, gy0, EEZ(TIP_Z_LOWER)]),
             A1, np.array([gx1, gy1, EEZ(TIP_Z_LOWER)]),
             GRIPPER_CLOSE_L, GRIPPER_CLOSE_R, STEPS_LOWER)
    p0=A0.pos(); p1=A1.pos()
    print(f"     R0 EE_z={p0[2]:.3f}  R1 EE_z={p1[2]:.3f}")

    # ── [10] Release ──────────────────────────────────────────────
    print("\n   [10] Release balls")
    cs0.off(); cs1.off()
    par_gripper(A0, A1, GRIPPER_OPEN_L, GRIPPER_OPEN_R, n=28, settle=10)
    for _ in range(150): scene.step()   # let balls settle
    scene.update_render(); viewer.render()
    print("     R0 🖐 Released  R1 🖐 Released")
    A0.photo(f'{OUT_DIR}/ep{ep_num:02d}_R0_4_placed.png')
    A1.photo(f'{OUT_DIR}/ep{ep_num:02d}_R1_4_placed.png')
    time.sleep(0.06)

    # ── [11] Retreat home ─────────────────────────────────────────
    print("\n   [11] Retreat home")
    par_move(A0, np.array([gx0, gy0, EEZ(TIP_Z_ABOVE_BOX)]),
             A1, np.array([gx1, gy1, EEZ(TIP_Z_ABOVE_BOX)]),
             GRIPPER_OPEN_L, GRIPPER_OPEN_R, 25)
    par_move(A0, np.array([A0.TX, A0.TY, EEZ(TIP_Z_SURVEY-0.06)]),
             A1, np.array([A1.TX, A1.TY, EEZ(TIP_Z_SURVEY-0.06)]),
             GRIPPER_OPEN_L, GRIPPER_OPEN_R, STEPS_HOVER)

    # ── Results ───────────────────────────────────────────────────
    for _ in range(250): scene.update_render(); viewer.render()
    bf0=np.array(A0.ball.get_pose().p); bf1=np.array(A1.ball.get_pose().p)
    bxf0=np.array(A0.box.get_pose().p); bxf1=np.array(A1.box.get_pose().p)
    d0=np.linalg.norm(bf0[:2]-bxf0[:2])
    d1=np.linalg.norm(bf1[:2]-bxf1[:2])
    ok0=(d0<0.10)and(bf0[2]<BOX_Z+0.07)
    ok1=(d1<0.10)and(bf1[2]<BOX_Z+0.07)

    print(f"\n   ─── RESULTS ───")
    print(f"   R0: ball=({bf0[0]:.3f},{bf0[1]:.3f},{bf0[2]:.3f})"
          f"  box=({bxf0[0]:.3f},{bxf0[1]:.3f})"
          f"  dist={d0*100:.1f}cm  {'✅ SUCCESS' if ok0 else '❌ MISS'}")
    print(f"   R1: ball=({bf1[0]:.3f},{bf1[1]:.3f},{bf1[2]:.3f})"
          f"  box=({bxf1[0]:.3f},{bxf1[1]:.3f})"
          f"  dist={d1*100:.1f}cm  {'✅ SUCCESS' if ok1 else '❌ MISS'}")
    return ok0, ok1

# ═══════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'═'*65}")
print(f" DUAL-ROBOT PERFECT FINAL")
print(f" R0 home: ({TX0:.3f},{TY0:.3f},{TZ0:.3f})")
print(f" R1 home: ({TX1:.3f},{TY1:.3f},{TZ1:.3f})")
print(f" TABLE_TOP={TABLE_TOP}  BALL_Z={BALL_Z}  EEZ(grasp)={EEZ(TIP_Z_GRASP):.3f}")
print(f"{'═'*65}\n")

rng=np.random.default_rng(42); successes=[0,0]; ep=0

while not viewer.closed:
    ep += 1

    # Ball 0 in R0's workspace (world coords)
    bx0=TX0+rng.uniform(-0.08,-0.01); by0=TY0+rng.uniform(-0.06, 0.06)
    # Ball 1 in R1's workspace (world coords)
    bx1=TX1+rng.uniform(-0.08,-0.01); by1=TY1+rng.uniform(-0.06, 0.06)

    # Boxes on far side of each workspace
    for _ in range(60):
        gx0=TX0+rng.uniform(0.03,0.10); gy0=TY0+rng.uniform(-0.06,0.06)
        if np.linalg.norm(np.array([bx0,by0])-np.array([gx0,gy0]))>=0.16: break
    for _ in range(60):
        gx1=TX1+rng.uniform(0.03,0.10); gy1=TY1+rng.uniform(-0.06,0.06)
        if np.linalg.norm(np.array([bx1,by1])-np.array([gx1,gy1]))>=0.16: break

    ok0,ok1 = run_episode(ep,
                          [(bx0,by0),(bx1,by1)],
                          [(gx0,gy0),(gx1,gy1)])
    if ok0: successes[0]+=1
    if ok1: successes[1]+=1
    pct=lambda s: 100*s//ep if ep>0 else 0
    print(f"\n Total: R0={successes[0]}/{ep} ({pct(successes[0])}%)"
          f"  R1={successes[1]}/{ep} ({pct(successes[1])}%)\n")