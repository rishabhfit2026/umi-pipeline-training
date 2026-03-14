"""
Dual-Robot Pick & Place — FIXED VERSION
=========================================
ROOT CAUSE FIX: Pinocchio IK expects coordinates in ROBOT LOCAL frame.
  Robot 0 base at y=-0.38 → must subtract (0, -0.38, 0) from world targets
  Robot 1 base at y=+0.38 → must subtract (0, +0.38, 0) from world targets

  Evidence from logs:
    Ball world (0.223, -0.378), robot0 base y=-0.38
    IK was given world coords → EE went to world (0.223, -0.758)
    XY_err = |−0.758 − (−0.378)| = 0.38m ≈ 37-40cm (matches every log)

  Fix: local_xyz = world_xyz - robot_base_xyz before every IK call.

Two robots work simultaneously:
  • Same approach style as v4 (survey→above→pre-close→slow_lower→grasp)
  • Constraint grasp from v7/v8 (ball follows gripper visually)
  • Each robot picks the nearest ball, places in nearest box
  • Ball + box assigned so total travel distance is minimized

conda activate maniskill2
python sapien_dual_robot_fixed.py
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
#  MODEL CONSTANTS
# ═══════════════════════════════════════════════════════════════════
OBS_HORIZON    = 2
ACTION_HORIZON = 16
ACTION_DIM     = 7
STATE_DIM      = 7
IMG_FEAT_DIM   = 512
IMG_SIZE       = 96
NUM_DIFF_STEPS = 100

# ═══════════════════════════════════════════════════════════════════
#  ROBOT PLACEMENT  (Y-axis separation so workspaces don't overlap)
# ═══════════════════════════════════════════════════════════════════
ROBOT0_BASE = np.array([0.0, -0.38, 0.0])
ROBOT1_BASE = np.array([0.0, +0.38, 0.0])

# ═══════════════════════════════════════════════════════════════════
#  GRIPPER & GEOMETRY CONSTANTS
# ═══════════════════════════════════════════════════════════════════
GRIPPER_OPEN_L  = 0.030
GRIPPER_OPEN_R  = 0.060
GRIPPER_CLOSE_L = 0.000
GRIPPER_CLOSE_R = 0.000
GRIPPER_HALF_L  = 0.018
GRIPPER_HALF_R  = 0.036

TABLE_TOP  = 0.052
BALL_R     = 0.026
BALL_Z     = TABLE_TOP + BALL_R   # 0.078
BOX_H      = 0.020
BOX_Z      = TABLE_TOP + BOX_H   # 0.072

FINGER_TIP_OFFSET = 0.100

TIP_Z_SURVEY    = 0.36
TIP_Z_ABOVE     = 0.18
TIP_Z_PRE       = 0.060
TIP_Z_GRASP     = 0.063
TIP_Z_LIFT      = 0.26
TIP_Z_ABOVE_BOX = 0.18
TIP_Z_LOWER     = 0.075

def EEZ(tip_z): return float(tip_z) + FINGER_TIP_OFFSET

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

# ═══════════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════════
img_transform = T.Compose([
    T.ToPILImage(), T.Resize((IMG_SIZE,IMG_SIZE)), T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

class Normalizer:
    @classmethod
    def load(cls,p):
        n=cls(); d=torch.load(p,map_location='cpu',weights_only=False)
        n.min=d['min']; n.max=d['max']; n.scale=d['scale']; return n
    def normalize(self,x):
        return 2.*(x-self.min.to(x.device))/self.scale.to(x.device)-1.
    def denormalize(self,x):
        return (x+1.)/2.*self.scale.to(x.device)+self.min.to(x.device)

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
        e=t.float()[:,None]*e[None,:]
        return torch.cat([e.sin(),e.cos()],dim=-1)

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
print("="*60)
print(" SAPIEN Dual-Robot Pick & Place — FIXED")
print("="*60)

scene = sapien.Scene()
scene.set_timestep(1/480)
scene.set_ambient_light([0.60,0.60,0.60])
scene.add_directional_light([0.3,0.8,-1.0],[1.0,0.96,0.88])
scene.add_directional_light([-0.3,-0.8,-0.6],[0.30,0.32,0.40])
scene.add_point_light([0.30, 0.0,1.10],[1.2,1.1,1.0])
scene.add_point_light([0.30,-0.5,0.80],[0.9,0.9,1.1])
scene.add_point_light([0.30,+0.5,0.80],[0.9,0.9,1.1])
scene.add_ground(altitude=0)

# ── Load two robots ───────────────────────────────────────────────
def load_robot(base_pos):
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot  = loader.load(URDF)
    robot.set_pose(sapien.Pose(p=base_pos.tolist()))
    joints = robot.get_active_joints()
    N      = len(joints)
    links  = {l.name:l for l in robot.get_links()}
    ee     = links['gripper']
    pm     = robot.create_pinocchio_model()
    for i,jt in enumerate(joints):
        if i<6: jt.set_drive_property(stiffness=50000,damping=5000)
        else:   jt.set_drive_property(stiffness=8000, damping=800)
    return robot, joints, N, ee, pm

print(" Loading Robot 0...")
robot0,joints0,N0,ee0,pm0 = load_robot(ROBOT0_BASE)
print(" Loading Robot 1...")
robot1,joints1,N1,ee1,pm1 = load_robot(ROBOT1_BASE)
N = N0
GRIPPER_L_IDX = 6
GRIPPER_R_IDX = 7 if N>=8 else None
print(f" Both robots: {N} joints")

def make_q_home():
    q = np.zeros(N)
    q[1]=-0.30; q[2]=0.50
    q[GRIPPER_L_IDX]=GRIPPER_OPEN_L
    if GRIPPER_R_IDX: q[GRIPPER_R_IDX]=GRIPPER_OPEN_R
    return q

q_home = make_q_home()

def reset_home(robot, joints):
    robot.set_qpos(q_home.copy())
    for i,jt in enumerate(joints): jt.set_drive_target(float(q_home[i]))

reset_home(robot0,joints0); reset_home(robot1,joints1)
for _ in range(600): scene.step()

home0 = np.array(ee0.get_entity_pose().p)
home1 = np.array(ee1.get_entity_pose().p)
TX0,TY0,TZ0 = home0
TX1,TY1,TZ1 = home1
print(f"\n R0 EE world home: ({TX0:.4f},{TY0:.4f},{TZ0:.4f})")
print(f" R1 EE world home: ({TX1:.4f},{TY1:.4f},{TZ1:.4f})")

# ── Orientation probe (using robot 0, calibration point in its LOCAL workspace) ──
eq = ee0.get_entity_pose().q
ee_rot = Rotation.from_quat([eq[1],eq[2],eq[3],eq[0]])

def make_grasp_quat(ax):
    tgt=np.array([0.,0.,-1.]); aw=ee_rot.apply(ax)
    cr=np.cross(aw,tgt); cn=np.linalg.norm(cr); dt=np.dot(aw,tgt)
    if cn<1e-6: corr=Rotation.identity() if dt>0 else Rotation.from_euler('x',np.pi)
    else:       corr=Rotation.from_rotvec(cr/cn*np.arctan2(cn,dt))
    q=(corr*ee_rot).as_quat()
    return (float(q[3]),float(q[0]),float(q[1]),float(q[2]))

print("\n Probing orientation (home restored each time)...")
# Calibration target in ROBOT 0 LOCAL frame (same as single robot v4)
cal_local = np.array([TX0-ROBOT0_BASE[0], TY0-ROBOT0_BASE[1], 0.22])
best_q,best_c = None, 1e18

for ax in [np.array([0,0,1]),np.array([0,0,-1]),
           np.array([0,1,0]),np.array([0,-1,0]),
           np.array([1,0,0]),np.array([-1,0,0])]:
    reset_home(robot0,joints0); reset_home(robot1,joints1)
    for _ in range(200): scene.step()
    try:
        wxyz=make_grasp_quat(ax)
        pose=sapien.Pose(p=cal_local.tolist(),q=list(wxyz))
        mask=np.ones(N,dtype=np.int32); mask[6:]=0
        qr,ok,_=pm0.compute_inverse_kinematics(
            ee0.get_index(),pose,
            initial_qpos=q_home.astype(np.float64),
            active_qmask=mask,max_iterations=800)
        if not ok: continue
        qs=np.array(qr); robot0.set_qpos(qs)
        for _ in range(30): scene.step()
        ae=np.array(ee0.get_entity_pose().p)
        # ae is world; convert to local to compare with cal_local
        ae_local=ae-ROBOT0_BASE
        pe=np.linalg.norm(ae_local-cal_local)
        w=np.array([1,1,1,1,5,3],dtype=float)
        c=float(np.sum(w*(qs[:6]-q_home[:6])**2))+pe*50.
        print(f"   {ax}  err={pe*100:.1f}cm  cost={c:.2f}")
        if c<best_c: best_c,best_q=c,wxyz
    except: pass

reset_home(robot0,joints0); reset_home(robot1,joints1)
for _ in range(300): scene.step()
if best_q is None:
    q=ee_rot.as_quat(); best_q=(float(q[3]),float(q[0]),float(q[1]),float(q[2]))
GRASP_QUAT=best_q
print(f" Best cost={best_c:.2f}")

# ═══════════════════════════════════════════════════════════════════
#  IK FUNCTION — THE KEY FIX
#  target_world: target position in WORLD frame
#  robot_base:   the robot's base position in world
#  IK solver expects LOCAL frame → convert before calling
# ═══════════════════════════════════════════════════════════════════
def ik_robot(pm, ee_link, target_world, robot_base, q_seed):
    """
    IK in robot LOCAL frame.
    target_world: np.array [x,y,z] in WORLD coordinates.
    robot_base:   np.array [x,y,z] robot base in world.
    Converts to local before calling pinocchio.
    """
    target_local = target_world - robot_base   # ← THE FIX
    pose = sapien.Pose(p=[float(v) for v in target_local], q=list(GRASP_QUAT))
    mask = np.ones(len(q_seed), dtype=np.int32); mask[6:]=0
    qr, ok, _ = pm.compute_inverse_kinematics(
        ee_link.get_index(), pose,
        initial_qpos=np.array(q_seed, dtype=np.float64),
        active_qmask=mask, max_iterations=1200)
    return np.array(qr), ok

# ═══════════════════════════════════════════════════════════════════
#  SCENE OBJECTS
# ═══════════════════════════════════════════════════════════════════
def make_static(half,rgba,pos,name=""):
    mt=sapien.render.RenderMaterial(); mt.base_color=rgba
    b=scene.create_actor_builder()
    b.add_box_visual(half_size=half,material=mt)
    b.add_box_collision(half_size=half)
    a=b.build_static(name=name); a.set_pose(sapien.Pose(p=pos)); return a

cx = (TX0+TX1)/2; cy = (TY0+TY1)/2
make_static([0.36,0.50,0.025],[0.50,0.34,0.16,1.0],[cx,cy,0.025],"table")
make_static([0.28,0.42,0.002],[0.95,0.92,0.84,1.0],[cx,cy,0.052],"mat")

def make_ball(color,name):
    mg=sapien.render.RenderMaterial(); mg.base_color=color
    bb=scene.create_actor_builder()
    bb.add_sphere_visual(radius=BALL_R,material=mg)
    pm_b=scene.create_physical_material(
        static_friction=BALL_FRICTION,dynamic_friction=BALL_FRICTION,
        restitution=BALL_RESTITUTION)
    bb.add_sphere_collision(radius=BALL_R,material=pm_b)
    b=bb.build(name=name)
    try:
        rb=b.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        if rb:
            rb.set_mass(BALL_MASS)
            rb.set_linear_damping(BALL_LINEAR_DAMPING)
            rb.set_angular_damping(BALL_ANGULAR_DAMPING)
    except: pass
    return b

def make_box(color,name):
    mr=sapien.render.RenderMaterial(); mr.base_color=color
    bx=scene.create_actor_builder()
    bx.add_box_visual(half_size=[0.055,0.055,BOX_H],material=mr)
    bx.add_box_collision(half_size=[0.055,0.055,BOX_H])
    return bx.build_static(name=name)

ball0=make_ball([0.05,0.92,0.12,1.0],"ball0")
ball1=make_ball([0.20,0.80,0.05,1.0],"ball1")
box0 =make_box([0.92,0.06,0.06,1.0],"box0")
box1 =make_box([0.70,0.08,0.08,1.0],"box1")

def make_dot(c):
    mw=sapien.render.RenderMaterial(); mw.base_color=c
    ew=scene.create_actor_builder(); ew.add_sphere_visual(radius=0.010,material=mw)
    return ew.build_static()

dot0=make_dot([1,1,1,0.8]); dot1=make_dot([0.8,0.8,1,0.8])

def make_cam(tx,ty,tz):
    ce=sapien.Entity(); cc=sapien.render.RenderCameraComponent(224,224)
    cc.set_fovy(np.deg2rad(58)); ce.add_component(cc)
    cr=Rotation.from_euler('xyz',[np.deg2rad(130),0,0]); cq=cr.as_quat()
    ce.set_pose(sapien.Pose(p=[tx-0.18,ty,tz+0.10],
        q=[float(cq[3]),float(cq[0]),float(cq[1]),float(cq[2])]))
    scene.add_entity(ce); return cc

cam0=make_cam(TX0,TY0,TZ0); cam1=make_cam(TX1,TY1,TZ1)

viewer=scene.create_viewer()
viewer.set_camera_xyz(TX0+0.80,cy,0.90)
viewer.set_camera_rpy(0,-0.30,0.50)

print(f" Scene built. Table centre ({cx:.3f},{cy:.3f})")

# ═══════════════════════════════════════════════════════════════════
#  CONSTRAINT GRASP
# ═══════════════════════════════════════════════════════════════════
class CS:
    def __init__(self,rid):
        self.rid=rid; self.active=False
        self.offset=np.zeros(3); self.ball=None; self.rb=None
    def on(self,ee_link,ball_actor):
        ep=np.array(ee_link.get_entity_pose().p)
        eq=ee_link.get_entity_pose().q
        er=Rotation.from_quat([eq[1],eq[2],eq[3],eq[0]])
        bp=np.array(ball_actor.get_pose().p)
        self.offset=er.inv().apply(bp-ep)
        self.active=True; self.ball=ball_actor
        try: self.rb=ball_actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        except: self.rb=None
        if self.rb: self.rb.set_linear_velocity([0,0,0]); self.rb.set_angular_velocity([0,0,0])
        print(f"     R{self.rid} 🔗 ON  offset={np.round(self.offset,3)}")
    def off(self):
        if self.active:
            self.active=False
            if self.rb: self.rb.set_linear_velocity([0,0,0]); self.rb.set_angular_velocity([0,0,0])
    def sync(self,ee_link):
        if not self.active or self.ball is None: return
        ep=np.array(ee_link.get_entity_pose().p)
        eq=ee_link.get_entity_pose().q
        er=Rotation.from_quat([eq[1],eq[2],eq[3],eq[0]])
        bp=ep+er.apply(self.offset)
        self.ball.set_pose(sapien.Pose(p=bp.tolist()))
        if self.rb: self.rb.set_linear_velocity([0,0,0]); self.rb.set_angular_velocity([0,0,0])

cs0=CS(0); cs1=CS(1)

# ═══════════════════════════════════════════════════════════════════
#  RENDER / DRIVE HELPERS
# ═══════════════════════════════════════════════════════════════════
def drives(joints,q):
    for j,jt in enumerate(joints): jt.set_drive_target(float(q[j]))

def step_render():
    for _ in range(SIM_PER_STEP): scene.step()
    cs0.sync(ee0); cs1.sync(ee1)
    dot0.set_pose(sapien.Pose(p=list(ee0.get_entity_pose().p)))
    dot1.set_pose(sapien.Pose(p=list(ee1.get_entity_pose().p)))
    scene.update_render(); viewer.render()

def save_img(path,cam):
    scene.update_render(); cam.take_picture()
    rgba=cam.get_picture('Color')
    PIL.Image.fromarray((np.clip(rgba[:,:,:3],0,1)*255).astype(np.uint8)).save(path)

def place_ball(actor,x,y):
    actor.set_pose(sapien.Pose(p=[x,y,BALL_Z]))
    try:
        rb=actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        if rb: rb.set_linear_velocity([0,0,0]); rb.set_angular_velocity([0,0,0])
    except: pass

# ═══════════════════════════════════════════════════════════════════
#  ROBOT ARM WRAPPER
# ═══════════════════════════════════════════════════════════════════
class Arm:
    def __init__(self,robot,joints,ee,pm,cs,cam,base,tx,ty,tz,rid):
        self.robot=robot; self.joints=joints; self.ee=ee
        self.pm=pm; self.cs=cs; self.cam=cam
        self.base=base  # world position of robot base
        self.TX=tx; self.TY=ty; self.TZ=tz
        self.q=make_q_home(); self.rid=rid
        self.ball=None; self.box_=None

    def ik(self,world_xyz,q_seed=None):
        if q_seed is None: q_seed=self.q
        return ik_robot(self.pm,self.ee,world_xyz,self.base,q_seed)

    def ee_world(self):
        return np.array(self.ee.get_entity_pose().p)

    def set_drives(self,q):
        drives(self.joints,q)

    def qpos(self):
        return self.robot.get_qpos().copy()

    def reset(self):
        self.robot.set_qpos(make_q_home())
        drives(self.joints,make_q_home())
        self.q=make_q_home(); self.cs.off()

    def snap(self,path):
        save_img(path,self.cam)
        print(f"   [R{self.rid}] 📷 {path}")

arm0=Arm(robot0,joints0,ee0,pm0,cs0,cam0,ROBOT0_BASE,TX0,TY0,TZ0,0)
arm1=Arm(robot1,joints1,ee1,pm1,cs1,cam1,ROBOT1_BASE,TX1,TY1,TZ1,1)

# ═══════════════════════════════════════════════════════════════════
#  PARALLEL MOTION  (both robots move simultaneously each physics step)
# ═══════════════════════════════════════════════════════════════════
def par_move(a0,a1,w0,w1,gl,gr,n):
    """Smooth move both robots to world targets w0,w1."""
    q0,ok0=a0.ik(w0,a0.q)
    if not ok0: q0,ok0=a0.ik(w0,q_home)
    if not ok0: print(f" R0 IK fail {np.round(w0,3)}"); q0=a0.q.copy()

    q1,ok1=a1.ik(w1,a1.q)
    if not ok1: q1,ok1=a1.ik(w1,q_home)
    if not ok1: print(f" R1 IK fail {np.round(w1,3)}"); q1=a1.q.copy()

    q0[GRIPPER_L_IDX]=float(gl); q1[GRIPPER_L_IDX]=float(gl)
    if GRIPPER_R_IDX: q0[GRIPPER_R_IDX]=float(gr); q1[GRIPPER_R_IDX]=float(gr)

    s0=a0.qpos(); s1=a1.qpos()
    for i in range(n):
        t=(i+1)/n; sm=t*t*(3.-2.*t)
        a0.set_drives(s0+sm*(q0-s0))
        a1.set_drives(s1+sm*(q1-s1))
        step_render()
        if viewer.closed: break
    for _ in range(15):
        a0.set_drives(q0); a1.set_drives(q1); step_render()
    a0.q=q0.copy(); a1.q=q1.copy()


def par_slow(a0,a1,bx0,by0,bx1,by1,z_start,z_end,gl,gr,n):
    """
    Slow descent with XY correction for both robots simultaneously.
    All positions are in WORLD frame.
    """
    for z_tip in np.linspace(z_start,z_end,n):
        ee_z=EEZ(z_tip)

        # Robot 0: correct XY toward its ball (world frame)
        ep0=a0.ee_world()
        dxy0=np.array([bx0,by0])-ep0[:2]
        tx0=float(np.clip(ep0[0]+dxy0[0]*0.70,
                          a0.TX-0.18,a0.TX+0.18))
        ty0=float(np.clip(ep0[1]+dxy0[1]*0.70,
                          a0.TY-0.18,a0.TY+0.18))
        tgt0=np.array([tx0,ty0,ee_z])

        # Robot 1: correct XY toward its ball (world frame)
        ep1=a1.ee_world()
        dxy1=np.array([bx1,by1])-ep1[:2]
        tx1=float(np.clip(ep1[0]+dxy1[0]*0.70,
                          a1.TX-0.18,a1.TX+0.18))
        ty1=float(np.clip(ep1[1]+dxy1[1]*0.70,
                          a1.TY-0.18,a1.TY+0.18))
        tgt1=np.array([tx1,ty1,ee_z])

        q0,ok0=a0.ik(tgt0,a0.q)
        if not ok0:
            q0,ok0=a0.ik(np.array([bx0,by0,ee_z]),a0.q)

        q1,ok1=a1.ik(tgt1,a1.q)
        if not ok1:
            q1,ok1=a1.ik(np.array([bx1,by1,ee_z]),a1.q)

        if ok0:
            q0[GRIPPER_L_IDX]=float(gl)
            if GRIPPER_R_IDX: q0[GRIPPER_R_IDX]=float(gr)
            a0.set_drives(q0); a0.q=q0.copy()
        if ok1:
            q1[GRIPPER_L_IDX]=float(gl)
            if GRIPPER_R_IDX: q1[GRIPPER_R_IDX]=float(gr)
            a1.set_drives(q1); a1.q=q1.copy()

        for _ in range(3): scene.step()
        cs0.sync(a0.ee); cs1.sync(a1.ee)
        dot0.set_pose(sapien.Pose(p=list(a0.ee.get_entity_pose().p)))
        dot1.set_pose(sapien.Pose(p=list(a1.ee.get_entity_pose().p)))
        scene.update_render(); viewer.render()
        if viewer.closed: break


def par_gripper(a0,a1,gl,gr,n=STEPS_GRIPPER):
    qt0=a0.q.copy(); qt0[GRIPPER_L_IDX]=float(gl)
    qt1=a1.q.copy(); qt1[GRIPPER_L_IDX]=float(gl)
    if GRIPPER_R_IDX: qt0[GRIPPER_R_IDX]=float(gr); qt1[GRIPPER_R_IDX]=float(gr)
    s0=a0.qpos(); s1=a1.qpos()
    for i in range(n):
        t=(i+1)/n
        a0.set_drives(s0+t*(qt0-s0)); a1.set_drives(s1+t*(qt1-s1)); step_render()
    for _ in range(25): a0.set_drives(qt0); a1.set_drives(qt1); step_render()
    a0.q=qt0.copy(); a1.q=qt1.copy()

# ═══════════════════════════════════════════════════════════════════
#  LOAD MODEL
# ═══════════════════════════════════════════════════════════════════
print("\n Loading vision model...")
obs_norm=Normalizer.load(f'{CKPT_DIR}/obs_normalizer.pt')
act_norm=Normalizer.load(f'{CKPT_DIR}/act_normalizer.pt')
net=VisionDiffusionNet().to(DEVICE)
ck=torch.load(f'{CKPT_DIR}/best_model.pt',map_location=DEVICE,weights_only=False)
net.load_state_dict(ck['model_state']); net.eval()
print(f" Loaded epoch={ck['epoch']} loss={ck['loss']:.5f}")
noise_sched=DDPMScheduler(num_train_timesteps=NUM_DIFF_STEPS,
    beta_schedule='squaredcos_cap_v2',clip_sample=True,prediction_type='epsilon')

# ═══════════════════════════════════════════════════════════════════
#  DUAL EPISODE RUNNER
# ═══════════════════════════════════════════════════════════════════
def run_dual(ep_num,balls_world,boxes_world):
    """
    balls_world = [(bx0,by0),(bx1,by1)] in WORLD frame
    boxes_world = [(gx0,gy0),(gx1,gy1)] in WORLD frame
    Nearest-ball assignment per robot.
    """
    b0=np.array(balls_world[0]); b1=np.array(balls_world[1])
    h0=np.array([arm0.TX,arm0.TY]); h1=np.array([arm1.TX,arm1.TY])

    # Assign ball/box so total robot→ball distance is minimised
    if np.linalg.norm(h0-b0)+np.linalg.norm(h1-b1) <= \
       np.linalg.norm(h0-b1)+np.linalg.norm(h1-b0):
        bx0,by0=balls_world[0]; gx0,gy0=boxes_world[0]; ba0=ball0; bx_=box0
        bx1,by1=balls_world[1]; gx1,gy1=boxes_world[1]; ba1=ball1; bx__=box1
    else:
        bx0,by0=balls_world[1]; gx0,gy0=boxes_world[1]; ba0=ball1; bx_=box0
        bx1,by1=balls_world[0]; gx1,gy1=boxes_world[0]; ba1=ball0; bx__=box1

    arm0.ball=ba0; arm0.box_=bx_
    arm1.ball=ba1; arm1.box_=bx__

    print(f"\n{'═'*65}")
    print(f" Episode {ep_num}")
    print(f"   R0: 🟢({bx0:.3f},{by0:.3f}) → 🔴({gx0:.3f},{gy0:.3f})")
    print(f"   R1: 🟢({bx1:.3f},{by1:.3f}) → 🔴({gx1:.3f},{gy1:.3f})")

    place_ball(arm0.ball,bx0,by0); place_ball(arm1.ball,bx1,by1)
    arm0.box_.set_pose(sapien.Pose(p=[gx0,gy0,BOX_Z]))
    arm1.box_.set_pose(sapien.Pose(p=[gx1,gy1,BOX_Z]))
    arm0.reset(); arm1.reset()
    for _ in range(400): scene.step()
    scene.update_render(); viewer.render(); time.sleep(0.08)

    # ── 1: Survey ────────────────────────────────────────────────
    print("\n   [1] Survey")
    par_move(arm0,arm1,
             np.array([arm0.TX,arm0.TY,EEZ(TIP_Z_SURVEY)]),
             np.array([arm1.TX,arm1.TY,EEZ(TIP_Z_SURVEY)]),
             GRIPPER_OPEN_L,GRIPPER_OPEN_R,STEPS_SURVEY)
    ep0=arm0.ee_world(); ep1=arm1.ee_world()
    # Verify local accuracy
    ep0_l=ep0-ROBOT0_BASE; ep1_l=ep1-ROBOT1_BASE
    print(f"     R0 EE world=({ep0[0]:.3f},{ep0[1]:.3f},{ep0[2]:.3f})"
          f"  local_z={ep0_l[2]:.3f}")
    print(f"     R1 EE world=({ep1[0]:.3f},{ep1[1]:.3f},{ep1[2]:.3f})"
          f"  local_z={ep1_l[2]:.3f}")
    arm0.snap(f'{OUT_DIR}/dual_ep{ep_num:02d}_R0_1_survey.png')
    arm1.snap(f'{OUT_DIR}/dual_ep{ep_num:02d}_R1_1_survey.png')

    # ── 2: Above balls ───────────────────────────────────────────
    print("\n   [2] Above balls")
    par_move(arm0,arm1,
             np.array([bx0,by0,EEZ(TIP_Z_ABOVE)]),
             np.array([bx1,by1,EEZ(TIP_Z_ABOVE)]),
             GRIPPER_OPEN_L,GRIPPER_OPEN_R,STEPS_HOVER)
    ep0=arm0.ee_world(); ep1=arm1.ee_world()
    xe0=np.linalg.norm(ep0[:2]-np.array([bx0,by0]))
    xe1=np.linalg.norm(ep1[:2]-np.array([bx1,by1]))
    print(f"     R0 XY_err={xe0*100:.1f}cm  R1 XY_err={xe1*100:.1f}cm")
    arm0.snap(f'{OUT_DIR}/dual_ep{ep_num:02d}_R0_2_above.png')
    arm1.snap(f'{OUT_DIR}/dual_ep{ep_num:02d}_R1_2_above.png')

    # ── 3: Pre-close ─────────────────────────────────────────────
    print("\n   [3] Pre-close grippers")
    par_gripper(arm0,arm1,GRIPPER_HALF_L,GRIPPER_HALF_R,n=20)

    # ── 4: Descend pre-grasp ─────────────────────────────────────
    print("\n   [4] Descend to pre-grasp")
    par_slow(arm0,arm1,bx0,by0,bx1,by1,
             TIP_Z_ABOVE,TIP_Z_PRE,
             GRIPPER_HALF_L,GRIPPER_HALF_R,STEPS_DESCEND)
    ep0=arm0.ee_world(); ep1=arm1.ee_world()
    xe0=np.linalg.norm(ep0[:2]-np.array([bx0,by0]))
    xe1=np.linalg.norm(ep1[:2]-np.array([bx1,by1]))
    print(f"     R0 XY_err={xe0*100:.1f}cm tip_z≈{ep0[2]-FINGER_TIP_OFFSET:.3f}")
    print(f"     R1 XY_err={xe1*100:.1f}cm tip_z≈{ep1[2]-FINGER_TIP_OFFSET:.3f}")

    # ── 5: Final grasp descent ───────────────────────────────────
    print("\n   [5] Final grasp descent")
    par_slow(arm0,arm1,bx0,by0,bx1,by1,
             TIP_Z_PRE,TIP_Z_GRASP,
             GRIPPER_HALF_L,GRIPPER_HALF_R,STEPS_GRASP)
    ep0=arm0.ee_world(); ep1=arm1.ee_world()
    bp0=np.array(arm0.ball.get_pose().p)
    bp1=np.array(arm1.ball.get_pose().p)
    xe0=np.linalg.norm(ep0[:2]-np.array([bx0,by0]))
    xe1=np.linalg.norm(ep1[:2]-np.array([bx1,by1]))
    print(f"     R0 tip_z={ep0[2]-FINGER_TIP_OFFSET:.4f} ball_z={bp0[2]:.4f} XY={xe0*100:.1f}cm")
    print(f"     R1 tip_z={ep1[2]-FINGER_TIP_OFFSET:.4f} ball_z={bp1[2]:.4f} XY={xe1*100:.1f}cm")

    # ── 6: Close + constraint ────────────────────────────────────
    print("\n   [6] Close grippers + activate constraints")
    par_gripper(arm0,arm1,GRIPPER_CLOSE_L,GRIPPER_CLOSE_R,n=45)
    for _ in range(60): scene.step()
    cs0.on(arm0.ee,arm0.ball)
    cs1.on(arm1.ee,arm1.ball)
    for _ in range(40):
        arm0.set_drives(arm0.q); arm1.set_drives(arm1.q); step_render()
    bz0=arm0.ball.get_pose().p[2]; bz1=arm1.ball.get_pose().p[2]
    print(f"     R0 ✊ GRASPED  ball_z={bz0:.4f}")
    print(f"     R1 ✊ GRASPED  ball_z={bz1:.4f}")
    arm0.snap(f'{OUT_DIR}/dual_ep{ep_num:02d}_R0_3_grasped.png')
    arm1.snap(f'{OUT_DIR}/dual_ep{ep_num:02d}_R1_3_grasped.png')

    # ── 7: Lift ───────────────────────────────────────────────────
    print("\n   [7] Lift")
    par_move(arm0,arm1,
             np.array([bx0,by0,EEZ(TIP_Z_LIFT)]),
             np.array([bx1,by1,EEZ(TIP_Z_LIFT)]),
             GRIPPER_CLOSE_L,GRIPPER_CLOSE_R,STEPS_LIFT)
    bz0=arm0.ball.get_pose().p[2]; bz1=arm1.ball.get_pose().p[2]
    print(f"     R0 ball_z={bz0:.4f} {'🎉 AIRBORNE!' if bz0>0.15 else '⚠'}")
    print(f"     R1 ball_z={bz1:.4f} {'🎉 AIRBORNE!' if bz1>0.15 else '⚠'}")
    time.sleep(0.05)

    # ── 8: Carry ──────────────────────────────────────────────────
    print("\n   [8] Carry to boxes")
    par_move(arm0,arm1,
             np.array([(bx0+gx0)/2,(by0+gy0)/2,EEZ(TIP_Z_LIFT+0.02)]),
             np.array([(bx1+gx1)/2,(by1+gy1)/2,EEZ(TIP_Z_LIFT+0.02)]),
             GRIPPER_CLOSE_L,GRIPPER_CLOSE_R,STEPS_CARRY)
    par_move(arm0,arm1,
             np.array([gx0,gy0,EEZ(TIP_Z_ABOVE_BOX)]),
             np.array([gx1,gy1,EEZ(TIP_Z_ABOVE_BOX)]),
             GRIPPER_CLOSE_L,GRIPPER_CLOSE_R,STEPS_CARRY)
    time.sleep(0.05)

    # ── 9: Lower ──────────────────────────────────────────────────
    print("\n   [9] Lower into boxes")
    par_move(arm0,arm1,
             np.array([gx0,gy0,EEZ(TIP_Z_LOWER)]),
             np.array([gx1,gy1,EEZ(TIP_Z_LOWER)]),
             GRIPPER_CLOSE_L,GRIPPER_CLOSE_R,STEPS_LOWER)

    # ── 10: Release ───────────────────────────────────────────────
    print("\n   [10] Release")
    cs0.off(); cs1.off()
    par_gripper(arm0,arm1,GRIPPER_OPEN_L,GRIPPER_OPEN_R,n=25)
    for _ in range(120): scene.step()
    scene.update_render(); viewer.render()
    print("     R0 🖐 Released  R1 🖐 Released")
    arm0.snap(f'{OUT_DIR}/dual_ep{ep_num:02d}_R0_4_placed.png')
    arm1.snap(f'{OUT_DIR}/dual_ep{ep_num:02d}_R1_4_placed.png')

    # ── 11: Retreat ───────────────────────────────────────────────
    print("\n   [11] Retreat")
    par_move(arm0,arm1,
             np.array([gx0,gy0,EEZ(TIP_Z_ABOVE_BOX)]),
             np.array([gx1,gy1,EEZ(TIP_Z_ABOVE_BOX)]),
             GRIPPER_OPEN_L,GRIPPER_OPEN_R,22)
    par_move(arm0,arm1,
             np.array([arm0.TX,arm0.TY,EEZ(TIP_Z_SURVEY-0.05)]),
             np.array([arm1.TX,arm1.TY,EEZ(TIP_Z_SURVEY-0.05)]),
             GRIPPER_OPEN_L,GRIPPER_OPEN_R,STEPS_HOVER)

    # ── Results ───────────────────────────────────────────────────
    for _ in range(200): scene.update_render(); viewer.render()
    bf0=np.array(arm0.ball.get_pose().p); bf1=np.array(arm1.ball.get_pose().p)
    bx0f=np.array(arm0.box_.get_pose().p); bx1f=np.array(arm1.box_.get_pose().p)
    d0=np.linalg.norm(bf0[:2]-bx0f[:2])
    d1=np.linalg.norm(bf1[:2]-bx1f[:2])
    ok0=(d0<0.10)and(bf0[2]<BOX_Z+0.06)
    ok1=(d1<0.10)and(bf1[2]<BOX_Z+0.06)
    print(f"\n   ─── RESULTS ───")
    print(f"   R0: {np.round(bf0,3)} dist={d0*100:.1f}cm  {'✅' if ok0 else '❌'}")
    print(f"   R1: {np.round(bf1,3)} dist={d1*100:.1f}cm  {'✅' if ok1 else '❌'}")
    return ok0,ok1

# ═══════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'═'*65}")
print(f" DUAL ROBOT READY (IK frame bug fixed)")
print(f" R0 world home: ({TX0:.3f},{TY0:.3f}) base: {ROBOT0_BASE}")
print(f" R1 world home: ({TX1:.3f},{TY1:.3f}) base: {ROBOT1_BASE}")
print(f" EE grasp z = {EEZ(TIP_Z_GRASP):.4f}")
print(f"{'═'*65}\n")

rng=np.random.default_rng(42); successes=[0,0]; ep=0

while not viewer.closed:
    ep+=1
    # Ball 0 in R0's workspace (world coords)
    bx0=TX0+rng.uniform(-0.08,-0.01)
    by0=TY0+rng.uniform(-0.06, 0.06)
    # Ball 1 in R1's workspace (world coords)
    bx1=TX1+rng.uniform(-0.08,-0.01)
    by1=TY1+rng.uniform(-0.06, 0.06)
    # Box 0 far from ball 0
    for _ in range(50):
        gx0=TX0+rng.uniform(0.03,0.10)
        gy0=TY0+rng.uniform(-0.06,0.06)
        if np.linalg.norm(np.array([bx0,by0])-np.array([gx0,gy0]))>=0.18: break
    # Box 1 far from ball 1
    for _ in range(50):
        gx1=TX1+rng.uniform(0.03,0.10)
        gy1=TY1+rng.uniform(-0.06,0.06)
        if np.linalg.norm(np.array([bx1,by1])-np.array([gx1,gy1]))>=0.18: break

    ok0,ok1=run_dual(ep,[(bx0,by0),(bx1,by1)],[(gx0,gy0),(gx1,gy1)])
    if ok0: successes[0]+=1
    if ok1: successes[1]+=1
    print(f"\n Total: R0={successes[0]}/{ep} R1={successes[1]}/{ep}"
          f"  ({100*successes[0]//ep}%  {100*successes[1]//ep}%)\n")