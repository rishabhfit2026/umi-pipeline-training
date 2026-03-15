"""
MyArm M750 + Unitree H1 (with dexterous hands) — Ball Handoff
==============================================================
SETUP (one-time):
  1. Copy h1_with_hand.urdf → ~/Downloads/h1_description/h1_with_hand.urdf
  2. Get meshes:
     git clone --filter=blob:none --no-checkout --depth=1 \
         https://github.com/unitreerobotics/unitree_ros.git /tmp/ur
     cd /tmp/ur && git sparse-checkout set robots/h1_description/meshes
     git checkout
     mkdir -p ~/Downloads/h1_description/meshes
     cp robots/h1_description/meshes/* ~/Downloads/h1_description/meshes/

H1 JOINT INDICES (from h1_with_hand.urdf, 45 active joints):
  [0-4]   left  leg  (hip_yaw, hip_roll, hip_pitch, knee, ankle)
  [5-9]   right leg
  [10]    torso
  [11-15] left arm   (shoulder_pitch/roll/yaw, elbow, hand)
  [16-27] left  dexterous fingers (12 joints)
  [28]    right_shoulder_pitch_joint   ← WE USE THESE
  [29]    right_shoulder_roll_joint
  [30]    right_shoulder_yaw_joint
  [31]    right_elbow_joint
  [32]    right_hand_joint
  [33-44] right dexterous fingers (12 joints)

conda activate maniskill2
python sapien_handoff_h1.py
"""

import math, time, numpy as np, torch, torch.nn as nn
import torchvision.transforms as T, torchvision.models as tvm
import sapien, PIL.Image
from scipy.spatial.transform import Rotation
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# ═══════════════════════════════════════════════════════════════════
#  PATHS
# ═══════════════════════════════════════════════════════════════════
MYARM_URDF = '/home/rishabh/Downloads/myarm_m750_fixed.urdf'
H1_URDF    = '/home/rishabh/Downloads/h1_description/h1_with_hand.urdf'
CKPT_DIR   = '/home/rishabh/Downloads/umi-pipeline-training/checkpoints_umi_vision'
OUT_DIR    = '/home/rishabh/Downloads/umi-pipeline-training'
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

# ═══════════════════════════════════════════════════════════════════
#  H1 EXACT JOINT INDICES (verified from URDF)
# ═══════════════════════════════════════════════════════════════════
H1_RSP = 28  # right_shoulder_pitch_joint
H1_RSR = 29  # right_shoulder_roll_joint
H1_RSY = 30  # right_shoulder_yaw_joint
H1_REB = 31  # right_elbow_joint
H1_RHD = 32  # right_hand_joint (wrist)
# Right dexterous fingers
H1_R_THUMB_YAW=33; H1_R_THUMB_PIT=34; H1_R_THUMB_INT=35; H1_R_THUMB_DIS=36
H1_R_INDEX_P=37;   H1_R_INDEX_I=38
H1_R_MID_P=39;     H1_R_MID_I=40
H1_R_RING_P=41;    H1_R_RING_I=42
H1_R_PINK_P=43;    H1_R_PINK_I=44
H1_N = 45  # total active joints

# ═══════════════════════════════════════════════════════════════════
#  H1 POSE BUILDER
# ═══════════════════════════════════════════════════════════════════
def h1_q(rsp=0, rsr=0, rsy=0, reb=0, rhd=0, fingers_open=True):
    q = np.zeros(H1_N)
    q[H1_RSP]=rsp; q[H1_RSR]=rsr; q[H1_RSY]=rsy
    q[H1_REB]=reb; q[H1_RHD]=rhd
    if fingers_open:
        q[H1_R_THUMB_YAW]=0.40
        # all other finger joints stay 0 (fully extended)
    else:
        # Curl fingers to cradle ball
        q[H1_R_THUMB_YAW]=-0.30; q[H1_R_THUMB_PIT]=0.60
        q[H1_R_THUMB_INT]=0.40;  q[H1_R_THUMB_DIS]=0.30
        q[H1_R_INDEX_P]=0.80;    q[H1_R_INDEX_I]=0.60
        q[H1_R_MID_P]=0.80;      q[H1_R_MID_I]=0.60
        q[H1_R_RING_P]=0.75;     q[H1_R_RING_I]=0.55
        q[H1_R_PINK_P]=0.70;     q[H1_R_PINK_I]=0.50
    return q

# H1 geometry: leg chain z-sum=-0.9742m → pelvis at z=1.034 to stand on floor
H1_PELVIS_Z = 1.034
H1_BASE_POS = np.array([0.62, 0.00, H1_PELVIS_Z])
# Shoulder world = pelvis + torso(0,0,0) + shoulder_pitch(0.0055,-0.1553,0.4300)
H1_SHOULDER = H1_BASE_POS + np.array([0.0055, -0.1553, 0.4300])

HANDOFF_XYZ = np.array([0.44, 0.00, 0.40])
H1_BOX_POS  = np.array([0.62, -0.30, 0.072])

# Arm poses — all angles derived from shoulder geometry
H1_Q_REST  = h1_q(rsp= 0.0, rsr=-0.2, rsy= 0.0, reb= 0.0, fingers_open=True)
H1_Q_REACH = h1_q(rsp=-1.10, rsr= 0.15, rsy=-0.20, reb= 1.20, rhd=0.10, fingers_open=True)
H1_Q_HOLD  = h1_q(rsp=-1.10, rsr= 0.15, rsy=-0.20, reb= 1.20, rhd=0.10, fingers_open=False)
H1_Q_CARRY = h1_q(rsp=-0.60, rsr=-0.10, rsy=-0.15, reb= 0.90, rhd=0.05, fingers_open=False)
H1_Q_PLACE = h1_q(rsp=-0.40, rsr=-0.90, rsy= 0.10, reb= 0.70, rhd=0.00, fingers_open=False)
H1_Q_LOWER = h1_q(rsp=-0.20, rsr=-1.00, rsy= 0.10, reb= 0.40, rhd=0.00, fingers_open=False)
H1_Q_OPEN  = h1_q(rsp=-0.20, rsr=-1.00, rsy= 0.10, reb= 0.40, rhd=0.00, fingers_open=True)

# ═══════════════════════════════════════════════════════════════════
#  MYARM GEOMETRY
# ═══════════════════════════════════════════════════════════════════
TABLE_TOP=0.052; BALL_R=0.026; BALL_Z=TABLE_TOP+BALL_R
BOX_H=0.020;     BOX_Z=TABLE_TOP+BOX_H
FINGER_TIP_OFFSET=0.100
BALL_GRIP_OFFSET=np.array([0.0,0.0,-0.100])  # hardcoded: ball inside gripper fingers
H1_BALL_OFFSET=np.array([0.0,-0.06,0.0])     # ball in H1 palm (~6cm from hand link)

def EEZ(t): return float(t)+FINGER_TIP_OFFSET
TIP_Z_SURVEY=0.36; TIP_Z_ABOVE=0.18; TIP_Z_PRE=0.060
TIP_Z_GRASP=0.063; TIP_Z_LIFT=0.28; TIP_Z_LOWER=0.085
STEPS_SURVEY=55; STEPS_HOVER=45; STEPS_DESCEND=35
STEPS_GRASP=45;  STEPS_LIFT=35;  STEPS_CARRY=40
STEPS_LOWER=30;  STEPS_GRIP=35;  SIM_PER_STEP=3
BALL_MASS=0.50;  BALL_FRIC=2.00; BALL_REST=0.02
BALL_LDAMP=15.0; BALL_ADAMP=15.0

# ═══════════════════════════════════════════════════════════════════
#  VISION MODEL
# ═══════════════════════════════════════════════════════════════════
OBS_H=2; ACT_H=16; ACT_D=7; ST_D=7; IMG_F=512; IMG_SZ=96; ND=100

class Norm:
    @classmethod
    def load(cls,p):
        n=cls(); d=torch.load(p,map_location='cpu',weights_only=False)
        n.min=d['min']; n.max=d['max']; n.scale=d['scale']; return n

class VisEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(*list(tvm.resnet18(weights=None).children())[:-1])
    def forward(self,x): return self.encoder(x).squeeze(-1).squeeze(-1)

class SinEmb(nn.Module):
    def __init__(self,d): super().__init__(); self.d=d
    def forward(self,t):
        h=self.d//2; e=math.log(10000)/(h-1)
        e=torch.exp(torch.arange(h,device=t.device)*-e)
        return torch.cat([(t.float()[:,None]*e[None,:]).sin(),
                          (t.float()[:,None]*e[None,:]).cos()],dim=-1)

class RB(nn.Module):
    def __init__(self,d,cd):
        super().__init__()
        self.n=nn.Sequential(nn.Linear(d,d),nn.Mish(),nn.Linear(d,d))
        self.c=nn.Linear(cd,d*2); self.ln=nn.LayerNorm(d)
    def forward(self,x,c):
        s,b=self.c(c).chunk(2,dim=-1); return x+self.n(self.ln(x)*(s+1)+b)

class VDN(nn.Module):
    def __init__(self,h=512,dep=8):
        super().__init__()
        fa=ACT_D*ACT_H; cd=512; fi=ST_D*OBS_H+IMG_F*OBS_H
        self.ve=VisEnc()
        self.of=nn.Sequential(nn.Linear(fi,512),nn.Mish(),nn.Linear(512,512),nn.Mish(),nn.Linear(512,256))
        self.te=nn.Sequential(SinEmb(128),nn.Linear(128,256),nn.Mish(),nn.Linear(256,256))
        self.cp=nn.Sequential(nn.Linear(512,cd),nn.Mish(),nn.Linear(cd,cd))
        self.ip=nn.Linear(fa,h)
        self.bl=nn.ModuleList([RB(h,cd) for _ in range(dep)])
        self.op=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,fa))
    def forward(self,n,ts,sf,imgs):
        B=n.shape[0]
        ig=torch.cat([self.ve(imgs[:,i]) for i in range(OBS_H)],dim=-1)
        c=self.cp(torch.cat([self.of(torch.cat([sf,ig],dim=-1)),self.te(ts)],dim=-1))
        x=self.ip(n.reshape(B,-1))
        for b in self.bl: x=b(x,c)
        return self.op(x).reshape(B,ACT_H,ACT_D)

# ═══════════════════════════════════════════════════════════════════
#  BUILD SCENE
# ═══════════════════════════════════════════════════════════════════
print("="*62)
print(" MyArm M750  +  Unitree H1 Dexterous Hands — Handoff")
print("="*62)

scene = sapien.Scene()
scene.set_timestep(1/500)
scene.set_ambient_light([0.50,0.50,0.52])
scene.add_directional_light([ 0.05, 0.08,-1.00],[1.00,1.00,1.00])
scene.add_directional_light([-0.40,-0.20,-0.50],[0.35,0.36,0.40])
scene.add_point_light([0.30,-0.30,1.90],[3.0,3.0,3.0])
scene.add_point_light([0.30, 0.30,1.90],[3.0,3.0,3.0])
scene.add_point_light([0.65, 0.00,1.80],[2.5,2.5,2.5])
scene.add_ground(altitude=0)

def vis_box(half,rgba,pos,rough=0.80,metal=0.0,spec=0.04):
    mt=sapien.render.RenderMaterial()
    mt.base_color=rgba; mt.roughness=rough; mt.metallic=metal; mt.specular=spec
    b=scene.create_actor_builder(); b.add_box_visual(half_size=half,material=mt)
    a=b.build_static(); a.set_pose(sapien.Pose(p=pos)); return a

def col_box(half,rgba,pos,name="",rough=0.80):
    mt=sapien.render.RenderMaterial(); mt.base_color=rgba; mt.roughness=rough
    b=scene.create_actor_builder()
    b.add_box_visual(half_size=half,material=mt)
    b.add_box_collision(half_size=half)
    a=b.build_static(name=name); a.set_pose(sapien.Pose(p=pos)); return a

vis_box([5.0,5.0,0.004],[0.84,0.84,0.85,1.0],[0.40,0.0,0.004],rough=0.88)
vis_box([0.016,5.0,2.60],[0.91,0.91,0.92,1.0],[1.35,0.0,1.30],rough=0.86)
col_box([0.40,0.45,0.025],[0.95,0.95,0.95,1.0],[0.25,0.0,0.025],name="table",rough=0.45)
col_box([0.34,0.39,0.003],[0.98,0.98,0.98,1.0],[0.25,0.0,TABLE_TOP-0.001],name="mat",rough=0.25)

# Handoff marker (small yellow sphere)
mhm=sapien.render.RenderMaterial(); mhm.base_color=[1.0,0.85,0.0,0.65]
hm=scene.create_actor_builder(); hm.add_sphere_visual(radius=0.014,material=mhm)
hm.build_static().set_pose(sapien.Pose(p=HANDOFF_XYZ.tolist()))

# ═══════════════════════════════════════════════════════════════════
#  LOAD MYARM
# ═══════════════════════════════════════════════════════════════════
print(" Loading MyArm M750...")
loader_arm=scene.create_urdf_loader(); loader_arm.fix_root_link=True
robot=loader_arm.load(MYARM_URDF); robot.set_pose(sapien.Pose(p=[0,0,0]))
joints=robot.get_active_joints(); N=len(joints)
links={l.name:l for l in robot.get_links()}
ee=links['gripper']; ee_idx=ee.get_index()
pm=robot.create_pinocchio_model()
GL=6; GR=7 if N>=8 else None
for i,jt in enumerate(joints):
    if i<6: jt.set_drive_property(stiffness=80000,damping=8000)
    else:   jt.set_drive_property(stiffness=20000,damping=2000)

def q_home_arm():
    q=np.zeros(N); q[1]=-0.30; q[2]=0.50; q[GL]=0.030
    if GR: q[GR]=0.060
    return q
Q_HOME=q_home_arm()
robot.set_qpos(Q_HOME.copy())
for i,jt in enumerate(joints): jt.set_drive_target(float(Q_HOME[i]))
for _ in range(600): scene.step()
home_ee=np.array(ee.get_entity_pose().p); TX,TY,TZ=home_ee
print(f" MyArm EE home: ({TX:.4f},{TY:.4f},{TZ:.4f})")

# ═══════════════════════════════════════════════════════════════════
#  LOAD H1 HUMANOID
# ═══════════════════════════════════════════════════════════════════
print("\n Loading Unitree H1 with dexterous hands...")
loader_h1=scene.create_urdf_loader(); loader_h1.fix_root_link=True
h1=loader_h1.load(H1_URDF)
h1.set_pose(sapien.Pose(p=H1_BASE_POS.tolist()))
h1_joints=h1.get_active_joints(); h1_links={l.name:l for l in h1.get_links()}
N_H1_ACT=len(h1_joints)
print(f" H1 loaded: {N_H1_ACT} active joints")

if N_H1_ACT!=H1_N:
    print(f" ⚠ WARNING: expected {H1_N} joints, got {N_H1_ACT}")

# Verify right arm indices
print(" Verifying joint indices:")
for idx,exp in [(H1_RSP,'right_shoulder_pitch'),(H1_RSR,'right_shoulder_roll'),
                (H1_RSY,'right_shoulder_yaw'),(H1_REB,'right_elbow'),(H1_RHD,'right_hand')]:
    if idx < N_H1_ACT:
        nm=h1_joints[idx].get_name()
        ok=exp in nm
        print(f"   [{idx:2d}] {nm}  {'✓' if ok else '✗ MISMATCH!'}")

# PD drives
for i,jt in enumerate(h1_joints):
    if i<=10:   jt.set_drive_property(stiffness=800,  damping=80)   # legs+torso
    elif i<=27: jt.set_drive_property(stiffness=300,  damping=30)   # left arm+hand
    else:       jt.set_drive_property(stiffness=400,  damping=40)   # right arm+hand

h1.set_qpos(H1_Q_REST.copy())
for i,jt in enumerate(h1_joints): jt.set_drive_target(float(H1_Q_REST[i]))
for _ in range(400): scene.step()

# Get H1 right hand link for ball attachment
H1_HAND = (h1_links.get('right_hand_link') or
            h1_links.get('R_hand_base_link') or
            h1_links.get('right_elbow_link'))
print(f" H1 hand link: '{H1_HAND.get_name() if H1_HAND else 'NOT FOUND'}'")
if H1_HAND:
    hp=np.array(H1_HAND.get_entity_pose().p)
    print(f" H1 hand at rest: ({hp[0]:.3f},{hp[1]:.3f},{hp[2]:.3f})")

# ═══════════════════════════════════════════════════════════════════
#  SCENE OBJECTS
# ═══════════════════════════════════════════════════════════════════
def make_ball():
    mt=sapien.render.RenderMaterial()
    mt.base_color=[0.05,0.92,0.10,1.0]; mt.roughness=0.20
    bb=scene.create_actor_builder()
    bb.add_sphere_visual(radius=BALL_R,material=mt)
    pm_b=scene.create_physical_material(BALL_FRIC,BALL_FRIC,BALL_REST)
    bb.add_sphere_collision(radius=BALL_R,material=pm_b)
    b=bb.build(name="ball")
    try:
        rb=b.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        if rb: rb.set_mass(BALL_MASS); rb.set_linear_damping(BALL_LDAMP); rb.set_angular_damping(BALL_ADAMP)
    except: pass
    return b

ball=make_ball()
ball_rb=None
try: ball_rb=ball.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
except: pass

mt_r=sapien.render.RenderMaterial(); mt_r.base_color=[0.90,0.08,0.05,1.0]
bxr=scene.create_actor_builder(); bxr.add_box_visual(half_size=[0.055,0.055,BOX_H],material=mt_r)
bxr.add_box_collision(half_size=[0.055,0.055,BOX_H]); robot_box=bxr.build_static(name="robot_box")

mt_b=sapien.render.RenderMaterial(); mt_b.base_color=[0.05,0.20,0.90,1.0]
bxb=scene.create_actor_builder(); bxb.add_box_visual(half_size=[0.065,0.065,BOX_H],material=mt_b)
bxb.add_box_collision(half_size=[0.065,0.065,BOX_H]); h1_box=bxb.build_static(name="h1_box")
h1_box.set_pose(sapien.Pose(p=H1_BOX_POS.tolist()))

mw=sapien.render.RenderMaterial(); mw.base_color=[1,1,1,0.7]
ew=scene.create_actor_builder(); ew.add_sphere_visual(radius=0.008,material=mw)
ee_dot=ew.build_static()

cam_e=sapien.Entity(); cam_c=sapien.render.RenderCameraComponent(224,224)
cam_c.set_fovy(np.deg2rad(58)); cam_e.add_component(cam_c)
cr=Rotation.from_euler('xyz',[np.deg2rad(130),0,0]); cq=cr.as_quat()
cam_e.set_pose(sapien.Pose(p=[TX-0.18,TY,TZ+0.10],
    q=[float(cq[3]),float(cq[0]),float(cq[1]),float(cq[2])])); scene.add_entity(cam_e)

viewer=scene.create_viewer()
viewer.set_camera_xyz(-0.30,-1.00,1.40)
viewer.set_camera_rpy(0.0,-0.28,0.64)
print(" Scene ready.")

# ═══════════════════════════════════════════════════════════════════
#  ORIENTATION PROBE
# ═══════════════════════════════════════════════════════════════════
eq=ee.get_entity_pose().q; ee_rot=Rotation.from_quat([eq[1],eq[2],eq[3],eq[0]])
def make_gq(ax):
    tgt=np.array([0.,0.,-1.]); aw=ee_rot.apply(ax)
    cr2=np.cross(aw,tgt); cn=np.linalg.norm(cr2); dt=np.dot(aw,tgt)
    if cn<1e-6: c=Rotation.identity() if dt>0 else Rotation.from_euler('x',np.pi)
    else: c=Rotation.from_rotvec(cr2/cn*np.arctan2(cn,dt))
    q=(c*ee_rot).as_quat(); return (float(q[3]),float(q[0]),float(q[1]),float(q[2]))

print("\n Probing orientation...")
cal=np.array([TX,TY,0.25]); bq,bc=None,1e18
for ax in [np.array([0,0,1]),np.array([0,0,-1]),np.array([0,1,0]),
           np.array([0,-1,0]),np.array([1,0,0]),np.array([-1,0,0])]:
    robot.set_qpos(Q_HOME.copy())
    for i,jt in enumerate(joints): jt.set_drive_target(float(Q_HOME[i]))
    for _ in range(200): scene.step()
    try:
        wxyz=make_gq(ax); pose=sapien.Pose(p=cal.tolist(),q=list(wxyz))
        mask=np.ones(N,dtype=np.int32); mask[6:]=0
        qr,ok,_=pm.compute_inverse_kinematics(ee_idx,pose,
            initial_qpos=Q_HOME.astype(np.float64),active_qmask=mask,max_iterations=1000)
        if not ok: continue
        qs=np.array(qr); robot.set_qpos(qs)
        for _ in range(30): scene.step()
        pe=np.linalg.norm(np.array(ee.get_entity_pose().p)-cal)
        w=np.array([1,1,1,1,5,3],dtype=float)
        c=float(np.sum(w*(qs[:6]-Q_HOME[:6])**2))+pe*50.
        print(f"   {ax}  err={pe*100:.1f}cm  cost={c:.2f}")
        if c<bc: bc,bq=c,wxyz
    except: pass
robot.set_qpos(Q_HOME.copy())
for i,jt in enumerate(joints): jt.set_drive_target(float(Q_HOME[i]))
for _ in range(400): scene.step()
if bq is None: q=ee_rot.as_quat(); bq=(float(q[3]),float(q[0]),float(q[1]),float(q[2]))
GRASP_QUAT=bq; print(f" Best cost={bc:.2f}")

def ik(xyz,seed=None):
    if seed is None: seed=robot.get_qpos().copy()
    pose=sapien.Pose(p=[float(v) for v in xyz],q=list(GRASP_QUAT))
    mask=np.ones(N,dtype=np.int32); mask[6:]=0
    qr,ok,_=pm.compute_inverse_kinematics(ee_idx,pose,
        initial_qpos=np.array(seed,dtype=np.float64),active_qmask=mask,max_iterations=1400)
    return np.array(qr),ok

# ═══════════════════════════════════════════════════════════════════
#  CONSTRAINT GRASP
# ═══════════════════════════════════════════════════════════════════
class CS:
    def __init__(self,name):
        self.name=name; self.active=False
        self.offset=np.zeros(3); self.link=None; self.ball=None; self.rb=None
    def on(self,link,ball_actor,fixed=None):
        self.link=link; self.ball=ball_actor; self.active=True
        ep=np.array(link.get_entity_pose().p)
        eq2=link.get_entity_pose().q; er=Rotation.from_quat([eq2[1],eq2[2],eq2[3],eq2[0]])
        self.offset = fixed.copy() if fixed is not None else er.inv().apply(np.array(ball_actor.get_pose().p)-ep)
        try: self.rb=ball_actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        except: self.rb=None
        if self.rb: self.rb.set_linear_velocity([0,0,0]); self.rb.set_angular_velocity([0,0,0])
        bp=ep+er.apply(self.offset); ball_actor.set_pose(sapien.Pose(p=bp.tolist()))
        print(f"     [{self.name}] 🔗 ON → ({bp[0]:.3f},{bp[1]:.3f},{bp[2]:.3f})")
    def off(self):
        if self.active:
            self.active=False
            if self.rb: self.rb.set_linear_velocity([0,0,0]); self.rb.set_angular_velocity([0,0,0])
    def sync(self):
        if not self.active or not self.ball or not self.link: return
        ep=np.array(self.link.get_entity_pose().p)
        eq2=self.link.get_entity_pose().q; er=Rotation.from_quat([eq2[1],eq2[2],eq2[3],eq2[0]])
        self.ball.set_pose(sapien.Pose(p=(ep+er.apply(self.offset)).tolist()))
        if self.rb: self.rb.set_linear_velocity([0,0,0]); self.rb.set_angular_velocity([0,0,0])

cs_arm=CS("MyArm"); cs_h1=CS("H1")

# ═══════════════════════════════════════════════════════════════════
#  RENDER / PRIMITIVES
# ═══════════════════════════════════════════════════════════════════
def drv_arm(q):
    for i,jt in enumerate(joints): jt.set_drive_target(float(q[i]))
def drv_h1(q):
    for i,jt in enumerate(h1_joints): jt.set_drive_target(float(q[i]))
def step():
    for _ in range(SIM_PER_STEP): scene.step()
    cs_arm.sync(); cs_h1.sync()
    ee_dot.set_pose(sapien.Pose(p=list(ee.get_entity_pose().p)))
    scene.update_render(); viewer.render()
def settle(n=20):
    for _ in range(n): step()
def reset_ball(x,y):
    cs_arm.off(); cs_h1.off()
    ball.set_pose(sapien.Pose(p=[x,y,BALL_Z]))
    if ball_rb: ball_rb.set_linear_velocity([0,0,0]); ball_rb.set_angular_velocity([0,0,0])
def save_img(path):
    scene.update_render(); cam_c.take_picture()
    rgba=cam_c.get_picture('Color')
    PIL.Image.fromarray((np.clip(rgba[:,:,:3],0,1)*255).astype(np.uint8)).save(path)
    print(f"   📷 {path}")

def arm_move(xyz,gl,gr,n,settle_n=20):
    q_cur=robot.get_qpos().copy()
    qs,ok=ik(xyz,q_cur)
    if not ok: qs,ok=ik(xyz,Q_HOME)
    if not ok: print(f"  ⚠ IK fail {np.round(xyz,3)}"); return
    qs[GL]=float(gl)
    if GR: qs[GR]=float(gr)
    s=robot.get_qpos().copy()
    for i in range(n):
        t=(i+1)/n; sm=t*t*(3.-2.*t); drv_arm(s+sm*(qs-s)); step()
        if viewer.closed: return
    for _ in range(settle_n): drv_arm(qs); step()
    robot.set_qpos(qs.copy())

def arm_slow(bx,by,z_start,z_end,gl,gr,n):
    q_cur=robot.get_qpos().copy()
    for z_tip in np.linspace(z_start,z_end,n):
        ep=np.array(ee.get_entity_pose().p); d=np.array([bx,by])-ep[:2]
        tx=float(np.clip(ep[0]+d[0]*0.80,TX-0.20,TX+0.20))
        ty=float(np.clip(ep[1]+d[1]*0.80,TY-0.20,TY+0.20))
        qs,ok=ik(np.array([tx,ty,EEZ(z_tip)]),q_cur)
        if not ok: qs,ok=ik(np.array([bx,by,EEZ(z_tip)]),q_cur)
        if not ok: continue
        qs[GL]=float(gl)
        if GR: qs[GR]=float(gr)
        q_cur=qs.copy(); drv_arm(qs)
        for _ in range(2): scene.step()
        cs_arm.sync(); cs_h1.sync()
        ee_dot.set_pose(sapien.Pose(p=list(ee.get_entity_pose().p)))
        scene.update_render(); viewer.render()
        if viewer.closed: break

def arm_grip(gl,gr,n=STEPS_GRIP,settle_n=30):
    qt=robot.get_qpos().copy(); qt[GL]=float(gl)
    if GR: qt[GR]=float(gr)
    s=robot.get_qpos().copy()
    for i in range(n): t=(i+1)/n; drv_arm(s+t*(qt-s)); step()
    for _ in range(settle_n): drv_arm(qt); step()
    robot.set_qpos(qt.copy())

def h1_move(q_target,n=55,settle_n=25,label=""):
    q_cur=h1.get_qpos().copy()
    for i in range(n):
        t=(i+1)/n; sm=t*t*(3.-2.*t); drv_h1(q_cur+sm*(q_target-q_cur)); step()
        if viewer.closed: return
    for _ in range(settle_n): drv_h1(q_target); step()
    if label and H1_HAND:
        hp=np.array(H1_HAND.get_entity_pose().p)
        print(f"     H1 {label} — hand: ({hp[0]:.3f},{hp[1]:.3f},{hp[2]:.3f})")

# ═══════════════════════════════════════════════════════════════════
#  LOAD VISION MODEL
# ═══════════════════════════════════════════════════════════════════
print("\n Loading vision model...")
on=Norm.load(f'{CKPT_DIR}/obs_normalizer.pt'); an=Norm.load(f'{CKPT_DIR}/act_normalizer.pt')
net=VDN().to(DEVICE)
ck=torch.load(f'{CKPT_DIR}/best_model.pt',map_location=DEVICE,weights_only=False)
net.load_state_dict(ck['model_state']); net.eval()
ns=DDPMScheduler(num_train_timesteps=ND,beta_schedule='squaredcos_cap_v2',
                 clip_sample=True,prediction_type='epsilon')
print(f" Loaded epoch={ck['epoch']}  loss={ck['loss']:.5f}")

# ═══════════════════════════════════════════════════════════════════
#  EPISODE RUNNER
# ═══════════════════════════════════════════════════════════════════
def run_episode(ep_num,bx_ball,by_ball):
    print(f"\n{'═'*65}")
    print(f" Episode {ep_num}  ball=({bx_ball:.3f},{by_ball:.3f})")
    print(f"   Handoff: {HANDOFF_XYZ}  |  H1 box: {H1_BOX_POS[:2]}")

    reset_ball(bx_ball,by_ball)
    robot_box.set_pose(sapien.Pose(p=[TX+0.06,TY,BOX_Z]))
    robot.set_qpos(Q_HOME.copy()); drv_arm(Q_HOME)
    h1.set_qpos(H1_Q_REST.copy()); drv_h1(H1_Q_REST)
    for _ in range(500): scene.step()
    scene.update_render(); viewer.render(); time.sleep(0.12)

    # [1] Survey
    print("\n   [1] MyArm survey")
    arm_move(np.array([TX,TY,EEZ(TIP_Z_SURVEY)]),0.030,0.060,STEPS_SURVEY)
    save_img(f'{OUT_DIR}/ho_ep{ep_num:02d}_1_survey.png')

    # [2] Above ball
    print("\n   [2] Above ball")
    arm_move(np.array([bx_ball,by_ball,EEZ(0.18)]),0.030,0.060,STEPS_HOVER)
    ep_=np.array(ee.get_entity_pose().p)
    print(f"     XY_err={np.linalg.norm(ep_[:2]-np.array([bx_ball,by_ball]))*100:.1f}cm")

    # [3] Pre-close
    print("\n   [3] Pre-close gripper")
    arm_grip(0.018,0.036,n=20,settle_n=10)

    # [4-5] Descend
    print("\n   [4-5] Descend and grasp")
    arm_slow(bx_ball,by_ball,TIP_Z_ABOVE,TIP_Z_PRE,0.018,0.036,STEPS_DESCEND)
    arm_slow(bx_ball,by_ball,TIP_Z_PRE,TIP_Z_GRASP,0.018,0.036,STEPS_GRASP)

    # [6] Close → ball snaps inside gripper
    print("\n   [6] Close gripper → ball inside fingers")
    arm_grip(0.000,0.000,n=45,settle_n=10)
    for _ in range(80): scene.step()
    cs_arm.on(ee,ball,fixed=BALL_GRIP_OFFSET)
    settle(50)
    bz=ball.get_pose().p[2]
    print(f"     ✊ GRASPED  ball_z={bz:.4f}  {'🎉 AIRBORNE!' if bz>0.15 else '⚠'}")
    save_img(f'{OUT_DIR}/ho_ep{ep_num:02d}_2_grasped.png')

    # [7] Lift
    print("\n   [7] Lift")
    arm_move(np.array([bx_ball,by_ball,EEZ(TIP_Z_LIFT)]),0.000,0.000,STEPS_LIFT)
    print(f"     ball_z={ball.get_pose().p[2]:.4f}")

    # [8] Carry to handoff
    print("\n   [8] Carry to HANDOFF")
    arm_move(np.array([HANDOFF_XYZ[0]-0.08,HANDOFF_XYZ[1],EEZ(TIP_Z_LIFT)]),0.000,0.000,STEPS_CARRY)
    arm_move(HANDOFF_XYZ.copy(),0.000,0.000,STEPS_CARRY)
    print(f"     At handoff  ball_z={ball.get_pose().p[2]:.4f}")
    save_img(f'{OUT_DIR}/ho_ep{ep_num:02d}_3_at_handoff.png')
    time.sleep(0.06)

    # [9] H1 reaches with fingers open
    print("\n   [9] H1 extends arm (fingers open) to receive")
    h1_move(H1_Q_REACH,n=65,settle_n=30,label="REACH")
    if H1_HAND:
        hp=np.array(H1_HAND.get_entity_pose().p)
        print(f"     Hand↔ball: {np.linalg.norm(hp-ball.get_pose().p)*100:.1f}cm")
    time.sleep(0.05)

    # [10] HANDOFF
    print("\n   [10] *** HANDOFF: ball transfers MyArm → H1 ***")
    cs_arm.off()
    arm_grip(0.030,0.060,n=18,settle_n=5)
    if H1_HAND: cs_h1.on(H1_HAND,ball)
    settle(15)
    print("     H1 curling dexterous fingers around ball...")
    h1_move(H1_Q_HOLD,n=35,settle_n=15)
    print(f"     H1 holding  ball_z={ball.get_pose().p[2]:.4f}")
    save_img(f'{OUT_DIR}/ho_ep{ep_num:02d}_4_handoff.png')

    # [11] MyArm retreats
    print("\n   [11] MyArm retreats")
    arm_move(np.array([HANDOFF_XYZ[0]-0.15,HANDOFF_XYZ[1],EEZ(TIP_Z_LIFT)]),0.030,0.060,30)
    arm_move(np.array([TX,TY,EEZ(TIP_Z_SURVEY-0.05)]),0.030,0.060,STEPS_HOVER)

    # [12] H1 carries to blue box
    print("\n   [12] H1 carries ball to blue box")
    h1_move(H1_Q_CARRY,n=55,settle_n=20,label="CARRY")
    h1_move(H1_Q_PLACE,n=60,settle_n=25,label="PLACE")

    # [13] H1 lowers into box
    print("\n   [13] H1 lowers into box")
    h1_move(H1_Q_LOWER,n=40,settle_n=15,label="LOWER")

    # [14] Dexterous fingers open → ball drops
    print("\n   [14] H1 dexterous fingers open → ball drops")
    cs_h1.off()
    h1_move(H1_Q_OPEN,n=30,settle_n=10)
    for _ in range(200): scene.step()
    scene.update_render(); viewer.render()
    print("     🖐 Released by H1 dexterous hand")
    save_img(f'{OUT_DIR}/ho_ep{ep_num:02d}_5_placed.png')
    time.sleep(0.06)

    # [15] Both home
    print("\n   [15] Both return to rest")
    h1_move(H1_Q_REST,n=50,settle_n=20)
    arm_move(np.array([TX,TY,EEZ(TIP_Z_SURVEY-0.05)]),0.030,0.060,STEPS_HOVER)

    # Result
    for _ in range(250): scene.update_render(); viewer.render()
    bf=np.array(ball.get_pose().p); bxf=np.array(h1_box.get_pose().p)
    dist=np.linalg.norm(bf[:2]-bxf[:2]); ok=(dist<0.15)and(bf[2]<0.30)
    print(f"\n   ─── RESULT ───")
    print(f"   Ball:  ({bf[0]:.3f},{bf[1]:.3f},{bf[2]:.3f})")
    print(f"   Box:   ({bxf[0]:.3f},{bxf[1]:.3f})")
    print(f"   dist={dist*100:.1f}cm  {'✅ SUCCESS' if ok else '❌ MISS'}")
    return ok

# ═══════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'═'*65}")
print(f" READY — MyArm + Unitree H1 Dexterous Handoff")
print(f" MyArm home:   ({TX:.3f},{TY:.3f},{TZ:.3f})")
print(f" H1 pelvis z:  {H1_PELVIS_Z:.3f}m  (feet on floor)")
print(f" H1 shoulder:  {np.round(H1_SHOULDER,3)}")
print(f" Handoff pt:   {HANDOFF_XYZ}")
print(f" H1 blue box:  {H1_BOX_POS}")
print(f"{'═'*65}\n")

rng=np.random.default_rng(42); successes=0; ep=0
while not viewer.closed:
    ep+=1
    bx=TX+rng.uniform(-0.08,-0.01); by=TY+rng.uniform(-0.05,0.05)
    ok=run_episode(ep,bx,by)
    if ok: successes+=1
    print(f"\n Total: {successes}/{ep} ({100*successes//ep if ep>0 else 0}%)\n")