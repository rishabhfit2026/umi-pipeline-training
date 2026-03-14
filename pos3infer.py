"""
Inference with marker-aware MLP model
conda activate maniskill2 && python step8_poseonly2_infer.py
"""
import sapien, numpy as np, torch, torch.nn as nn
from scipy.spatial.transform import Rotation

URDF   = '/home/rishabh/Downloads/myarm_m750_fixed.urdf'
OUT    = '/home/rishabh/Downloads/umi-pipeline-training/outputs/poseonly_model2'
DEVICE = 'cuda:0'
PRED_STEPS=8; N_DOFS=7; IN_DIMS=13
EPISODE_LEN=360

obs_mean = np.load(f'{OUT}/obs_mean.npy')
obs_std  = np.load(f'{OUT}/obs_std.npy')
act_mean = np.load(f'{OUT}/act_mean.npy')
act_std  = np.load(f'{OUT}/act_std.npy')

class PoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IN_DIMS,512),nn.ReLU(),
            nn.Linear(512,512),nn.ReLU(),
            nn.Linear(512,512),nn.ReLU(),
            nn.Linear(512,512),nn.ReLU(),
            nn.Linear(512,256),nn.ReLU(),
            nn.Linear(256,PRED_STEPS*N_DOFS))
    def forward(self,x): return self.net(x)

model=PoseNet().to(DEVICE).eval()
model.load_state_dict(torch.load(f'{OUT}/model_final.pt'))
print("Model loaded!")

scene=sapien.Scene()
scene.set_timestep(1/120)
scene.set_ambient_light([0.6,0.6,0.6])
scene.add_directional_light([0,1,-1],[1,1,1])
scene.add_ground(altitude=0)

loader=scene.create_urdf_loader(); loader.fix_root_link=True
robot=loader.load(URDF); robot.set_pose(sapien.Pose(p=[0,0,0]))
joints=robot.get_active_joints(); N=len(joints)
links={l.name:l for l in robot.get_links()}; ee=links['gripper']
ee_idx=ee.get_index(); pm=robot.create_pinocchio_model()
for jt in joints: jt.set_drive_property(stiffness=8000,damping=800)

q0=np.zeros(N); q0[1]=-0.3; q0[2]=0.5
robot.set_qpos(q0)
for i,jt in enumerate(joints): jt.set_drive_target(float(q0[i]))
for _ in range(300): scene.step()
real_ee=np.array(ee.get_pose().p); TX,TY=real_ee[0],real_ee[1]
print(f"EE home: ({TX:.3f},{TY:.3f})")

def make_box(half,color,pos,static=True,name=""):
    mt=sapien.render.RenderMaterial(); mt.base_color=color
    b=scene.create_actor_builder()
    b.add_box_visual(half_size=half,material=mt)
    b.add_box_collision(half_size=half)
    a=b.build_static(name=name) if static else b.build(name=name)
    a.set_pose(sapien.Pose(p=pos)); return a

make_box([0.30,0.28,0.025],[0.52,0.33,0.15,1.0],[TX,TY,0.025],True,"table")
make_box([0.20,0.18,0.002],[0.96,0.96,0.94,1.0],[TX,TY,0.052],True,"mat")
mr=sapien.render.RenderMaterial(); mr.base_color=[0.95,0.1,0.1,1.0]
bm=scene.create_actor_builder(); bm.add_sphere_visual(radius=0.018,material=mr)
sim_marker=bm.build(name="marker")
mg=sapien.render.RenderMaterial(); mg.base_color=[0.05,0.80,0.15,1.0]
gb=scene.create_actor_builder()
gb.add_box_visual(half_size=[0.055,0.055,0.025],material=mg)
gb.add_box_collision(half_size=[0.055,0.055,0.025])
sim_box=gb.build_static(name="box")

def solve_ik(xyz,grip_val):
    r=Rotation.from_euler('xyz',[np.pi,0,0]); qv=r.as_quat()
    pose=sapien.Pose(p=list(xyz),q=[qv[3],qv[0],qv[1],qv[2]])
    mask=np.ones(N,dtype=np.int32); mask[6:]=0
    qr,ok,_=pm.compute_inverse_kinematics(ee_idx,pose,
        initial_qpos=robot.get_qpos().astype(np.float64),
        active_qmask=mask,max_iterations=300)
    q=np.array(qr)
    if N>=7: q[6]=grip_val
    if N>=8: q[7]=grip_val
    return q,ok

viewer=scene.create_viewer()
viewer.set_camera_xyz(0.8,-0.8,1.2)
viewer.set_camera_rpy(0,-0.55,0.65)

rng=np.random.default_rng(99); episode=0
grasped=False; step=0; queue=[]; placed=False
mx=my=bx=by=0.0

def reset_episode():
    global grasped,step,queue,placed,mx,my,bx,by,episode
    episode+=1
    mx=TX+rng.uniform(-0.05,0.05); my=TY+rng.uniform(-0.10,0.10)
    bx=TX+rng.uniform(-0.05,0.05); by=TY+rng.uniform(0.05,0.15)
    sim_marker.set_pose(sapien.Pose(p=[mx,my,0.076]))
    sim_box.set_pose(sapien.Pose(p=[bx,by,0.075]))
    robot.set_qpos(q0)
    for i,jt in enumerate(joints): jt.set_drive_target(float(q0[i]))
    for _ in range(60): scene.step()
    grasped=False; step=0; queue=[]; placed=False
    print(f"\n=== Episode {episode} | marker=({mx:.2f},{my:.2f}) box=({bx:.2f},{by:.2f}) ===")

reset_episode()
print("Running inference!")

while not viewer.closed:
    if step>=EPISODE_LEN:
        reset_episode(); continue

    ep_p=np.array(ee.get_pose().p,dtype=np.float32)
    ep_q=np.array(ee.get_pose().q,dtype=np.float32)
    rot_aa=Rotation.from_quat([ep_q[1],ep_q[2],ep_q[3],ep_q[0]]).as_euler('xyz').astype(np.float32)
    grip_w=np.array([robot.get_qpos()[6] if N>=7 else 0.0],dtype=np.float32)

    # Input = ee state + marker pos + box pos
    mpos=np.array(sim_marker.get_pose().p,dtype=np.float32)
    bpos=np.array(sim_box.get_pose().p,dtype=np.float32)
    obs=np.concatenate([ep_p,rot_aa,grip_w,mpos,bpos[:3]])

    if not queue:
        obs_n=(obs-obs_mean)/obs_std
        with torch.no_grad():
            inp=torch.tensor(obs_n,dtype=torch.float32).unsqueeze(0).to(DEVICE)
            pred=model(inp).cpu().numpy()[0]
        acts=(pred.reshape(PRED_STEPS,N_DOFS)*act_std)+act_mean
        queue=list(acts)
        if step%40==0:
            print(f"  step={step:3d}/{EPISODE_LEN} "
                  f"ee=({ep_p[0]:.3f},{ep_p[1]:.3f},{ep_p[2]:.3f}) "
                  f"grip={grip_w[0]:.4f} grasped={grasped}")

    act=queue.pop(0)
    tpos=act[:3]; grip=float(act[6])
    new_q,_=solve_ik(tpos,grip)
    for i,jt in enumerate(joints): jt.set_drive_target(float(new_q[i]))

    # Grasp/place
    dist=np.linalg.norm(ep_p-mpos)
    if not grasped and grip<0.005 and dist<0.10:
        grasped=True; print(f"  ✅ GRASPED! step={step}")
    if grasped:
        sim_marker.set_pose(sapien.Pose(p=ep_p.tolist()))
        mpos=ep_p.copy()
    if grasped and grip>0.025 and not placed:
        grasped=False; placed=True
        if np.linalg.norm(ep_p[:2]-bpos[:2])<0.12:
            print(f"  🎉 PLACED in box! step={step}")
        else:
            print(f"  📍 Released at ({ep_p[0]:.3f},{ep_p[1]:.3f}) step={step}")

    for _ in range(4): scene.step()
    scene.update_render(); viewer.render()
    step+=1