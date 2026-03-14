"""
Diffusion Policy Inference — Episode 74 focus
Model predicts EEF positions → IK converts to joint angles → robot moves
Red ball picked up and placed in green box using DIFFUSION POLICY

conda activate maniskill2
python diffusion_infer_ep74.py
"""

import sapien, numpy as np, torch, zarr
from scipy.spatial.transform import Rotation
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import sys
sys.path.insert(0, '/home/rishabh/Downloads/umi-pipeline-training')

SAVE_DIR = '/home/rishabh/Downloads/umi-pipeline-training/checkpoints_pose_only'
from poseonly import PoseDiffusionNet, Normalizer, OBS_HORIZON, ACTION_HORIZON, ACTION_DIM, OBS_DIM

URDF     = '/home/rishabh/Downloads/myarm_m750_fixed.urdf'
ZARR     = '/home/rishabh/Downloads/umi-pipeline-training/outputs/sim_replay_buffer3.zarr'
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
START_EP = 73   # 0-indexed = episode 74
EXEC_H   = 6    # execute 6 actions then re-predict

# ── Load model ────────────────────────────────────────────────────
ckpt  = torch.load(f'{SAVE_DIR}/best_model.pt', map_location=DEVICE)
model = PoseDiffusionNet(ACTION_DIM, ACTION_HORIZON, OBS_DIM, OBS_HORIZON).to(DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()
obs_norm = Normalizer.load(f'{SAVE_DIR}/obs_normalizer.pt')
act_norm = Normalizer.load(f'{SAVE_DIR}/act_normalizer.pt')
sched    = DDIMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2',
                         clip_sample=True, prediction_type='epsilon')
sched.set_timesteps(16)
print(f"✅ Model loaded — epoch {ckpt['epoch']}, loss={ckpt['loss']:.5f}")

def predict_actions(obs_history):
    obs_parts = [obs_norm.normalize(torch.tensor(s, dtype=torch.float32))
                 for s in obs_history]
    obs_flat = torch.cat(obs_parts).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        noisy = torch.randn(1, ACTION_HORIZON, ACTION_DIM, device=DEVICE)
        for t in sched.timesteps:
            ts    = torch.tensor([t], device=DEVICE).long()
            noisy = sched.step(model(noisy, ts, obs_flat), t, noisy).prev_sample
        return act_norm.denormalize(noisy[0]).cpu().numpy()

# ── Scene ─────────────────────────────────────────────────────────
scene = sapien.Scene()
scene.set_timestep(1/120)
scene.set_ambient_light([0.5,0.5,0.5])
scene.add_directional_light([0,1,-1],[1.0,0.95,0.85])
scene.add_directional_light([1,0,-1],[0.5,0.5,0.6])
scene.add_ground(altitude=0)

loader = scene.create_urdf_loader(); loader.fix_root_link=True
robot  = loader.load(URDF); robot.set_pose(sapien.Pose(p=[0,0,0]))
joints = robot.get_active_joints(); N=len(joints)
links  = {l.name:l for l in robot.get_links()}
ee     = links['gripper']
ee_idx = ee.get_index(); pm=robot.create_pinocchio_model()
for jt in joints: jt.set_drive_property(stiffness=20000, damping=2000)

q0=np.zeros(N); q0[1]=-0.3; q0[2]=0.5
robot.set_qpos(q0)
for i,jt in enumerate(joints): jt.set_drive_target(float(q0[i]))
for _ in range(200): scene.step()
real_ee=np.array(ee.get_entity_pose().p); TX,TY=real_ee[0],real_ee[1]
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

# Red ball marker
mr=sapien.render.RenderMaterial(); mr.base_color=[0.95,0.10,0.10,1.0]
bm=scene.create_actor_builder()
bm.add_sphere_visual(radius=0.022,material=mr)
bm.add_sphere_collision(radius=0.022)
sim_marker=bm.build(name="marker")

# Green box target
mg=sapien.render.RenderMaterial(); mg.base_color=[0.05,0.80,0.15,1.0]
gb=scene.create_actor_builder()
gb.add_box_visual(half_size=[0.060,0.060,0.010],material=mg)
gb.add_box_collision(half_size=[0.060,0.060,0.010])
sim_box=gb.build_static(name="box")

def solve_ik(target_pos, grip_val):
    r   =Rotation.from_euler('xyz',[np.pi,0,0]); qv=r.as_quat()
    pose=sapien.Pose(p=list(target_pos),q=[qv[3],qv[0],qv[1],qv[2]])
    mask=np.ones(N,dtype=np.int32); mask[6:]=0
    qr,ok,_=pm.compute_inverse_kinematics(ee_idx,pose,
        initial_qpos=robot.get_qpos().astype(np.float64),
        active_qmask=mask,max_iterations=500)
    q=np.array(qr)
    if N>=7: q[6]=grip_val
    if N>=8: q[7]=grip_val
    return q

def get_eef_state():
    p   =np.array(ee.get_entity_pose().p)
    grip=float(np.clip(robot.get_qpos()[6],0,0.035)) if N>6 else 0.0
    return np.array([p[0],p[1],p[2],0,0,0,grip],dtype=np.float32)

# ── Viewer — better angle to see robot ───────────────────────────
viewer=scene.create_viewer()
viewer.set_camera_xyz(TX+0.5, TY-0.5, 0.9)
viewer.set_camera_rpy(0, -0.4, 0.5)

# ── Episodes ──────────────────────────────────────────────────────
zf   =zarr.open(ZARR,'r'); ends=zf['meta']['episode_ends'][:]
rng  =np.random.default_rng(42)
ep_markers=[]; ep_boxes=[]
for _ in range(len(ends)):
    mx=TX+rng.uniform(-0.05,0.05); my=TY+rng.uniform(-0.10,0.10)
    bx=TX+rng.uniform(-0.05,0.05); by=TY+rng.uniform(0.05,0.15)
    while abs(mx-bx)<0.04 and abs(my-by)<0.04:
        bx=TX+rng.uniform(-0.05,0.05); by=TY+rng.uniform(0.05,0.15)
    ep_markers.append([mx,my,0.076])
    ep_boxes.append([bx,by,0.075])

print(f"\n{'='*55}")
print(f" DIFFUSION POLICY INFERENCE")
print(f" Starting at Episode 74")
print(f"{'='*55}\n")

ep_idx=START_EP; MAX_STEPS=500

while not viewer.closed:
    mx,my,mz=ep_markers[ep_idx]
    bx,by,bz=ep_boxes[ep_idx]

    sim_marker.set_pose(sapien.Pose(p=[mx,my,mz]))
    sim_box.set_pose(sapien.Pose(p=[bx,by,bz]))
    robot.set_qpos(q0)
    for i,jt in enumerate(joints): jt.set_drive_target(float(q0[i]))
    for _ in range(80): scene.step()
    scene.update_render(); viewer.render()

    print(f"{'='*55}")
    print(f"Episode {ep_idx+1} | 🔴 marker=({mx:.3f},{my:.3f}) | 🟢 box=({bx:.3f},{by:.3f})")
    print(f"{'='*55}")

    obs_buf=[]; act_q=[]; grasped=False
    step=0; success=False; grasp_step=None; grip_val=0.035

    while step<MAX_STEPS and not viewer.closed:
        state=get_eef_state()
        obs_buf.append(state)
        if len(obs_buf)>OBS_HORIZON: obs_buf.pop(0)

        # Re-predict every EXEC_H steps
        if len(obs_buf)==OBS_HORIZON and len(act_q)==0:
            preds=predict_actions(list(obs_buf))
            act_q.extend(preds[:EXEC_H].tolist())
            ee_now=np.array(ee.get_entity_pose().p)
            print(f"  step={step:3d} | 🧠 Diffusion predicts next {EXEC_H} actions"
                  f" | ee=({ee_now[0]:.3f},{ee_now[1]:.3f},{ee_now[2]:.3f})"
                  f" | grip={grip_val:.3f}")
            print(f"          predicted pos[0]={np.round(preds[0,:3],3)}"
                  f"  grip[0]={preds[0,6]:.4f}")

        if act_q:
            a=np.array(act_q.pop(0))
            target_pos=np.clip(a[:3],
                [TX-0.20, TY-0.25, 0.05],
                [TX+0.25, TY+0.25, 0.85])  # safety clamp
            grip_val=float(np.clip(a[6],0,0.035))
            q_tgt=solve_ik(target_pos,grip_val)
            for i,jt in enumerate(joints): jt.set_drive_target(float(q_tgt[i]))

        for _ in range(2): scene.step()
        ee_pos=np.array(ee.get_entity_pose().p)

        # Grasp detection: near marker xy, low z, gripper closed
        dxy=np.linalg.norm(ee_pos[:2]-np.array([mx,my]))
        if not grasped and grip_val<0.006 and dxy<0.07 and ee_pos[2]<0.14:
            grasped=True; grasp_step=step
            print(f"\n  ✅ GRASPED! step={step} | ee=({ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f})\n")

        if grasped and grip_val>0.018:
            grasped=False
            drop=np.linalg.norm(ee_pos[:2]-np.array([bx,by]))
            success=drop<0.10
            sim_marker.set_pose(sapien.Pose(p=[bx,by,mz]))
            print(f"\n  📦 PLACED! step={step} | dist_to_box={drop:.3f} | "
                  f"{'🎯 SUCCESS' if success else '❌ MISSED'}\n")
            if success: break

        if grasped: sim_marker.set_pose(sapien.Pose(p=ee_pos.tolist()))
        scene.update_render(); viewer.render()
        step+=1

    result='🎯 SUCCESS' if success else '❌ FAILED'
    print(f"\nEpisode {ep_idx+1}: Grasped={'step '+str(grasp_step) if grasp_step else 'NEVER'} | {result}\n")
    ep_idx=(ep_idx+1)%len(ends)