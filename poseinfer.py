"""
Pose-Only Inference — no camera needed
Uses current EEF state to predict next actions

conda activate maniskill2
python infer_pose_only.py
"""

import sapien, numpy as np, torch, zarr
from scipy.spatial.transform import Rotation
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import sys
sys.path.insert(0, '/home/rishabh/Downloads/umi-pipeline-training')
SAVE_DIR = "./checkpoints_pose_only"
from poseonly import (
    PoseDiffusionNet, Normalizer,
    OBS_HORIZON, ACTION_HORIZON, ACTION_DIM, OBS_DIM
)

URDF   = '/home/rishabh/Downloads/myarm_m750_fixed.urdf'
ZARR   = '/home/rishabh/Downloads/umi-pipeline-training/outputs/sim_replay_buffer3.zarr'
CKPT   = f'{SAVE_DIR}/best_model.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_INFERENCE_STEPS = 16

# ── Load model ────────────────────────────────────────────────────
print("Loading model...")
ckpt = torch.load(CKPT, map_location=DEVICE)
model = PoseDiffusionNet(
    action_dim=ACTION_DIM, action_horizon=ACTION_HORIZON,
    obs_dim=OBS_DIM, obs_horizon=OBS_HORIZON
).to(DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"✅ Model loaded — epoch {ckpt['epoch']}, loss={ckpt['loss']:.5f}")

obs_norm = Normalizer.load(f'{SAVE_DIR}/obs_normalizer.pt')
act_norm = Normalizer.load(f'{SAVE_DIR}/act_normalizer.pt')

noise_scheduler = DDIMScheduler(
    num_train_timesteps=100,
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True, prediction_type='epsilon')
noise_scheduler.set_timesteps(NUM_INFERENCE_STEPS)

def predict_actions(obs_history):
    """
    obs_history: list of OBS_HORIZON numpy arrays, each (7,)
    returns: (ACTION_HORIZON, 7) numpy array
    """
    # ✅ FIX: normalize each 7-dim state individually, then flatten
    obs_parts = []
    for s in obs_history:
        t = torch.tensor(s, dtype=torch.float32)  # (7,)
        obs_parts.append(obs_norm.normalize(t))    # (7,)
    obs_flat = torch.cat(obs_parts).unsqueeze(0).to(DEVICE)  # (1, obs_h*7)

    with torch.no_grad():
        noisy = torch.randn(1, ACTION_HORIZON, ACTION_DIM, device=DEVICE)
        for t in noise_scheduler.timesteps:
            ts = torch.tensor([t], device=DEVICE).long()
            pred_noise = model(noisy, ts, obs_flat)
            noisy = noise_scheduler.step(pred_noise, t, noisy).prev_sample
        actions = act_norm.denormalize(noisy[0])  # (ACTION_HORIZON, 7)
    return actions.cpu().numpy()

# ── Scene ─────────────────────────────────────────────────────────
scene = sapien.Scene()
scene.set_timestep(1/120)
scene.set_ambient_light([0.6,0.6,0.6])
scene.add_directional_light([0,1,-1],[1.0,0.95,0.85])
scene.add_ground(altitude=0)

loader = scene.create_urdf_loader(); loader.fix_root_link=True
robot  = loader.load(URDF); robot.set_pose(sapien.Pose(p=[0,0,0]))
joints = robot.get_active_joints(); N=len(joints)
links  = {l.name:l for l in robot.get_links()}
ee     = links['gripper']
ee_idx = ee.get_index(); pm = robot.create_pinocchio_model()
for jt in joints: jt.set_drive_property(stiffness=20000, damping=2000)

q0=np.zeros(N); q0[1]=-0.3; q0[2]=0.5
robot.set_qpos(q0)
for i,jt in enumerate(joints): jt.set_drive_target(float(q0[i]))
for _ in range(200): scene.step()
real_ee=np.array(ee.get_entity_pose().p); TX,TY=real_ee[0],real_ee[1]
print(f"EE home: ({TX:.3f},{TY:.3f})")

def make_box(half,color,pos_,static=True,name=""):
    mt=sapien.render.RenderMaterial(); mt.base_color=color
    b=scene.create_actor_builder()
    b.add_box_visual(half_size=half,material=mt)
    b.add_box_collision(half_size=half)
    a=b.build_static(name=name) if static else b.build(name=name)
    a.set_pose(sapien.Pose(p=pos_)); return a

make_box([0.30,0.28,0.025],[0.52,0.33,0.15,1.0],[TX,TY,0.025],True,"table")
make_box([0.20,0.18,0.002],[0.96,0.96,0.94,1.0],[TX,TY,0.052],True,"mat")

mr=sapien.render.RenderMaterial(); mr.base_color=[0.95,0.10,0.10,1.0]
bm=scene.create_actor_builder(); bm.add_sphere_visual(radius=0.018,material=mr)
sim_marker=bm.build(name="marker")

mg=sapien.render.RenderMaterial(); mg.base_color=[0.05,0.80,0.15,1.0]
gb=scene.create_actor_builder()
gb.add_box_visual(half_size=[0.055,0.055,0.025],material=mg)
gb.add_box_collision(half_size=[0.055,0.055,0.025])
sim_box=gb.build_static(name="box")

def solve_ik(target_pos, grip_val):
    r   = Rotation.from_euler('xyz',[np.pi,0,0]); qv=r.as_quat()
    pose= sapien.Pose(p=list(target_pos), q=[qv[3],qv[0],qv[1],qv[2]])
    mask= np.ones(N,dtype=np.int32); mask[6:]=0
    qr,ok,_=pm.compute_inverse_kinematics(ee_idx,pose,
        initial_qpos=robot.get_qpos().astype(np.float64),
        active_qmask=mask,max_iterations=500)
    q=np.array(qr)
    if N>=7: q[6]=grip_val
    if N>=8: q[7]=grip_val
    return q

def get_eef_state():
    """Current EEF as 7-dim vector [pos(3), rot(3), grip(1)]"""
    p    = np.array(ee.get_entity_pose().p)
    rot  = np.zeros(3)  # zero rotation — model focuses on pos+grip
    grip = float(np.clip(robot.get_qpos()[6], 0, 0.035)) if N > 6 else 0.0
    return np.array([p[0],p[1],p[2], rot[0],rot[1],rot[2], grip], dtype=np.float32)

viewer=scene.create_viewer()
viewer.set_camera_xyz(TX+0.8,TY-0.8,1.2)
viewer.set_camera_rpy(0,-0.55,0.65)

z=zarr.open(ZARR,'r'); ends=z['meta']['episode_ends'][:]
rng=np.random.default_rng(42)
episode_marker=[]; episode_box=[]
for _ in range(len(ends)):
    mx=TX+rng.uniform(-0.05,0.05); my=TY+rng.uniform(-0.10,0.10)
    bx=TX+rng.uniform(-0.05,0.05); by=TY+rng.uniform(0.05,0.15)
    while abs(mx-bx)<0.04 and abs(my-by)<0.04:
        bx=TX+rng.uniform(-0.05,0.05); by=TY+rng.uniform(0.05,0.15)
    episode_marker.append([mx,my,0.076])
    episode_box.append([bx,by,0.075])

print(f"\n🤖 POSE-ONLY INFERENCE — no camera needed!")
print(f"Press Ctrl+C to stop.\n")

MAX_STEPS=450; ep_idx=0; grip_val=0.0

while not viewer.closed:
    mx,my,mz=episode_marker[ep_idx]
    bx,by,bz=episode_box[ep_idx]
    sim_marker.set_pose(sapien.Pose(p=[mx,my,mz]))
    sim_box.set_pose(sapien.Pose(p=[bx,by,bz]))

    robot.set_qpos(q0)
    for i,jt in enumerate(joints): jt.set_drive_target(float(q0[i]))
    for _ in range(60): scene.step()

    print(f"=== Episode {ep_idx+1} | "
          f"marker=({mx:.2f},{my:.2f}) box=({bx:.2f},{by:.2f}) ===")

    obs_buffer  = []
    action_queue= []
    grasped=False; step=0; success=False
    grasp_step=None; release_step=None

    while step < MAX_STEPS and not viewer.closed:
        state = get_eef_state()
        obs_buffer.append(state)
        if len(obs_buffer) > OBS_HORIZON: obs_buffer.pop(0)

        # Predict new chunk when queue empty and buffer full
        if len(action_queue) == 0 and len(obs_buffer) == OBS_HORIZON:
            actions = predict_actions(list(obs_buffer))
            action_queue.extend(actions.tolist())
            if step % 50 == 0:
                print(f"  step={step:3d}  🧠 predicted {ACTION_HORIZON} actions  "
                      f"ee=({state[0]:.3f},{state[1]:.3f},{state[2]:.3f})  grip={state[6]:.4f}")

        if action_queue:
            action     = np.array(action_queue.pop(0))
            target_pos = action[:3]
            grip_val   = float(np.clip(action[6], 0, 0.035))
            q_tgt = solve_ik(target_pos, grip_val)
            for i,jt in enumerate(joints): jt.set_drive_target(float(q_tgt[i]))

        for _ in range(2): scene.step()

        ee_pos = np.array(ee.get_entity_pose().p)
        dist   = np.linalg.norm(ee_pos - np.array([mx,my,mz]))

        if not grasped and grip_val < 0.005 and dist < 0.15:
            grasped=True; grasp_step=step
            print(f"  ✅ GRASPED   step={step:3d}  "
                  f"ee=({ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f})")

        if grasped and grip_val > 0.020:
            grasped=False; release_step=step
            drop_dist = np.linalg.norm(ee_pos[:2] - np.array([bx,by]))
            success   = drop_dist < 0.08
            sim_marker.set_pose(sapien.Pose(p=[bx,by,mz]))
            print(f"  📦 RELEASED  step={step:3d}  dist={drop_dist:.3f}  "
                  f"{'🎯 SUCCESS' if success else '❌ MISSED'}")

        if grasped:
            sim_marker.set_pose(sapien.Pose(p=ee_pos.tolist()))

        scene.update_render(); viewer.render()
        step += 1

    print(f"  Grasped : {'step '+str(grasp_step) if grasp_step else 'NEVER'}")
    print(f"  Result  : {'🎯 SUCCESS' if success else '❌ FAILED'}\n")
    ep_idx = (ep_idx + 1) % len(ends)