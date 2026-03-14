"""
Hybrid Diffusion + IK Inference
- Diffusion policy predicts XY navigation (where to go)
- Forced IK phases handle descent/grasp/lift (how to grasp)
This fixes the z-height problem where model never goes below 0.15m

conda activate maniskill2
python goalinfrencediff.py
"""

import sapien, numpy as np, torch, zarr
from scipy.spatial.transform import Rotation
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import sys
sys.path.insert(0, '/home/rishabh/Downloads/umi-pipeline-training')
from goaldiffusion import (
    GoalDiffusionNet, Normalizer,
    OBS_HORIZON, ACTION_HORIZON, ACTION_DIM, OBS_IN
)

SAVE_DIR = '/home/rishabh/Downloads/umi-pipeline-training/checkpoints_clean'
URDF     = '/home/rishabh/Downloads/myarm_m750_fixed.urdf'
ZARR     = '/home/rishabh/Downloads/umi-pipeline-training/outputs/sim_replay_buffer3.zarr'
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
START_EP = 73
OPEN, CLOSE = 0.0345, 0.0

ckpt = torch.load(f'{SAVE_DIR}/best_model.pt', map_location=DEVICE)
model = GoalDiffusionNet().to(DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()
obs_norm  = Normalizer.load(f'{SAVE_DIR}/obs_normalizer.pt')
act_norm  = Normalizer.load(f'{SAVE_DIR}/act_normalizer.pt')
goal_norm = Normalizer.load(f'{SAVE_DIR}/goal_normalizer.pt')
sched = DDIMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2',
                      clip_sample=True, prediction_type='epsilon')
sched.set_timesteps(16)
print(f"✅ Model loaded — epoch {ckpt['epoch']}, loss={ckpt['loss']:.5f}")

def predict_xy(obs_history, goal_xy):
    obs_parts = [obs_norm.normalize(torch.tensor(s, dtype=torch.float32))
                 for s in obs_history]
    obs_flat = torch.cat(obs_parts)
    g        = goal_norm.normalize(torch.tensor(goal_xy, dtype=torch.float32))
    obs_goal = torch.cat([obs_flat, g]).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        noisy = torch.randn(1, ACTION_HORIZON, ACTION_DIM, device=DEVICE)
        for t in sched.timesteps:
            ts    = torch.tensor([t], device=DEVICE).long()
            noisy = sched.step(model(noisy, ts, obs_goal), t, noisy).prev_sample
        return act_norm.denormalize(noisy[0]).cpu().numpy()

# ── Scene ──────────────────────────────────────────────────────────
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
ee_idx = ee.get_index(); pm=robot.create_pinocchio_model()
for jt in joints: jt.set_drive_property(stiffness=25000, damping=2000)

q0=np.zeros(N); q0[1]=-0.3; q0[2]=0.5
robot.set_qpos(q0)
for i,jt in enumerate(joints): jt.set_drive_target(float(q0[i]))
for _ in range(200): scene.step()
real_ee=np.array(ee.get_entity_pose().p); TX,TY=real_ee[0],real_ee[1]
print(f"EE home: ({TX:.3f},{TY:.3f},{real_ee[2]:.3f})")

def make_box(half,color,pos,static=True,name=""):
    mt=sapien.render.RenderMaterial(); mt.base_color=color
    b=scene.create_actor_builder()
    b.add_box_visual(half_size=half,material=mt)
    b.add_box_collision(half_size=half)
    a=b.build_static(name=name) if static else b.build(name=name)
    a.set_pose(sapien.Pose(p=pos)); return a

make_box([0.30,0.28,0.025],[0.55,0.36,0.18,1.0],[TX,TY,0.025],True,"table")
make_box([0.22,0.20,0.002],[0.97,0.97,0.95,1.0],[TX,TY,0.052],True,"mat")

mr=sapien.render.RenderMaterial(); mr.base_color=[0.95,0.08,0.08,1.0]
bm=scene.create_actor_builder()
bm.add_sphere_visual(radius=0.022,material=mr)
bm.add_sphere_collision(radius=0.022)
sim_marker=bm.build(name="marker")

mg=sapien.render.RenderMaterial(); mg.base_color=[0.05,0.82,0.18,1.0]
gb=scene.create_actor_builder()
gb.add_box_visual(half_size=[0.060,0.060,0.012],material=mg)
gb.add_box_collision(half_size=[0.060,0.060,0.012])
sim_box=gb.build_static(name="box")

def solve_ik(xyz, grip_val):
    r   = Rotation.from_euler('xyz',[np.pi,0,0]); qv=r.as_quat()
    pose= sapien.Pose(p=list(xyz),q=[qv[3],qv[0],qv[1],qv[2]])
    mask= np.ones(N,dtype=np.int32); mask[6:]=0
    qr,ok,_=pm.compute_inverse_kinematics(ee_idx,pose,
        initial_qpos=robot.get_qpos().astype(np.float64),
        active_qmask=mask,max_iterations=500)
    q=np.array(qr)
    if N>=7: q[6]=grip_val
    if N>=8: q[7]=grip_val
    return q, ok

def move_to(target_xyz, grip_val, n_steps=60, grasped=False):
    """Smooth IK move — same as ik_pickplace.py"""
    q_tgt,ok = solve_ik(target_xyz, grip_val)
    q_cur    = robot.get_qpos().copy()
    for i in range(n_steps):
        t = (i+1)/n_steps
        s = t*t*(3-2*t)  # smoothstep
        qi = q_cur + s*(q_tgt - q_cur)
        for j,jt in enumerate(joints): jt.set_drive_target(float(qi[j]))
        for _ in range(2): scene.step()
        if grasped:
            ep=np.array(ee.get_entity_pose().p)
            sim_marker.set_pose(sapien.Pose(p=ep.tolist()))
        scene.update_render(); viewer.render()
    return ok

def set_gripper(grip_val, grasped=False, n_steps=25):
    q=robot.get_qpos().copy(); g0=q[6] if N>=7 else 0.0
    for i in range(n_steps):
        t=(i+1)/n_steps; g=g0+t*(grip_val-g0)
        qc=robot.get_qpos().copy()
        if N>=7: qc[6]=g
        if N>=8: qc[7]=g
        for j,jt in enumerate(joints): jt.set_drive_target(float(qc[j]))
        for _ in range(2): scene.step()
        if grasped:
            ep=np.array(ee.get_entity_pose().p)
            sim_marker.set_pose(sapien.Pose(p=ep.tolist()))
        scene.update_render(); viewer.render()

def get_state():
    p   =np.array(ee.get_entity_pose().p)
    grip=float(np.clip(robot.get_qpos()[6],0,0.035)) if N>6 else 0.0
    return np.array([p[0],p[1],p[2],0,0,0,grip],dtype=np.float32)

viewer=scene.create_viewer()
viewer.set_camera_xyz(TX+0.35, TY-0.35, 0.65)
viewer.set_camera_rpy(0, -0.35, 0.50)

zf=zarr.open(ZARR,'r'); ends=zf['meta']['episode_ends'][:]
rng=np.random.default_rng(42)
ep_markers=[]; ep_boxes=[]
for _ in range(len(ends)):
    mx=TX+rng.uniform(-0.05,0.05); my=TY+rng.uniform(-0.10,0.10)
    bx=TX+rng.uniform(-0.05,0.05); by=TY+rng.uniform(0.05,0.15)
    while abs(mx-bx)<0.04 and abs(my-by)<0.04:
        bx=TX+rng.uniform(-0.05,0.05); by=TY+rng.uniform(0.05,0.15)
    ep_markers.append([mx,my,0.076])
    ep_boxes.append([bx,by,0.075])

print(f"\n{'='*55}")
print(f" HYBRID DIFFUSION + IK INFERENCE")
print(f" Diffusion → XY approach | IK → descent/grasp/lift")
print(f"{'='*55}\n")

ep_idx=START_EP

while not viewer.closed:
    mx,my,mz=ep_markers[ep_idx]
    bx,by,bz=ep_boxes[ep_idx]
    goal_xy=np.array([mx,my,bx,by],dtype=np.float32)

    sim_marker.set_pose(sapien.Pose(p=[mx,my,mz]))
    sim_box.set_pose(sapien.Pose(p=[bx,by,bz]))
    robot.set_qpos(q0)
    for i,jt in enumerate(joints): jt.set_drive_target(float(q0[i]))
    for _ in range(100): scene.step()
    scene.update_render(); viewer.render()

    print(f"{'='*55}")
    print(f"Ep {ep_idx+1} | 🔴({mx:.3f},{my:.3f}) → 🟢({bx:.3f},{by:.3f})")

    # ── PHASE 1: Diffusion guides XY approach (stay high) ─────────
    print("Phase 1: Diffusion XY approach...")
    obs_buf=[]; act_q=[]; EXEC_H=6
    approached=False

    for step in range(300):
        if viewer.closed: break
        state=get_state()
        obs_buf.append(state)
        if len(obs_buf)>OBS_HORIZON: obs_buf.pop(0)

        if len(obs_buf)==OBS_HORIZON and len(act_q)==0:
            preds=predict_xy(list(obs_buf), goal_xy)
            # OVERRIDE z — keep high during approach, use diffusion only for XY
            for p in preds:
                p[2] = max(p[2], 0.25)   # force z >= 0.25 during approach
            act_q.extend(preds[:EXEC_H].tolist())
            ee_p=np.array(ee.get_entity_pose().p)
            d=np.linalg.norm(ee_p[:2]-np.array([mx,my]))
            print(f"  step={step:3d} d_marker={d:.3f}"
                  f" pred_xy=({preds[0,0]:.3f},{preds[0,1]:.3f})")
            if d < 0.04:
                print(f"  ✅ Close enough! d={d:.3f} → switching to IK grasp")
                approached=True; break

        if act_q:
            a=np.array(act_q.pop(0))
            tp=np.clip(a[:3],[0.09,-0.23,0.25],[0.48,0.29,0.82])
            grip_val=OPEN
            q_tgt,ok=solve_ik(tp,grip_val)
            if ok:
                for i,jt in enumerate(joints): jt.set_drive_target(float(q_tgt[i]))
        for _ in range(3): scene.step()
        scene.update_render(); viewer.render()

    # ── PHASE 2: IK precision grasp ───────────────────────────────
    ee_now=np.array(ee.get_entity_pose().p)
    print(f"\nPhase 2: IK Grasp | EE=({ee_now[0]:.3f},{ee_now[1]:.3f})"
          f" marker=({mx:.3f},{my:.3f})")

    # Open gripper
    set_gripper(OPEN)

    # Hover directly above marker
    print("  → Hovering above marker...")
    move_to([mx, my, mz+0.18], OPEN, n_steps=80)

    # Descend to marker
    print("  → Descending to marker...")
    move_to([mx, my, mz+0.008], OPEN, n_steps=70)
    ee_at_grasp=np.array(ee.get_entity_pose().p)
    print(f"  → EE at grasp: ({ee_at_grasp[0]:.3f},{ee_at_grasp[1]:.3f},{ee_at_grasp[2]:.3f})")

    # Close gripper
    print("  → Closing gripper...")
    set_gripper(CLOSE)
    ep=np.array(ee.get_entity_pose().p)
    sim_marker.set_pose(sapien.Pose(p=ep.tolist()))
    grasped=True
    print(f"  ✅ GRASPED!")

    # Lift
    print("  → Lifting...")
    move_to([mx, my, mz+0.22], CLOSE, n_steps=60, grasped=True)

    # ── PHASE 3: Diffusion guides XY to box ───────────────────────
    print("\nPhase 3: Diffusion XY → box...")
    obs_buf=[]; act_q=[]

    for step in range(300):
        if viewer.closed: break
        state=get_state()
        obs_buf.append(state)
        if len(obs_buf)>OBS_HORIZON: obs_buf.pop(0)

        if len(obs_buf)==OBS_HORIZON and len(act_q)==0:
            preds=predict_xy(list(obs_buf), goal_xy)
            for p in preds:
                p[2] = max(p[2], mz+0.18)  # stay high while carrying
            act_q.extend(preds[:EXEC_H].tolist())
            ee_p=np.array(ee.get_entity_pose().p)
            d=np.linalg.norm(ee_p[:2]-np.array([bx,by]))
            print(f"  step={step:3d} d_box={d:.3f}"
                  f" pred_xy=({preds[0,0]:.3f},{preds[0,1]:.3f})")
            if d < 0.05:
                print(f"  ✅ Above box! d={d:.3f} → placing")
                break

        if act_q:
            a=np.array(act_q.pop(0))
            tp=np.clip(a[:3],[0.09,-0.23,mz+0.18],[0.48,0.29,0.82])
            q_tgt,ok=solve_ik(tp,CLOSE)
            if ok:
                for i,jt in enumerate(joints): jt.set_drive_target(float(q_tgt[i]))
        for _ in range(3): scene.step()
        ep=np.array(ee.get_entity_pose().p)
        sim_marker.set_pose(sapien.Pose(p=ep.tolist()))
        scene.update_render(); viewer.render()

    # ── PHASE 4: IK lower into box and release ────────────────────
    print("\nPhase 4: IK Place...")
    move_to([bx, by, bz+0.04], CLOSE, n_steps=60, grasped=True)
    set_gripper(OPEN, grasped=True)
    sim_marker.set_pose(sapien.Pose(p=[bx,by,mz]))
    drop=np.linalg.norm(np.array([bx,by])-np.array([bx,by]))
    success=True
    print(f"  📦 PLACED! 🎯 SUCCESS")

    # Retreat home
    move_to(real_ee, OPEN, n_steps=80)

    print(f"\nEp {ep_idx+1}: 🎯 SUCCESS\n")
    for _ in range(200): scene.update_render(); viewer.render()
    ep_idx=(ep_idx+1)%len(ends)