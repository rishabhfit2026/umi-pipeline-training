"""
Replay actual recorded trajectories from sim_replay_buffer3.zarr
Plays back exact EEF positions using IK — no ML model needed
conda activate maniskill2 && python step8_replay.py
"""
import sapien, numpy as np, zarr
from scipy.spatial.transform import Rotation

URDF   = '/home/rishabh/Downloads/myarm_m750_fixed.urdf'
ZARR   = '/home/rishabh/Downloads/umi-pipeline-training/outputs/sim_replay_buffer3.zarr'

# ── Load zarr data ────────────────────────────────────────────────
print("Loading zarr data...")
z    = zarr.open(ZARR, 'r')
pos  = z['data']['robot0_eef_pos'][:]            # (N,3)
rot  = z['data']['robot0_eef_rot_axis_angle'][:] # (N,3)
grip = z['data']['robot0_gripper_width'][:]      # (N,1)
ends = z['meta']['episode_ends'][:]              # (200,)
starts = np.concatenate([[0], ends[:-1]])
print(f"Loaded {len(ends)} episodes, {len(pos)} total frames")
print(f"EEF z range: {pos[:,2].min():.3f} → {pos[:,2].max():.3f}")
print(f"Grip range:  {grip.min():.3f} → {grip.max():.3f}")

# ── Scene ─────────────────────────────────────────────────────────
scene = sapien.Scene()
scene.set_timestep(1/120)
scene.set_ambient_light([0.6,0.6,0.6])
scene.add_directional_light([0,1,-1],[1.0,0.95,0.85])
scene.add_ground(altitude=0)

loader = scene.create_urdf_loader(); loader.fix_root_link=True
robot  = loader.load(URDF); robot.set_pose(sapien.Pose(p=[0,0,0]))
joints = robot.get_active_joints(); N=len(joints)
links  = {l.name:l for l in robot.get_links()}; ee=links['gripper']
ee_idx = ee.get_index(); pm=robot.create_pinocchio_model()
for jt in joints: jt.set_drive_property(stiffness=20000, damping=2000)

q0=np.zeros(N); q0[1]=-0.3; q0[2]=0.5
robot.set_qpos(q0)
for i,jt in enumerate(joints): jt.set_drive_target(float(q0[i]))
for _ in range(200): scene.step()
real_ee=np.array(ee.get_pose().p); TX,TY=real_ee[0],real_ee[1]
print(f"EE home: ({TX:.3f},{TY:.3f})")

# ── Scene objects ─────────────────────────────────────────────────
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

# ── IK solver ─────────────────────────────────────────────────────
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

# ── Viewer ────────────────────────────────────────────────────────
viewer=scene.create_viewer()
viewer.set_camera_xyz(TX+0.8, TY-0.8, 1.2)
viewer.set_camera_rpy(0,-0.55,0.65)

# ── Reconstruct marker/box positions using same RNG as data gen ───
rng=np.random.default_rng(42)
episode_marker=[]
episode_box=[]
for _ in range(len(ends)):
    mx=TX+rng.uniform(-0.05,0.05); my=TY+rng.uniform(-0.10,0.10)
    bx=TX+rng.uniform(-0.05,0.05); by=TY+rng.uniform(0.05,0.15)
    while abs(mx-bx)<0.04 and abs(my-by)<0.04:
        bx=TX+rng.uniform(-0.05,0.05); by=TY+rng.uniform(0.05,0.15)
    episode_marker.append([mx,my,0.076])
    episode_box.append([bx,by,0.075])

print(f"\nEpisode 0 marker: {episode_marker[0]}")
print(f"Episode 0 box:    {episode_box[0]}")
print(f"\nReplaying episodes — watching exact recorded trajectories...")
print(f"Press Ctrl+C to stop\n")

ep_idx = 0
grasped = False

while not viewer.closed:
    s  = starts[ep_idx]
    e  = ends[ep_idx]
    mx,my,mz = episode_marker[ep_idx]
    bx,by,bz = episode_box[ep_idx]

    # Place marker and box
    sim_marker.set_pose(sapien.Pose(p=[mx,my,mz]))
    sim_box.set_pose(sapien.Pose(p=[bx,by,bz]))

    # Reset robot
    robot.set_qpos(q0)
    for i,jt in enumerate(joints): jt.set_drive_target(float(q0[i]))
    for _ in range(60): scene.step()
    grasped=False

    print(f"=== Episode {ep_idx+1}/{len(ends)} | "
          f"marker=({mx:.2f},{my:.2f}) box=({bx:.2f},{by:.2f}) "
          f"| frames={e-s} ===")

    # Replay every frame from zarr
    for fi in range(s, e):
        if viewer.closed: break

        target_pos  = pos[fi]       # exact recorded EEF xyz
        grip_val    = float(grip[fi,0])  # exact recorded gripper

        # Solve IK for this exact recorded position
        q_tgt = solve_ik(target_pos, grip_val)
        for i,jt in enumerate(joints):
            jt.set_drive_target(float(q_tgt[i]))

        # Step sim
        for _ in range(2): scene.step()

        # Move marker with gripper when grasped
        ee_pos = np.array(ee.get_pose().p)
        dist   = np.linalg.norm(ee_pos - np.array([mx,my,mz]))

        if not grasped and grip_val < 0.005 and dist < 0.15:
            grasped=True
            print(f"  ✅ GRASPED  frame={fi-s:3d}  "
                  f"ee=({ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f})")

        if grasped and grip_val > 0.020:
            grasped=False
            print(f"  📦 RELEASED frame={fi-s:3d}  "
                  f"ee=({ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f})")
            sim_marker.set_pose(sapien.Pose(p=[bx,by,mz]))

        if grasped:
            sim_marker.set_pose(sapien.Pose(p=ee_pos.tolist()))

        if fi % 60 == 0:
            print(f"  frame={fi-s:3d}/{e-s}  "
                  f"ee=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f})  "
                  f"grip={grip_val:.4f}")

        scene.update_render(); viewer.render()

    ep_idx = (ep_idx + 1) % len(ends)
    print(f"  Episode done. Moving to next...\n")