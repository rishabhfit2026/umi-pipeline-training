"""
Sim data generation — FIXED camera quaternion
Generates 200 episodes of pick+place with correct top-down camera images
conda activate maniskill2
python step10_fixed.py
"""
import sapien
import numpy as np
import zarr, os, time, shutil
from scipy.spatial.transform import Rotation

URDF       = '/home/rishabh/Downloads/myarm_m750_fixed.urdf'
OUT_ZARR   = '/home/rishabh/Downloads/umi-pipeline-training/outputs/sim_replay_buffer3.zarr'
N_EPISODES = 200
IMG_SIZE   = 224
OPEN       = 0.0345
CLOSE      = 0.0
MARKER_Z   = 0.076

# ── Scene ────────────────────────────────────────────────────────
scene = sapien.Scene()
scene.set_timestep(1/60)
scene.set_ambient_light([0.6, 0.6, 0.6])
scene.add_directional_light([0, 1, -1], [1.0, 0.95, 0.85])
scene.add_ground(altitude=0)

loader = scene.create_urdf_loader()
loader.fix_root_link = True
robot  = loader.load(URDF)
robot.set_pose(sapien.Pose(p=[0,0,0]))
joints = robot.get_active_joints()
N      = len(joints)
links  = {l.name: l for l in robot.get_links()}
ee     = links['gripper']
ee_idx = ee.get_index()
pm     = robot.create_pinocchio_model()
for jt in joints: jt.set_drive_property(stiffness=20000, damping=2000)

# Settle robot to find EE home
q0 = np.zeros(N); q0[1]=-0.3; q0[2]=0.5
robot.set_qpos(q0)
for i,jt in enumerate(joints): jt.set_drive_target(float(q0[i]))
for _ in range(200): scene.step()
real_ee = np.array(ee.get_pose().p)
TX, TY  = real_ee[0], real_ee[1]
print(f"EE home: ({TX:.3f},{TY:.3f},{real_ee[2]:.3f})")

# ── Static scene objects ─────────────────────────────────────────
def make_box(half, color, pos, static=True, name=""):
    mt = sapien.render.RenderMaterial(); mt.base_color=color
    b  = scene.create_actor_builder()
    b.add_box_visual(half_size=half, material=mt)
    b.add_box_collision(half_size=half)
    a  = b.build_static(name=name) if static else b.build(name=name)
    a.set_pose(sapien.Pose(p=pos)); return a

# Table + mat (bright colours so camera sees them clearly)
make_box([0.30,0.28,0.025], [0.52,0.33,0.15,1.0], [TX,TY,0.025],  True, "table")
make_box([0.20,0.18,0.002], [0.96,0.96,0.94,1.0], [TX,TY,0.052],  True, "mat")
for lx,ly in [(TX+0.27,TY+0.23),(TX+0.27,TY-0.23),
              (TX-0.27,TY+0.23),(TX-0.27,TY-0.23)]:
    make_box([0.02,0.02,0.025],[0.38,0.22,0.10,1.0],[lx,ly,0.012],True,"leg")

# Red marker (no collision — robot passes through)
mr = sapien.render.RenderMaterial(); mr.base_color=[0.95,0.10,0.10,1.0]
bm = scene.create_actor_builder()
bm.add_sphere_visual(radius=0.018, material=mr)
sim_marker = bm.build(name="marker")

# Green box
mg = sapien.render.RenderMaterial(); mg.base_color=[0.05,0.80,0.15,1.0]
gb = scene.create_actor_builder()
gb.add_box_visual(half_size=[0.055,0.055,0.025], material=mg)
gb.add_box_collision(half_size=[0.055,0.055,0.025])
sim_box = gb.build_static(name="box")

# ── Virtual top-down camera ───────────────────────────────────────
# Test ALL quaternions to find which one looks straight down
# In SAPIEN: camera default looks along +X. We need it to look along -Z (down).
# Rotation: -90deg around Y axis
# scipy: Rotation.from_euler('y', -90, degrees=True).as_quat() → [0, -0.707, 0, 0.707]
# SAPIEN uses wxyz: w=0.707, x=0, y=-0.707, z=0
cam_ent = sapien.Entity()
cam     = sapien.render.RenderCameraComponent(IMG_SIZE, IMG_SIZE)
cam.set_fovy(float(np.deg2rad(60)), True)
cam.near = 0.01
cam.far  = 10.0
cam_ent.add_component(cam)
cam_ent.set_pose(sapien.Pose(
    p=[TX, TY, 0.85],
    q=[0.7071068, 0, 0.7071068, 0]    # F quaternion — confirmed sees scene
))
scene.add_entity(cam_ent)

# ── Viewer (with camera shown) ────────────────────────────────────
viewer = scene.create_viewer()
viewer.set_camera_xyz(TX+0.8, TY-0.8, 1.2)
viewer.set_camera_rpy(0, -0.55, 0.65)

# ── Verify camera sees the scene ─────────────────────────────────
print("\nVerifying camera sees scene...")
sim_marker.set_pose(sapien.Pose(p=[TX, TY, MARKER_Z]))
sim_box.set_pose(sapien.Pose(p=[TX, TY+0.12, 0.075]))
for _ in range(10): scene.step()
scene.update_render()
cam.take_picture()
rgba = cam.get_picture('Color')
img  = (np.clip(rgba[:,:,:3],0,1)*255).astype(np.uint8)
mean_brightness = img.mean()
print(f"Camera image mean brightness: {mean_brightness:.1f}  "
      f"({'✅ OK' if mean_brightness > 30 else '❌ TOO DARK — camera wrong'})")

# Save a test frame so you can verify visually
import cv2
os.makedirs('/home/rishabh/Downloads/umi-pipeline-training/outputs', exist_ok=True)
cv2.imwrite('/home/rishabh/Downloads/umi-pipeline-training/outputs/camera_test.png',
            img[:,:,::-1])
print("Saved test frame → outputs/camera_test.png")
print("Open it to verify camera sees table+marker+box before running 200 episodes\n")

# Wait for user to check
print("Starting generation automatically...")

# ── IK ────────────────────────────────────────────────────────────
def solve_ik(xyz, grip_val):
    r    = Rotation.from_euler('xyz',[np.pi,0,0]); qv=r.as_quat()
    pose = sapien.Pose(p=list(xyz), q=[qv[3],qv[0],qv[1],qv[2]])
    mask = np.ones(N,dtype=np.int32); mask[6:]=0
    qr,ok,_ = pm.compute_inverse_kinematics(ee_idx, pose,
        initial_qpos=robot.get_qpos().astype(np.float64),
        active_qmask=mask, max_iterations=500)
    q = np.array(qr)
    if N>=7: q[6]=grip_val
    if N>=8: q[7]=grip_val
    return q, ok

def move_to(xyz, grip_val, steps, grasped, frames):
    q_tgt,_ = solve_ik(xyz, grip_val)
    q_cur   = robot.get_qpos().copy()
    for i in range(steps):
        t = (i+1)/steps; s = t*t*(3-2*t)
        qi = q_cur + s*(q_tgt-q_cur)
        for j,jt in enumerate(joints): jt.set_drive_target(float(qi[j]))
        for _ in range(3): scene.step()
        if grasped:
            p = np.array(ee.get_pose().p)
            sim_marker.set_pose(sapien.Pose(p=p.tolist()))
        # Capture frame
        scene.update_render()
        cam.take_picture()
        rgba = cam.get_picture('Color')
        rgb  = (np.clip(rgba[:,:,:3],0,1)*255).astype(np.uint8)
        ep_p = np.array(ee.get_pose().p, dtype=np.float32)
        ep_q = np.array(ee.get_pose().q, dtype=np.float32)
        rot_aa = Rotation.from_quat([ep_q[1],ep_q[2],ep_q[3],ep_q[0]]).as_euler('xyz').astype(np.float32)
        grip_w = np.array([robot.get_qpos()[6] if N>=7 else 0.0], dtype=np.float32)
        frames.append((rgb, ep_p, rot_aa, grip_w))
        # Show in viewer
        viewer.render()

def run_episode(mx, my, bx, by):
    frames  = []
    grasped = False

    sim_marker.set_pose(sapien.Pose(p=[mx, my, MARKER_Z]))
    sim_box.set_pose(sapien.Pose(p=[bx, by, 0.075]))

    # Reset robot
    robot.set_qpos(q0)
    for i,jt in enumerate(joints): jt.set_drive_target(float(q0[i]))
    for _ in range(60): scene.step()

    # 1. Open gripper
    q1 = robot.get_qpos().copy()
    if N>=7: q1[6]=OPEN
    if N>=8: q1[7]=OPEN
    move_to(real_ee, OPEN, 20, False, frames)

    # 2. Hover above marker
    move_to([mx, my, MARKER_Z+0.15], OPEN, 50, False, frames)

    # 3. Descend to marker
    move_to([mx, my, MARKER_Z+0.01], OPEN, 60, False, frames)

    # 4. Close gripper (grasp)
    q4 = robot.get_qpos().copy()
    if N>=7: q4[6]=CLOSE
    if N>=8: q4[7]=CLOSE
    for j,jt in enumerate(joints): jt.set_drive_target(float(q4[j]))
    for _ in range(40): scene.step()
    p = np.array(ee.get_pose().p)
    sim_marker.set_pose(sapien.Pose(p=p.tolist()))
    grasped = True
    # Record grasp frames
    for _ in range(15):
        scene.update_render(); cam.take_picture()
        rgba = cam.get_picture('Color')
        rgb  = (np.clip(rgba[:,:,:3],0,1)*255).astype(np.uint8)
        ep_p = np.array(ee.get_pose().p, dtype=np.float32)
        ep_q = np.array(ee.get_pose().q, dtype=np.float32)
        rot_aa = Rotation.from_quat([ep_q[1],ep_q[2],ep_q[3],ep_q[0]]).as_euler('xyz').astype(np.float32)
        grip_w = np.array([CLOSE], dtype=np.float32)
        frames.append((rgb, ep_p, rot_aa, grip_w))
        viewer.render()

    # 5. Lift
    move_to([mx, my, MARKER_Z+0.20], CLOSE, 50, True, frames)

    # 6. Carry to box
    move_to([bx, by, MARKER_Z+0.20], CLOSE, 60, True, frames)

    # 7. Lower into box
    move_to([bx, by, MARKER_Z+0.05], CLOSE, 50, True, frames)

    # 8. Release
    q8 = robot.get_qpos().copy()
    if N>=7: q8[6]=OPEN
    if N>=8: q8[7]=OPEN
    for j,jt in enumerate(joints): jt.set_drive_target(float(q8[j]))
    for _ in range(40): scene.step()
    sim_marker.set_pose(sapien.Pose(p=[bx, by, MARKER_Z]))
    grasped = False
    for _ in range(15):
        scene.update_render(); cam.take_picture()
        rgba = cam.get_picture('Color')
        rgb  = (np.clip(rgba[:,:,:3],0,1)*255).astype(np.uint8)
        ep_p = np.array(ee.get_pose().p, dtype=np.float32)
        ep_q = np.array(ee.get_pose().q, dtype=np.float32)
        rot_aa = Rotation.from_quat([ep_q[1],ep_q[2],ep_q[3],ep_q[0]]).as_euler('xyz').astype(np.float32)
        grip_w = np.array([OPEN], dtype=np.float32)
        frames.append((rgb, ep_p, rot_aa, grip_w))
        viewer.render()

    # 9. Retreat
    move_to([bx, by, MARKER_Z+0.22], OPEN, 40, False, frames)

    return frames

# ── Generate all episodes ─────────────────────────────────────────
print(f"Generating {N_EPISODES} episodes...")
all_imgs, all_pos, all_rot, all_grip, episode_ends = [], [], [], [], []
total = 0
rng   = np.random.default_rng(42)
t0    = time.time()

for ep in range(N_EPISODES):
    mx = TX + rng.uniform(-0.05, 0.05)
    my = TY + rng.uniform(-0.10, 0.10)
    bx = TX + rng.uniform(-0.05, 0.05)
    by = TY + rng.uniform( 0.05, 0.15)
    # Keep marker and box apart
    while abs(mx-bx)<0.04 and abs(my-by)<0.04:
        bx = TX + rng.uniform(-0.05, 0.05)
        by = TY + rng.uniform( 0.05, 0.15)

    frames = run_episode(mx, my, bx, by)

    for rgb, pos, rot, grip in frames:
        all_imgs.append(rgb); all_pos.append(pos)
        all_rot.append(rot);  all_grip.append(grip)
    total += len(frames)
    episode_ends.append(total)

    elapsed = time.time()-t0
    eta     = elapsed/(ep+1)*(N_EPISODES-ep-1)
    print(f"  [{ep+1:3d}/{N_EPISODES}] frames={len(frames):3d} "
          f"total={total:6d}  elapsed={elapsed:.0f}s  eta={eta:.0f}s")

# ── Save zarr ─────────────────────────────────────────────────────
print(f"\nSaving to {OUT_ZARR} ...")
if os.path.exists(OUT_ZARR): shutil.rmtree(OUT_ZARR)
root = zarr.open(OUT_ZARR, mode='w')
data = root.require_group('data')
meta = root.require_group('meta')

imgs = np.stack(all_imgs).astype(np.uint8)
pos  = np.stack(all_pos).astype(np.float32)
rot  = np.stack(all_rot).astype(np.float32)
grip = np.stack(all_grip).astype(np.float32)
ends = np.array(episode_ends, dtype=np.int64)

data.array('camera0_rgb',               imgs,  chunks=(1,224,224,3), dtype='uint8')
data.array('robot0_eef_pos',            pos,   chunks=(1000,3),      dtype='float32')
data.array('robot0_eef_rot_axis_angle', rot,   chunks=(1000,3),      dtype='float32')
data.array('robot0_gripper_width',      grip,  chunks=(1000,1),      dtype='float32')
meta.array('episode_ends',              ends,  chunks=(1000,),       dtype='int64')

print(f"\n✅ Done!")
print(f"   Episodes:     {N_EPISODES}")
print(f"   Total frames: {total}")
print(f"   Image mean:   {imgs.mean():.1f}  (should be > 30)")
print(f"   EEF z range:  {pos[:,2].min():.3f} → {pos[:,2].max():.3f}")
print(f"   Grip range:   {grip.min():.3f} → {grip.max():.3f}")
print(f"\nNext: verify images then retrain")
print(f"   python viz_training_data.py")
print(f"   source umi_env/bin/activate && cd RDT2 && python step3_sim.py")