"""
IK-based Pick and Place — no ML needed!
Robot picks red marker and places it in green box using pure IK.
Each episode: different marker + box positions.
Press Ctrl+C to stop.

conda activate maniskill2
python ik_pickplace.py
"""

import sapien
import numpy as np
import zarr
from scipy.spatial.transform import Rotation

URDF     = '/home/rishabh/Downloads/myarm_m750_fixed.urdf'
ZARR     = '/home/rishabh/Downloads/umi-pipeline-training/outputs/sim_replay_buffer3.zarr'
OPEN     = 0.0345
CLOSE    = 0.0
MARKER_Z = 0.076

# ── Scene ─────────────────────────────────────────────────────────
scene = sapien.Scene()
scene.set_timestep(1/120)
scene.set_ambient_light([0.6, 0.6, 0.6])
scene.add_directional_light([0, 1, -1], [1.0, 0.95, 0.85])
scene.add_ground(altitude=0)

loader = scene.create_urdf_loader(); loader.fix_root_link = True
robot  = loader.load(URDF); robot.set_pose(sapien.Pose(p=[0,0,0]))
joints = robot.get_active_joints(); N = len(joints)
links  = {l.name: l for l in robot.get_links()}
ee     = links['gripper']
ee_idx = ee.get_index()
pm     = robot.create_pinocchio_model()
for jt in joints: jt.set_drive_property(stiffness=20000, damping=2000)

# Home pose
q0 = np.zeros(N); q0[1] = -0.3; q0[2] = 0.5
robot.set_qpos(q0)
for i,jt in enumerate(joints): jt.set_drive_target(float(q0[i]))
for _ in range(200): scene.step()
real_ee = np.array(ee.get_entity_pose().p)
TX, TY  = real_ee[0], real_ee[1]
print(f"EE home: ({TX:.3f}, {TY:.3f}, {real_ee[2]:.3f})")
print(f"Active joints: {N}")
for i,jt in enumerate(joints):
    print(f"  Joint {i}: {jt.name}")

# ── Scene objects ─────────────────────────────────────────────────
def make_box(half, color, pos, static=True, name=""):
    mt = sapien.render.RenderMaterial(); mt.base_color = color
    b  = scene.create_actor_builder()
    b.add_box_visual(half_size=half, material=mt)
    b.add_box_collision(half_size=half)
    a  = b.build_static(name=name) if static else b.build(name=name)
    a.set_pose(sapien.Pose(p=pos)); return a

make_box([0.30,0.28,0.025], [0.52,0.33,0.15,1.0], [TX,TY,0.025], True, "table")
make_box([0.20,0.18,0.002], [0.96,0.96,0.94,1.0], [TX,TY,0.052], True, "mat")

# Red marker (sphere)
mr = sapien.render.RenderMaterial(); mr.base_color = [0.95, 0.10, 0.10, 1.0]
bm = scene.create_actor_builder()
bm.add_sphere_visual(radius=0.022, material=mr)
bm.add_sphere_collision(radius=0.022)
sim_marker = bm.build(name="marker")

# Green box (target)
mg = sapien.render.RenderMaterial(); mg.base_color = [0.05, 0.80, 0.15, 1.0]
gb = scene.create_actor_builder()
gb.add_box_visual(half_size=[0.055,0.055,0.025], material=mg)
gb.add_box_collision(half_size=[0.055,0.055,0.025])
sim_box = gb.build_static(name="box")

# ── IK ────────────────────────────────────────────────────────────
def solve_ik(xyz, grip_val):
    r    = Rotation.from_euler('xyz', [np.pi, 0, 0]); qv = r.as_quat()
    pose = sapien.Pose(p=list(xyz), q=[qv[3],qv[0],qv[1],qv[2]])
    mask = np.ones(N, dtype=np.int32); mask[6:] = 0
    qr, ok, _ = pm.compute_inverse_kinematics(
        ee_idx, pose,
        initial_qpos=robot.get_qpos().astype(np.float64),
        active_qmask=mask, max_iterations=500)
    q = np.array(qr)
    if N >= 7: q[6] = grip_val
    if N >= 8: q[7] = grip_val
    return q, ok

def move_to(target_xyz, grip_val, n_steps=60, grasped=False):
    """Smooth interpolation to target position"""
    q_tgt, ok = solve_ik(target_xyz, grip_val)
    q_cur     = robot.get_qpos().copy()
    for i in range(n_steps):
        t  = (i + 1) / n_steps
        s  = t * t * (3 - 2 * t)          # smoothstep
        qi = q_cur + s * (q_tgt - q_cur)
        for j,jt in enumerate(joints): jt.set_drive_target(float(qi[j]))
        for _ in range(2): scene.step()
        # Keep marker attached to EE while grasped
        if grasped:
            ep = np.array(ee.get_entity_pose().p)
            sim_marker.set_pose(sapien.Pose(p=ep.tolist()))
        scene.update_render(); viewer.render()
    return ok

def set_gripper(grip_val, grasped=False, n_steps=30):
    """Open or close gripper in place"""
    q = robot.get_qpos().copy()
    q_start_grip = q[6] if N >= 7 else 0.0
    for i in range(n_steps):
        t = (i + 1) / n_steps
        g = q_start_grip + t * (grip_val - q_start_grip)
        q_cur = robot.get_qpos().copy()
        if N >= 7: q_cur[6] = g
        if N >= 8: q_cur[7] = g
        for j,jt in enumerate(joints): jt.set_drive_target(float(q_cur[j]))
        for _ in range(2): scene.step()
        if grasped:
            ep = np.array(ee.get_entity_pose().p)
            sim_marker.set_pose(sapien.Pose(p=ep.tolist()))
        scene.update_render(); viewer.render()

# ── Viewer ────────────────────────────────────────────────────────
viewer = scene.create_viewer()
viewer.set_camera_xyz(TX + 0.7, TY - 0.7, 1.1)
viewer.set_camera_rpy(0, -0.5, 0.6)

# ── Episode positions (same RNG as training data) ─────────────────
z    = zarr.open(ZARR, 'r'); ends = z['meta']['episode_ends'][:]
rng  = np.random.default_rng(42)
episode_marker = []; episode_box = []
for _ in range(len(ends)):
    mx = TX + rng.uniform(-0.05, 0.05); my = TY + rng.uniform(-0.10, 0.10)
    bx = TX + rng.uniform(-0.05, 0.05); by = TY + rng.uniform( 0.05, 0.15)
    while abs(mx-bx) < 0.04 and abs(my-by) < 0.04:
        bx = TX + rng.uniform(-0.05, 0.05); by = TY + rng.uniform(0.05, 0.15)
    episode_marker.append([mx, my, MARKER_Z])
    episode_box.append([bx, by, 0.075])

print(f"\n🤖 IK Pick & Place — {len(ends)} episodes ready")
print(f"Press Ctrl+C to stop\n")

ep_idx = 0

while not viewer.closed:
    mx, my, mz = episode_marker[ep_idx]
    bx, by, bz = episode_box[ep_idx]

    sim_marker.set_pose(sapien.Pose(p=[mx, my, mz]))
    sim_box.set_pose(sapien.Pose(p=[bx, by, bz]))

    # Reset robot to home
    robot.set_qpos(q0)
    for i,jt in enumerate(joints): jt.set_drive_target(float(q0[i]))
    for _ in range(80): scene.step()
    scene.update_render(); viewer.render()

    print(f"=== Episode {ep_idx+1}/{len(ends)} ===")
    print(f"  Marker: ({mx:.3f}, {my:.3f}, {mz:.3f})  [RED]")
    print(f"  Box:    ({bx:.3f}, {by:.3f}, {bz:.3f})  [GREEN]")

    # Print all joint values at start
    q_now = robot.get_qpos()
    print(f"  Start joints: {np.round(q_now, 3)}")

    success = False

    # ── Phase 1: Open gripper ──────────────────────────────────────
    print("  Phase 1: Opening gripper...")
    set_gripper(OPEN, grasped=False)

    # ── Phase 2: Move above marker ─────────────────────────────────
    hover_z = mz + 0.18
    print(f"  Phase 2: Hover above marker at z={hover_z:.3f}...")
    ok = move_to([mx, my, hover_z], OPEN, n_steps=80, grasped=False)
    print(f"    IK ok={ok}  ee={np.round(ee.get_entity_pose().p, 3)}")

    # ── Phase 3: Descend to marker ─────────────────────────────────
    grasp_z = mz + 0.008
    print(f"  Phase 3: Descending to marker at z={grasp_z:.3f}...")
    move_to([mx, my, grasp_z], OPEN, n_steps=70, grasped=False)
    q_now = robot.get_qpos()
    print(f"    Joint values: {np.round(q_now, 3)}")
    print(f"    EE pos: {np.round(ee.get_entity_pose().p, 3)}")

    # ── Phase 4: Close gripper (grasp) ────────────────────────────
    print("  Phase 4: Closing gripper (GRASP)...")
    set_gripper(CLOSE, grasped=False)
    # Snap marker to EE
    ep = np.array(ee.get_entity_pose().p)
    sim_marker.set_pose(sapien.Pose(p=ep.tolist()))
    grasped = True
    print(f"    GRASPED! EE={np.round(ep, 3)}")

    # ── Phase 5: Lift ──────────────────────────────────────────────
    lift_z = mz + 0.22
    print(f"  Phase 5: Lifting to z={lift_z:.3f}...")
    move_to([mx, my, lift_z], CLOSE, n_steps=60, grasped=True)

    # ── Phase 6: Move to above box ────────────────────────────────
    print(f"  Phase 6: Carrying to box ({bx:.3f},{by:.3f})...")
    move_to([bx, by, lift_z], CLOSE, n_steps=80, grasped=True)

    # ── Phase 7: Lower into box ────────────────────────────────────
    place_z = bz + 0.04
    print(f"  Phase 7: Lowering to z={place_z:.3f}...")
    move_to([bx, by, place_z], CLOSE, n_steps=60, grasped=True)

    # ── Phase 8: Release ──────────────────────────────────────────
    print("  Phase 8: Releasing (PLACE)...")
    set_gripper(OPEN, grasped=True)
    sim_marker.set_pose(sapien.Pose(p=[bx, by, mz]))
    grasped = False

    # Check success
    marker_pos = np.array(sim_marker.get_pose().p)
    box_pos    = np.array([bx, by])
    dist       = np.linalg.norm(marker_pos[:2] - box_pos)
    success    = dist < 0.08
    print(f"    Marker at {np.round(marker_pos,3)}, dist to box={dist:.3f}")

    # ── Phase 9: Retreat ──────────────────────────────────────────
    print("  Phase 9: Retreating...")
    move_to([bx, by, lift_z], OPEN, n_steps=50, grasped=False)
    move_to(real_ee, OPEN, n_steps=60, grasped=False)

    q_now = robot.get_qpos()
    print(f"  End joints:   {np.round(q_now, 3)}")
    print(f"  Result: {'🎯 SUCCESS' if success else '❌ FAILED'}\n")

    # Pause between episodes
    for _ in range(120):
        scene.update_render(); viewer.render()

    ep_idx = (ep_idx + 1) % len(ends)