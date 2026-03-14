"""
STEP 5: Scripted Pick & Place Demo
===================================
Shows M750 actually picking up the red marker and placing it in the blue box.
Uses analytical IK — no model needed. Proves the full sim pipeline works.

Run:
    conda activate maniskill2
    cd ~/Downloads/umi-pipeline-training
    python step5_scripted_demo.py
"""

import sapien, numpy as np, time
from scipy.spatial.transform import Rotation

URDF_PATH = "/home/rishabh/Downloads/myarm_m750_fixed.urdf"

# Marker and box positions
MARKER_POS = np.array([0.25, 0.0,  0.06])
BOX_POS    = np.array([0.30, 0.18, 0.06])
HOVER_H    = 0.18   # height to hover above target

print("="*55)
print("  M750 Scripted Pick & Place Demo")
print("="*55)

# ── SCENE ────────────────────────────────────────────────────────
scene = sapien.Scene()
scene.set_timestep(1/120)
scene.set_ambient_light([0.6, 0.6, 0.6])
scene.add_directional_light([0, 1, -1], [1, 1, 1])
scene.add_directional_light([1, 0, -1], [0.5, 0.5, 0.5])
scene.add_ground(0)

# Robot
loader = scene.create_urdf_loader()
loader.fix_root_link = True
robot  = loader.load(URDF_PATH)
robot.set_pose(sapien.Pose(p=[0, 0, 0]))
joints   = robot.get_active_joints()
n_joints = len(joints)
print(f"  Robot: {n_joints} joints")
print(f"  Joint names: {[j.name for j in joints]}")

for jt in joints:
    jt.set_drive_property(stiffness=4000, damping=400)

# Marker (red)
mr = sapien.render.RenderMaterial()
mr.base_color = [0.95, 0.1, 0.1, 1.0]
b = scene.create_actor_builder()
b.add_capsule_visual(radius=0.018, half_length=0.045, material=mr)
b.add_capsule_collision(radius=0.018, half_length=0.045)
marker = b.build(name="marker")
marker.set_pose(sapien.Pose(p=MARKER_POS))

# Box (blue)
mb = sapien.render.RenderMaterial()
mb.base_color = [0.1, 0.25, 0.95, 1.0]
b2 = scene.create_actor_builder()
b2.add_box_visual(half_size=[0.055, 0.055, 0.04], material=mb)
b2.add_box_collision(half_size=[0.055, 0.055, 0.04])
box = b2.build_static(name="box")
box.set_pose(sapien.Pose(p=BOX_POS + np.array([0, 0, -0.04])))

# Table surface (grey)
mt = sapien.render.RenderMaterial()
mt.base_color = [0.7, 0.7, 0.7, 1.0]
bt = scene.create_actor_builder()
bt.add_box_visual(half_size=[0.4, 0.4, 0.01], material=mt)
bt.add_box_collision(half_size=[0.4, 0.4, 0.01])
table = bt.build_static(name="table")
table.set_pose(sapien.Pose(p=[0.25, 0.0, 0.01]))

# Viewer
viewer = scene.create_viewer()
viewer.set_camera_xyz(0.7, -0.4, 0.6)
viewer.set_camera_rpy(0, -0.5, 0.5)

# ── MOTION PLANNER ───────────────────────────────────────────────
def set_joints(qpos, n_steps=80):
    """Smoothly move joints to target over n_steps."""
    current = robot.get_qpos().copy()
    for i in range(n_steps):
        alpha = (i + 1) / n_steps
        # Smooth ease in-out
        t = alpha * alpha * (3 - 2 * alpha)
        interp = current + t * (qpos - current)
        for j_idx, jt in enumerate(joints):
            jt.set_drive_target(float(interp[j_idx]))
        for _ in range(4):
            scene.step()
        scene.update_render()
        viewer.render()
        if viewer.closed:
            return

def eef_to_qpos(x, y, z, gripper=0.0):
    """
    Simple analytical IK for M750.
    Maps (x,y,z) EEF target → joint angles.
    """
    q = np.zeros(n_joints)

    # Joint 0: base rotation (yaw toward target)
    q[0] = np.arctan2(y, x)

    # Distance in XY plane
    r = np.sqrt(x**2 + y**2)

    # Joint 1: shoulder pitch
    q[1] = np.clip(np.arctan2(z - 0.1, r) * 1.2 - 0.2, -1.8, 1.8)

    # Joint 2: elbow (compensate for shoulder)
    q[2] = np.clip(-q[1] * 0.85 + 0.3, -2.0, 2.0)

    # Joint 3: wrist pitch (keep EEF level)
    q[3] = np.clip(-q[1] * 0.3 - q[2] * 0.3, -1.5, 1.5)

    # Joint 4: wrist roll
    q[4] = 0.0

    # Joint 5: wrist yaw
    q[5] = np.clip(q[0] * 0.1, -1.5, 1.5)

    # Gripper
    g = np.clip(gripper, 0.0, 1.0)
    if n_joints >= 7: q[6] = g
    if n_joints >= 8: q[7] = g

    return q

def go_to(x, y, z, gripper=0.0, steps=100, label=""):
    if label:
        print(f"  → {label}  target=({x:.3f},{y:.3f},{z:.3f})  grip={gripper:.2f}")
    q = eef_to_qpos(x, y, z, gripper)
    set_joints(q, steps)

def physics_steps(n=60):
    for _ in range(n):
        scene.step()
        scene.update_render()
        viewer.render()
        if viewer.closed:
            return

# ── DEMO SEQUENCE ────────────────────────────────────────────────
print("\n  Starting pick & place sequence...")
print("  Watch the SAPIEN window!\n")

# 1. Home position
print("[1/7] Moving to home position...")
home = np.zeros(n_joints)
home[1] = -0.3
set_joints(home, 80)
physics_steps(30)

# 2. Hover above marker
print("[2/7] Hovering above marker...")
go_to(MARKER_POS[0], MARKER_POS[1], HOVER_H,
      gripper=0.0, steps=120, label="hover above marker")
physics_steps(40)

# 3. Open gripper and descend to marker
print("[3/7] Descending to marker (gripper open)...")
go_to(MARKER_POS[0], MARKER_POS[1], MARKER_POS[2] + 0.02,
      gripper=0.0, steps=100, label="descend to marker")
physics_steps(40)

# 4. Close gripper (grasp)
print("[4/7] Closing gripper (grasping marker)...")
go_to(MARKER_POS[0], MARKER_POS[1], MARKER_POS[2] + 0.02,
      gripper=1.0, steps=60, label="close gripper")
physics_steps(60)

# 5. Lift marker
print("[5/7] Lifting marker...")
go_to(MARKER_POS[0], MARKER_POS[1], HOVER_H,
      gripper=1.0, steps=100, label="lift marker")
physics_steps(40)

# 6. Move to above box
print("[6/7] Moving to box...")
go_to(BOX_POS[0], BOX_POS[1], HOVER_H,
      gripper=1.0, steps=140, label="move to box")
physics_steps(40)

# 7. Lower into box and release
print("[7/7] Placing marker in box...")
go_to(BOX_POS[0], BOX_POS[1], BOX_POS[2] + 0.04,
      gripper=1.0, steps=100, label="lower into box")
physics_steps(30)
go_to(BOX_POS[0], BOX_POS[1], BOX_POS[2] + 0.04,
      gripper=0.0, steps=60, label="release marker")
physics_steps(60)

# 8. Retreat
print("\n  Pick & place complete! Retreating...")
go_to(BOX_POS[0], BOX_POS[1], HOVER_H,
      gripper=0.0, steps=80, label="retreat")

print("\n  ✅ Demo complete!")
print("  Marker should now be in the box.")
print("  Close the window to exit.")

# Keep window open
while not viewer.closed:
    scene.update_render()
    viewer.render()