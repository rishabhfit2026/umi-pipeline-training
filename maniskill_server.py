"""
ManiSkill2 Server — SAPIEN Direct (no ManiSkill2 wrapper)
==========================================================
Terminal 1:
    conda activate maniskill2
    python maniskill_server.py
"""

import socket, struct, io, time, threading
import numpy as np
from PIL import Image

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 9999
IMG_WIDTH   = 640
IMG_HEIGHT  = 480
URDF_PATH   = "/home/rishabh/Downloads/myarm_m750_fixed.urdf"

print("="*55)
print("  SAPIEN M750 Server — Pick & Place")
print("="*55)

# ── SAPIEN SETUP ──────────────────────────────────────────────────────────────
import sapien.core as sapien
from sapien.core import Pose
from sapien.utils import Viewer
import numpy as np

engine  = sapien.Engine()
renderer = sapien.VulkanRenderer()
engine.set_renderer(renderer)

scene_config = sapien.SceneConfig()
scene = engine.create_scene(scene_config)
scene.set_timestep(1/100)
scene.add_ground(0)

# Lighting
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 1, -1], [1, 1, 1])
print("✅ SAPIEN scene created")

# ── LOAD M750 ROBOT ───────────────────────────────────────────────────────────
print(f"  Loading robot from: {URDF_PATH}")
loader = scene.create_urdf_loader()
loader.fix_root_link = True
loader.scale = 1.0

try:
    robot = loader.load(URDF_PATH)
    robot.set_root_pose(Pose([0, 0, 0]))
    dof = robot.dof
    joint_names = [j.name for j in robot.get_active_joints()]
    print(f"✅ Robot loaded! DOF={dof}")
    print(f"   Joints: {joint_names}")

    # Set drive properties for each joint
    joints = robot.get_active_joints()
    for j in joints:
        j.set_drive_property(stiffness=1000, damping=100)

    # Home position
    robot.set_qpos(np.zeros(dof))
    ROBOT_OK = True

except Exception as e:
    print(f"❌ Robot load failed: {e}")
    ROBOT_OK = False
    robot = None
    dof = 7

# ── ADD OBJECTS ───────────────────────────────────────────────────────────────
# Table
builder = scene.create_actor_builder()
builder.add_box_collision(half_size=[0.3, 0.3, 0.02])
builder.add_box_visual(half_size=[0.3, 0.3, 0.02],
                       material=renderer.create_material())
table = builder.build_static(name="table")
table.set_pose(Pose([0.3, 0, 0.02]))

# Marker (red)
mat_red = renderer.create_material()
mat_red.set_base_color([1, 0, 0, 1])
builder = scene.create_actor_builder()
builder.add_cylinder_collision(radius=0.015, half_length=0.06)
builder.add_cylinder_visual(radius=0.015, half_length=0.06,
                            material=mat_red)
marker = builder.build(name="marker")
marker.set_pose(Pose([0.25, -0.1, 0.10]))

# Box (blue)
mat_blue = renderer.create_material()
mat_blue.set_base_color([0, 0, 1, 1])
builder = scene.create_actor_builder()
builder.add_box_collision(half_size=[0.04, 0.04, 0.03])
builder.add_box_visual(half_size=[0.04, 0.04, 0.03],
                       material=mat_blue)
box_obj = builder.build_static(name="box")
box_obj.set_pose(Pose([0.25, 0.1, 0.07]))

print("✅ Scene objects added: table + marker (red) + box (blue)")

# ── CAMERA ────────────────────────────────────────────────────────────────────
cam_mount = scene.create_actor_builder().build_kinematic(name="cam_mount")
camera = scene.add_mounted_camera(
    name="main_cam",
    actor=cam_mount,
    pose=Pose(),
    width=IMG_WIDTH,
    height=IMG_HEIGHT,
    fovy=np.deg2rad(60),
    near=0.01,
    far=10,
)
# Position camera looking at robot workspace
cam_mount.set_pose(Pose(
    p=[0.6, 0.0, 0.8],
    q=[0.7071, -0.7071, 0, 0]  # look down
))
print("✅ Camera ready")

# ── VIEWER (GUI window) ───────────────────────────────────────────────────────
try:
    viewer = Viewer(renderer)
    viewer.set_scene(scene)
    viewer.set_camera_xyz(0.8, 0, 0.8)
    viewer.set_camera_rpy(0, -0.6, 3.14)
    viewer.window.set_cursor_enabled(False)
    USE_VIEWER = True
    print("✅ Viewer window ready")
except Exception as e:
    print(f"⚠️  Viewer: {e} — running headless")
    USE_VIEWER = False

def get_frame_bytes():
    """Render camera and return JPEG bytes"""
    scene.update_render()
    camera.take_picture()
    rgba = camera.get_color_rgba()           # (H, W, 4) float
    rgb  = (rgba[:,:,:3]*255).clip(0,255).astype(np.uint8)
    img  = Image.fromarray(rgb)
    buf  = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()

def apply_actions(actions):
    """actions: (24,7) in meters/radians — apply to robot"""
    if not ROBOT_OK:
        return
    # Use step 0 action
    x, y, z, rx, ry, rz, grip = actions[0]

    # Map to joint targets
    # rx,ry,rz → joint1,2,3 directly (radians)
    # x,y,z scaled → joint4,5,6
    target = np.zeros(dof)
    if dof >= 1: target[0] = float(np.clip(rx, -3.14, 3.14))
    if dof >= 2: target[1] = float(np.clip(ry, -3.14, 3.14))
    if dof >= 3: target[2] = float(np.clip(rz, -3.14, 3.14))
    if dof >= 4: target[3] = float(np.clip(x * 3.0, -3.14, 3.14))
    if dof >= 5: target[4] = float(np.clip(y * 3.0, -3.14, 3.14))
    if dof >= 6: target[5] = float(np.clip(z * 3.0, -3.14, 3.14))
    if dof >= 7: target[6] = float(np.clip(grip / 0.06 * 0.04, 0, 0.04))
    if dof >= 8: target[7] = float(np.clip(grip / 0.06 * 0.04, 0, 0.04))

    # Apply drive targets
    joints = robot.get_active_joints()
    for i, j in enumerate(joints[:dof]):
        j.set_drive_target(float(target[i]))

    # Step physics
    for _ in range(10):
        scene.step()

# ── SOCKET SERVER ─────────────────────────────────────────────────────────────
def run_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind((SERVER_HOST, SERVER_PORT))
    except OSError:
        print(f"❌ Port busy: sudo fuser -k {SERVER_PORT}/tcp")
        return

    s.listen(1)
    print(f"\n✅ Listening on {SERVER_HOST}:{SERVER_PORT}")
    print("   Terminal 2: python rdt2_client.py\n")

    while True:
        try:
            conn, addr = s.accept()
            print(f"✅ RDT2 client connected: {addr}")
            cycle = 0

            while True:
                cycle += 1

                # Send camera frame
                jpeg = get_frame_bytes()
                conn.sendall(struct.pack(">I", len(jpeg)) + jpeg)

                # Receive actions
                raw = conn.recv(4)
                if not raw: break
                dlen = struct.unpack(">I", raw)[0]
                data = b""
                while len(data) < dlen:
                    chunk = conn.recv(dlen - len(data))
                    if not chunk: break
                    data += chunk
                if len(data) < dlen: break

                actions = np.array(
                    struct.unpack(f">{168}f", data)).reshape(24, 7)

                print(f"  [{cycle:>3}] "
                      f"x={actions[0,0]*1000:+5.0f}mm "
                      f"y={actions[0,1]*1000:+5.0f}mm "
                      f"z={actions[0,2]*1000:+5.0f}mm "
                      f"grip={actions[0,6]/0.06*100:.0f}%")

                apply_actions(actions)
                time.sleep(0.5)

        except Exception as e:
            print(f"Error: {e}")
            try: conn.close()
            except: pass
            print("Waiting for reconnect...")

# Start server in background
thread = threading.Thread(target=run_server, daemon=True)
thread.start()

# ── MAIN RENDER LOOP ──────────────────────────────────────────────────────────
print("✅ Running! Press Ctrl+C to stop.\n")
try:
    while True:
        scene.step()
        scene.update_render()
        if USE_VIEWER:
            viewer.render()
        else:
            time.sleep(0.033)
except KeyboardInterrupt:
    print("\nStopped.")