"""
ISAAC SERVER — runs INSIDE Isaac Sim Script Editor
===================================================
1. Open Isaac Sim
2. Window → Script Editor
3. Paste this entire file
4. Click Run

This script:
- Captures camera frames from Isaac Sim
- Sends them to the RDT2 inference server (running in umi_env)
- Receives predicted actions back
- Moves the robot in simulation
"""

import socket, struct, io, time, threading
import numpy as np
from PIL import Image

# ── CONFIG ────────────────────────────────────────────────────────────────────
ROBOT_PRIM    = "/myarm_m750"
CAMERA_PRIM   = "/World/Camera"   # we'll create this
SERVER_HOST   = "127.0.0.1"
SERVER_PORT   = 9999              # talks to rdt2_server.py
IMG_WIDTH     = 640
IMG_HEIGHT    = 480
ACTION_HZ     = 10

# ── ISAAC SIM IMPORTS ─────────────────────────────────────────────────────────
import omni
import omni.isaac.core as isaac_core
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.types import ArticulationAction
from pxr import UsdGeom, Gf, Sdf
import carb

print("="*50)
print("  Isaac Sim RDT2 Server")
print("="*50)

# ── WORLD SETUP ───────────────────────────────────────────────────────────────
try:
    world = World.instance()
    if world is None:
        world = World(stage_units_in_meters=1.0)
    print("  ✅ World ready")
except Exception as e:
    print(f"  Creating world: {e}")
    world = World(stage_units_in_meters=1.0)

stage = get_current_stage()

# ── ROBOT SETUP ───────────────────────────────────────────────────────────────
try:
    robot = world.scene.get_object("m750")
    if robot is None:
        robot = Robot(prim_path=ROBOT_PRIM, name="m750")
        world.scene.add(robot)
    world.reset()
    art_controller = robot.get_articulation_controller()
    dof_names = robot.dof_names
    print(f"  ✅ Robot ready. DOFs: {dof_names}")
except Exception as e:
    print(f"  ⚠️  Robot: {e}")
    art_controller = None
    dof_names = []

# ── CAMERA SETUP ─────────────────────────────────────────────────────────────
# Create a fixed overhead camera if not already present
CAMERA_PRIM_PATH = "/World/InferenceCamera"
try:
    # Create camera prim
    camera_geom = UsdGeom.Camera.Define(stage, CAMERA_PRIM_PATH)
    # Position camera above scene looking down at robot workspace
    xform = UsdGeom.Xformable(camera_geom)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 1.2))   # above robot
    xform.AddRotateXYZOp().Set(Gf.Vec3d(-90, 0, 0))        # look down
    # Set focal length
    camera_geom.GetFocalLengthAttr().Set(24.0)

    camera = Camera(
        prim_path=CAMERA_PRIM_PATH,
        resolution=(IMG_WIDTH, IMG_HEIGHT),
    )
    camera.initialize()
    print(f"  ✅ Camera created at {CAMERA_PRIM_PATH}")
    CAMERA_OK = True
except Exception as e:
    print(f"  ⚠️  Camera: {e}")
    CAMERA_OK = False

def get_frame_bytes():
    """Capture Isaac Sim frame → JPEG bytes"""
    world.step(render=True)
    if CAMERA_OK:
        try:
            rgba = camera.get_rgba()
            if rgba is not None:
                rgb = rgba[:, :, :3].astype(np.uint8)
                img = Image.fromarray(rgb)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                return buf.getvalue()
        except: pass

    # Fallback: viewport screenshot
    try:
        import omni.kit.viewport.utility as vpu
        vp = vpu.get_active_viewport()
        from omni.kit.viewport.utility import capture_viewport_to_buffer
        buf_obj = capture_viewport_to_buffer(vp)
        arr = np.frombuffer(buf_obj.data, dtype=np.uint8).reshape(
            buf_obj.height, buf_obj.width, 4)
        img = Image.fromarray(arr[:, :, :3])
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception as e:
        print(f"  Frame capture failed: {e}")
        # Return blank image
        img = Image.fromarray(np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

def apply_actions(actions_flat):
    """
    actions_flat: list of 24*7=168 floats
    [x,y,z,rx,ry,rz,grip] × 24 steps
    Applies first few steps to robot
    """
    if art_controller is None:
        return

    actions = np.array(actions_flat).reshape(24, 7)

    for step in range(min(8, 24)):  # apply first 8 steps
        x, y, z, rx, ry, rz, grip = actions[step]

        # Map EE pose → joint angles
        # Your M750 data: [x,y,z] in meters, [rx,ry,rz] in radians
        # Isaac Sim articulation takes joint angles in radians
        # Use rx,ry,rz as joint angle targets (approximation)
        # For proper IK, use omni.isaac.manipulators RMPFlow

        # Simple mapping: use rx,ry,rz for first 3 joints
        # x,y,z for last 3 joints (scaled)
        joint_targets = np.array([
            rx,              # joint1
            ry,              # joint2
            rz,              # joint3
            x * 5.0,         # joint4 (scale meters → radians approx)
            y * 5.0,         # joint5
            z * 5.0,         # joint6
        ])

        # Gripper
        grip_normalized = np.clip(grip / 0.06, 0, 1)

        try:
            art_controller.apply_action(ArticulationAction(
                joint_positions=joint_targets
            ))
            world.step(render=True)
            time.sleep(1.0 / ACTION_HZ)
        except Exception as e:
            print(f"  Action error: {e}")
            break

# ── SOCKET SERVER ─────────────────────────────────────────────────────────────
def run_server():
    """
    Protocol:
      Isaac sends: 4-byte length + JPEG bytes (camera frame)
      RDT2 sends back: 4-byte length + 168 floats as bytes (24×7 actions)
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((SERVER_HOST, SERVER_PORT))
    server.listen(1)
    print(f"\n  Isaac server listening on {SERVER_HOST}:{SERVER_PORT}")
    print("  Waiting for RDT2 inference client...")

    conn, addr = server.accept()
    print(f"  ✅ RDT2 client connected from {addr}")

    try:
        cycle = 0
        while True:
            cycle += 1

            # 1. Capture frame
            jpeg_bytes = get_frame_bytes()

            # 2. Send frame to RDT2 server
            length = struct.pack(">I", len(jpeg_bytes))
            conn.sendall(length + jpeg_bytes)

            # 3. Receive predicted actions
            raw_len = conn.recv(4)
            if not raw_len:
                break
            data_len = struct.unpack(">I", raw_len)[0]

            data = b""
            while len(data) < data_len:
                chunk = conn.recv(data_len - len(data))
                if not chunk:
                    break
                data += chunk

            # 4. Parse actions (168 floats = 24 steps × 7 dims)
            actions_flat = list(struct.unpack(f">{168}f", data))
            actions = np.array(actions_flat).reshape(24, 7)

            print(f"  Cycle {cycle}: "
                  f"x={actions[0,0]*1000:.1f}mm "
                  f"y={actions[0,1]*1000:.1f}mm "
                  f"z={actions[0,2]*1000:.1f}mm "
                  f"grip={actions[0,6]/0.06*100:.0f}%")

            # 5. Apply to robot
            apply_actions(actions_flat)

    except Exception as e:
        print(f"  Server error: {e}")
    finally:
        conn.close()
        server.close()

# Run server in background thread so Isaac Sim stays responsive
thread = threading.Thread(target=run_server, daemon=True)
thread.start()
print("\n✅ Isaac server running in background")
print("   Now run rdt2_client.py in your umi_env terminal")