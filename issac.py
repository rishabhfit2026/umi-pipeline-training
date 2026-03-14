"""
Isaac Sim RDT2 Server - v5
NO viewport capture — uses test image to avoid freeze
Paste into Isaac Sim Script Editor → Ctrl+Enter
"""
import socket, struct, io, time, threading
import numpy as np
from PIL import Image, ImageDraw

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 9999
IMG_WIDTH   = 640
IMG_HEIGHT  = 480

from omni.isaac.core import World
from omni.isaac.core.utils.stage import get_current_stage
from pxr import UsdGeom, Gf, UsdPhysics

print("="*50)
print("  Isaac Sim RDT2 Server v5")
print("="*50)

world = World.instance()
stage = get_current_stage()
print("✅ World ready")

# ── JOINTS ────────────────────────────────────────────────────────────────────
def get_joint_prims():
    joints = {}
    prim = stage.GetPrimAtPath("/myarm_m750/joints")
    if prim.IsValid():
        for child in prim.GetChildren():
            joints[child.GetName()] = child
        print(f"✅ Joints: {list(joints.keys())}")
    else:
        print("⚠️  No joints found")
    return joints

joint_prims = get_joint_prims()

def set_joint(name, angle_rad):
    if name not in joint_prims:
        return
    try:
        drive = UsdPhysics.DriveAPI.Get(joint_prims[name], "angular")
        if drive:
            drive.GetTargetPositionAttr().Set(float(np.degrees(angle_rad)))
    except: pass

def apply_actions(actions):
    x, y, z, rx, ry, rz, grip = actions[0]
    for name, val in [("joint1", rx), ("joint2", ry), ("joint3", rz),
                      ("joint4", x*2.0), ("joint5", y*2.0), ("joint6", z*2.0)]:
        set_joint(name, float(np.clip(val, -3.14, 3.14)))

# ── FRAME — pre-load from training shards, NO viewport capture ───────────────
import glob, tarfile

def load_test_image():
    """Load one real image from shards — no Isaac Sim capture needed"""
    try:
        tars = sorted(glob.glob(
            "/home/rishabh/Downloads/umi-pipeline-training/shards/*.tar"))
        if tars:
            with tarfile.open(tars[0]) as t:
                for m in t.getmembers():
                    if m.name.endswith(".image.jpg"):
                        raw = t.extractfile(m).read()
                        img = Image.open(io.BytesIO(raw)).convert("RGB")
                        w, h = img.size
                        img = img.crop((0, 0, h, h))  # left half
                        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
                        print(f"✅ Test image loaded from shards: {img.size}")
                        return img
    except Exception as e:
        print(f"⚠️  Shard image load failed: {e}")

    # Fallback: grey image with text
    img = Image.fromarray(
        np.ones((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8) * 128)
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "M750 Test", fill=(255,255,255))
    print("✅ Using grey test image")
    return img

test_image = load_test_image()
buf = io.BytesIO()
test_image.save(buf, format="JPEG", quality=85)
TEST_JPEG = buf.getvalue()
print(f"  Frame size: {len(TEST_JPEG)} bytes")

def get_frame_bytes():
    # Always return the same test image — zero Isaac Sim load
    return TEST_JPEG

# ── SERVER ────────────────────────────────────────────────────────────────────
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
    print("   Run: python rdt2_client.py\n")

    while True:
        try:
            conn, addr = s.accept()
            print(f"✅ Client connected: {addr}")
            cycle = 0

            while True:
                cycle += 1

                # Send frame (no viewport capture — no freeze!)
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
                      f"x={actions[0,0]*1000:+6.0f}mm "
                      f"y={actions[0,1]*1000:+6.0f}mm "
                      f"z={actions[0,2]*1000:+6.0f}mm "
                      f"grip={actions[0,6]/0.06*100:4.0f}%")

                # Move robot joints
                apply_actions(actions)

                # Small delay — let Isaac Sim render
                time.sleep(1.0)

        except Exception as e:
            print(f"Error: {e}")
            try: conn.close()
            except: pass
            print("Waiting for reconnect...")

thread = threading.Thread(target=run_server, daemon=True)
thread.start()
print("✅ Server ready — no viewport capture, no freeze!")