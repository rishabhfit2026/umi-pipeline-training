"""
RDT2 Real Robot Inference — M750 Direct Hardware
=================================================
Single script — no Isaac Sim, no server/client.

Setup:
  - M750 connected via USB → /dev/ttyACM0
  - Webcam (wrist mounted) → USB

Run:
    cd /home/rishabh/Downloads/umi-pipeline-training/RDT2
    source /home/rishabh/Downloads/umi-pipeline-training/umi_env/bin/activate
    python run_robot.py
"""

import os, sys, time, json, torch
import numpy as np
import cv2
from PIL import Image
import torch.distributed as dist

# ── CONFIG ────────────────────────────────────────────────────────────────────
RDT2_DIR    = "/home/rishabh/Downloads/umi-pipeline-training/RDT2"
MODEL_DIR   = "/home/rishabh/Downloads/umi-pipeline-training/outputs/rdt2-m750-v3/checkpoint-final"
VQVAE_CKPT  = "/home/rishabh/Downloads/umi-pipeline-training/outputs/vqvae-m750-7dof/vqvae_final.pt"
NORM_PATH   = "/home/rishabh/Downloads/umi-pipeline-training/outputs/m750_normalizer_7dof.pt"
INSTRUCTION = "pick the marker and place in the box"

ROBOT_PORT  = "/dev/ttyACM0"
ROBOT_BAUD  = 115200
CAMERA_ID   = 0          # try 0, 1, 2 if wrong
IMG_SIZE    = 336
VALID_LEN   = 27
VOCAB_SIZE  = 512
DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"

# Execution config
EXEC_HZ     = 10         # send commands at 10Hz
EXEC_STEPS  = 12         # execute 12 of 24 predicted steps then re-plan
SPEED       = 30         # robot movement speed (0-100)

# Safety workspace limits (from your training data)
SAFE_X      = (-800,  400)   # mm
SAFE_Y      = (-900,  300)   # mm
SAFE_Z      = ( -50,  420)   # mm
SAFE_RX     = (-180,  180)   # degrees
SAFE_RY     = (-180,  180)   # degrees
SAFE_RZ     = ( -90,  120)   # degrees
SAFE_GRIP   = (   0,  100)   # 0=closed 100=open

sys.path.insert(0, RDT2_DIR)
os.makedirs("/tmp/rdt2_robot", exist_ok=True)

print("="*60)
print("  RDT2 → M750 Real Robot")
print("="*60)
print(f"  Model  : {MODEL_DIR}")
print(f"  Robot  : {ROBOT_PORT}")
print(f"  Camera : USB wrist cam (id={CAMERA_ID})")
print(f"  Device : {DEVICE}")

# ── DISTRIBUTED INIT ─────────────────────────────────────────────────────────
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29511"
if not dist.is_initialized():
    dist.init_process_group(backend="gloo", rank=0, world_size=1)

# ── LOAD NORMALIZER + VQ-VAE ─────────────────────────────────────────────────
print("\n[1/4] Loading VQ-VAE + normalizer...")
norm  = torch.load(NORM_PATH, map_location="cpu")
mean  = norm["mean"].to(DEVICE)
std   = norm["std"].to(DEVICE)

from vqvae.models.multivqvae import MultiVQVAE
ckpt  = torch.load(VQVAE_CKPT, map_location="cpu")
vqvae = MultiVQVAE(**ckpt["config"]).eval().to(DEVICE)
vqvae.load_state_dict(ckpt["model"])
print("  ✅ VQ-VAE ready")

# ── LOAD RDT2 MODEL ───────────────────────────────────────────────────────────
print("[2/4] Loading RDT2 model...")
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

processor = AutoProcessor.from_pretrained(MODEL_DIR, use_fast=True)
base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager")
base.resize_token_embeddings(len(processor.tokenizer))
model = PeftModel.from_pretrained(base, MODEL_DIR).eval().to(DEVICE)

ACTION_TOKEN_ID   = processor.tokenizer.additional_special_tokens_ids[
    processor.tokenizer.additional_special_tokens.index("<action>")]
VOCAB_SIZE_ACTUAL = processor.tokenizer.vocab_size
print("  ✅ RDT2 ready")

@torch.no_grad()
def predict(pil_image):
    """PIL image → (24, 7) robot actions in real units"""
    img = pil_image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": INSTRUCTION}]},
        {"role": "assistant", "content": [
            {"type": "text",
             "text": f"<|quad_start|>{'<action>'*VALID_LEN}<|quad_end|>"}]}
    ]
    text   = processor.apply_chat_template(messages, add_generation_prompt=False)
    inputs = processor(text=[text], images=[[img]],
                       return_tensors="pt", padding=True).to(DEVICE)

    input_ids   = inputs["input_ids"][0]
    action_mask = (input_ids == ACTION_TOKEN_ID)
    outputs     = model(**{k: v for k, v in inputs.items() if k != "labels"})
    logits      = outputs.logits[0]

    token_ids = []
    for pos in action_mask.nonzero(as_tuple=True)[0]:
        pred = logits[pos].argmax().item()
        tok  = max(0, min(VOCAB_SIZE-1, VOCAB_SIZE_ACTUAL - (pred+1)))
        token_ids.append(tok)

    while len(token_ids) < VALID_LEN:
        token_ids.append(0)
    toks = torch.tensor(token_ids[:VALID_LEN],
                        dtype=torch.long).unsqueeze(0).to(DEVICE)

    act_norm = vqvae.decode(toks).squeeze(0)        # (24,7) normalized
    act_real = (act_norm * std + mean).cpu().numpy() # (24,7) real units
    return act_real

# ── CONNECT CAMERA ────────────────────────────────────────────────────────────
print("[3/4] Connecting camera...")
cap = None
for cam_id in [CAMERA_ID, 1, 2]:
    cap = cv2.VideoCapture(cam_id)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Warmup
        for _ in range(5):
            cap.read()
        print(f"  ✅ Camera {cam_id} ready")
        break
    cap = None

if cap is None:
    print("  ❌ No camera found!")
    sys.exit(1)

def get_frame():
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Camera read failed")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# ── CONNECT ROBOT ─────────────────────────────────────────────────────────────
print("[4/4] Connecting M750...")
try:
    from pymycobot.mycobot import MyCobot
    mc = MyCobot(ROBOT_PORT, ROBOT_BAUD)
    time.sleep(0.5)
    angles = mc.get_angles()
    coords = mc.get_coords()
    print(f"  ✅ M750 ready!")
    print(f"     Angles : {angles}")
    print(f"     Coords : {coords}")
    ROBOT_OK = True
except Exception as e:
    print(f"  ⚠️  Robot not connected: {e}")
    print("  Running in DRY RUN mode")
    ROBOT_OK = False
    mc = None

def safe_clip(actions):
    """Clip to safe workspace"""
    a = actions.copy()
    # Convert m → mm for position
    a[:,0] = np.clip(a[:,0]*1000, SAFE_X[0],  SAFE_X[1])
    a[:,1] = np.clip(a[:,1]*1000, SAFE_Y[0],  SAFE_Y[1])
    a[:,2] = np.clip(a[:,2]*1000, SAFE_Z[0],  SAFE_Z[1])
    # Convert rad → deg for rotation
    a[:,3] = np.clip(np.degrees(a[:,3]), SAFE_RX[0], SAFE_RX[1])
    a[:,4] = np.clip(np.degrees(a[:,4]), SAFE_RY[0], SAFE_RY[1])
    a[:,5] = np.clip(np.degrees(a[:,5]), SAFE_RZ[0], SAFE_RZ[1])
    # Gripper: normalize to 0-100
    a[:,6] = np.clip(a[:,6]/0.06*100, SAFE_GRIP[0], SAFE_GRIP[1])
    return a

def send_action(x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg, grip_pct):
    """Send one timestep to robot"""
    if ROBOT_OK:
        try:
            mc.send_coords(
                [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg],
                speed=SPEED, mode=0)
            mc.set_gripper_value(int(grip_pct), speed=SPEED)
        except Exception as e:
            print(f"    ⚠️  Send error: {e}")
    else:
        print(f"    [DRY] coords=({x_mm:.0f}, {y_mm:.0f}, {z_mm:.0f}mm) "
              f"rot=({rx_deg:.1f}, {ry_deg:.1f}, {rz_deg:.1f}°) "
              f"grip={grip_pct:.0f}%")

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  ROBOT READY")
print("="*60)

if ROBOT_OK:
    print("\n⚠️  Moving to HOME position in 3 seconds...")
    print("   Press Ctrl+C NOW to cancel!")
    time.sleep(3)
    mc.send_angles([0, 0, 0, 0, 0, 0], speed=20)
    time.sleep(3)
    print("✅ At home position")

input("\nPress ENTER to start inference loop (Ctrl+C to stop)...")

try:
    cycle  = 0
    dt     = 1.0 / EXEC_HZ

    while True:
        cycle += 1
        t_cycle = time.time()

        # 1. Capture frame from wrist camera
        frame = get_frame()
        frame.save(f"/tmp/rdt2_robot/frame_{cycle:04d}.jpg")

        # 2. Run RDT2 inference
        print(f"\n─── Cycle {cycle} ───────────────────────────")
        t0 = time.time()
        actions_raw = predict(frame)          # (24,7) meters/radians
        t1 = time.time()
        print(f"  Inference: {t1-t0:.2f}s")

        # 3. Safety clip + unit convert
        actions = safe_clip(actions_raw)
        # actions now in: mm, mm, mm, deg, deg, deg, 0-100

        # Print trajectory summary
        print(f"  {'Step':>4}  {'X':>6}  {'Y':>6}  {'Z':>6}  "
              f"{'RX':>6}  {'Grip':>5}")
        for t in [0, 6, 11]:
            print(f"  {t:>4}  "
                  f"{actions[t,0]:>6.0f}  "
                  f"{actions[t,1]:>6.0f}  "
                  f"{actions[t,2]:>6.0f}  "
                  f"{actions[t,3]:>6.1f}  "
                  f"{actions[t,6]:>5.0f}%")

        # 4. Execute on robot
        print(f"  Executing {EXEC_STEPS} steps at {EXEC_HZ}Hz...")
        for t in range(EXEC_STEPS):
            send_action(
                actions[t,0], actions[t,1], actions[t,2],
                actions[t,3], actions[t,4], actions[t,5],
                actions[t,6])
            time.sleep(dt)

        elapsed = time.time() - t_cycle
        print(f"  Cycle time: {elapsed:.2f}s total")

except KeyboardInterrupt:
    print("\n\n⛔ Stopped by user")
    if ROBOT_OK:
        print("Returning to home...")
        mc.send_angles([0, 0, 0, 0, 0, 0], speed=15)
        time.sleep(3)
    cap.release()
    print("✅ Done")