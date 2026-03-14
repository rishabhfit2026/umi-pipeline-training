#!/usr/bin/env python3
"""
RDT2-VQ → myArm M750 Single-Arm Inference
============================================
- Single arm (7 dims): x, y, z, rx, ry, rz, gripper
- Pads 27 tokens → 30 for VQ-VAE decode → (24, 20)
- Takes right arm [0:10] → converts rot6d to axis-angle → 7 dims
- Unnormalizes with 7-dim normalizer
- Sends to robot when connected, prints otherwise
"""

import os, sys, time, torch, cv2, io
import numpy as np
from PIL import Image
import torch.nn.functional as F

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = "/home/rishabh/Downloads/umi-pipeline-training"
RDT2_DIR = f"{BASE_DIR}/RDT2"
CHECKPOINT_DIR = f"{BASE_DIR}/outputs/rdt2-finetuned/checkpoint-5000"
NORMALIZER_PATH = f"{BASE_DIR}/umi_normalizer_official.pt"

CAMERA_INDEX = 0
INSTRUCTION = "Pick up the marker."

# Robot config
USE_ROBOT = False  # Set True once robot serial is working
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

# Movement tuning
INTERP_ALPHA = 0.30
MOVE_SPEED = 25
GRIPPER_SPEED = 50
LOOP_DELAY = 0.15

# M750 workspace limits (mm)
X_RANGE = (-300, 300)
Y_RANGE = (-305, 305)
Z_RANGE = (20, 350)
Y_OFFSET_MM = 150.0

sys.path.insert(0, RDT2_DIR)

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from models.normalizer.normalizer import LinearNormalizer
from vqvae.models.multivqvae import MultiVQVAE

device = torch.device("cuda:0")
dtype = torch.bfloat16

# =========================================================
# LOAD MODELS
# =========================================================
print("\n🚀 Loading models...")

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    padding_side="left",
)

base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "robotics-diffusion-transformer/RDT2-VQ",
    torch_dtype=dtype,
    device_map={"": device},
    attn_implementation="flash_attention_2",
)

model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
model = model.merge_and_unload()
model.visual.to(dtype=dtype)
model.eval()

vae = MultiVQVAE.from_pretrained(
    "robotics-diffusion-transformer/RVQActionTokenizer",
    n_codebooks={"pos": 6, "rot": 3, "grip": 1},
)
vae.to(device=device, dtype=torch.float32).eval()

normalizer = LinearNormalizer.load(NORMALIZER_PATH)
normalizer.to(device)

valid_action_id_length = vae.pos_id_len + vae.rot_id_len + vae.grip_id_len  # 30
vocab_size = processor.tokenizer.vocab_size
quad_end_id = processor.tokenizer.convert_tokens_to_ids("<|quad_end|>")

print(f"   valid_action_id_length = {valid_action_id_length}")
print(f"   vocab_size = {vocab_size}")
print("✅ All models loaded\n")

# =========================================================
# CAMERA
# =========================================================
cap = None
USE_CAMERA = False

if CAMERA_INDEX >= 0:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if cap.isOpened():
        USE_CAMERA = True
        print(f"📷 Camera {CAMERA_INDEX} opened")
    else:
        print("⚠️  No camera, using dummy image")


def get_frame():
    if USE_CAMERA and cap is not None:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (384, 384))
            return frame
    return np.ones((384, 384, 3), dtype=np.uint8) * 128


# =========================================================
# ROTATION: 6D → AXIS-ANGLE
# =========================================================
def rot6d_to_axis_angle(rot6d):
    """rot6d: (..., 6) → axis_angle: (..., 3)"""
    x, y = rot6d[..., 0:3], rot6d[..., 3:6]
    b1 = F.normalize(x, dim=-1)
    b2 = F.normalize(y - (b1 * y).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    trace = b1[..., 0] + b2[..., 1] + b3[..., 2]
    angle = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0))
    safe_sin = torch.sin(angle) + 1e-6
    scale = torch.where(
        angle.abs() > 1e-4,
        angle / (2.0 * safe_sin),
        torch.tensor(0.5, device=rot6d.device),
    )
    return torch.stack([
        (b3[..., 1] - b2[..., 2]) * scale,
        (b1[..., 2] - b3[..., 0]) * scale,
        (b2[..., 0] - b1[..., 1]) * scale,
    ], dim=-1)


# =========================================================
# INFERENCE — SINGLE ARM PIPELINE
# =========================================================
def predict_action(image_np, instruction):
    """
    Returns: numpy array (24, 7) — single arm actions in real-world units
             [x, y, z, rx, ry, rz, gripper_width]
             positions in meters, rotations in radians, gripper in meters
    """
    # 1. JPEG compress (matches training)
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    jpg_bytes = cv2.imencode('.jpg', img_bgr)[1].tobytes()
    pil_img = Image.open(io.BytesIO(jpg_bytes))

    # 2. Build prompt exactly like official utils.py
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction},
        ]}
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=False)
    text += "<|im_start|>assistant\n<|quad_start|>"

    # 3. Tokenize and generate
    inputs = processor(
        text=[text],
        images=[[pil_img]],
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=(valid_action_id_length + 5),  # extra margin
            do_sample=False,
        )

    # 4. Extract generated tokens
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = generated[0, prompt_len:]

    # 5. Find <|quad_end|> and extract action tokens
    quad_end_positions = (gen_ids == quad_end_id).nonzero(as_tuple=True)[0]

    if len(quad_end_positions) > 0:
        end_idx = quad_end_positions[0].item()
        action_ids = gen_ids[0:end_idx]
    else:
        # No quad_end found — use all non-special tokens
        action_ids = gen_ids

    n_generated = len(action_ids)

    # 6. Pad to 30 tokens if needed (model may generate fewer for single-arm)
    if n_generated < valid_action_id_length:
        # Pad with the last token (better than zeros)
        pad_token = action_ids[-1] if len(action_ids) > 0 else torch.tensor(vocab_size - 1, device=device)
        padding = pad_token.repeat(valid_action_id_length - n_generated)
        action_ids = torch.cat([action_ids, padding])
    elif n_generated > valid_action_id_length:
        action_ids = action_ids[:valid_action_id_length]

    # 7. Map token IDs → VQ-VAE codebook indices (REVERSED mapping)
    action_tokens = vocab_size - (action_ids + 1)
    action_tokens = torch.clamp(action_tokens, min=0, max=vae.num_embeddings - 1)

    # 8. Decode with VQ-VAE → (1, 24, 20) bimanual
    action_tokens = action_tokens.unsqueeze(0)  # (1, 30)
    with torch.no_grad():
        raw_bimanual = vae.decode(action_tokens)  # (1, 24, 20)

    # 9. Extract RIGHT ARM only: columns [0:10]
    #    [0:3] = position (meters)
    #    [3:9] = rotation 6D
    #    [9]   = gripper width (meters)
    right_arm = raw_bimanual[0]  # (24, 20) → take first example

    pos = right_arm[:, 0:3]        # (24, 3)
    rot6d = right_arm[:, 3:9]      # (24, 6)
    grip = right_arm[:, 9:10]      # (24, 1)

    # 10. Convert rot6d → axis-angle (3 values)
    axis_angle = rot6d_to_axis_angle(rot6d)  # (24, 3)

    # 11. Combine into 7-dim: [x, y, z, rx, ry, rz, gripper]
    action_7d = torch.cat([pos, axis_angle, grip], dim=-1)  # (24, 7)
    action_7d = action_7d.unsqueeze(0)  # (1, 24, 7) for normalizer

    # 12. Unnormalize with 7-dim normalizer
    with torch.no_grad():
        real = normalizer["action"].unnormalize(action_7d)

    return real.detach().cpu().numpy()[0], n_generated  # (24, 7), n_tokens


# =========================================================
# CONVERT TO M750 COMMANDS
# =========================================================
def action_to_m750(action_7d_step):
    """
    action_7d_step: (7,) — x,y,z in meters, rx,ry,rz in radians, gripper in meters
    Returns: dict with 'coords' [x,y,z,rx,ry,rz] in mm/degrees, 'gripper' 0=open/1=closed
    """
    x_mm = action_7d_step[0] * 1000.0
    y_mm = action_7d_step[1] * 1000.0 + Y_OFFSET_MM
    z_mm = action_7d_step[2] * 1000.0

    x_mm = np.clip(x_mm, *X_RANGE)
    y_mm = np.clip(y_mm, *Y_RANGE)
    z_mm = np.clip(z_mm, *Z_RANGE)

    rx_deg = np.rad2deg(action_7d_step[3])
    ry_deg = np.rad2deg(action_7d_step[4])
    rz_deg = np.rad2deg(action_7d_step[5])

    grip_width = action_7d_step[6]
    gripper_state = 0 if grip_width > 0.04 else 1  # 0=open, 1=closed

    return {
        "coords": [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg],
        "gripper": gripper_state,
    }


# =========================================================
# ROBOT SETUP (optional)
# =========================================================
mc = None
if USE_ROBOT:
    try:
        from pymycobot.myarm import MyArm
        print(f"🔌 Connecting to M750 on {SERIAL_PORT}...")
        mc = MyArm(SERIAL_PORT, BAUD_RATE)
        time.sleep(2)
        mc.power_on()
        time.sleep(2)
        mc.set_fresh_mode(1)
        time.sleep(0.5)

        print("🏠 Moving to home...")
        mc.sync_send_angles([0, 0, 0, 0, 0, 0, 0], 20)
        time.sleep(5)

        for attempt in range(10):
            coords = mc.get_coords()
            if coords is not None and coords != -1:
                print(f"📍 Home: {coords}")
                break
            time.sleep(1)
        print("✅ Robot ready")
    except Exception as e:
        print(f"⚠️  Robot connection failed: {e}")
        print("   Continuing in print-only mode")
        mc = None
        USE_ROBOT = False

# =========================================================
# MAIN LOOP
# =========================================================
print("\n" + "=" * 60)
print("🤖 RDT2 → M750 SINGLE-ARM INFERENCE")
print(f"   Instruction : {INSTRUCTION}")
print(f"   Camera      : {'ON' if USE_CAMERA else 'DUMMY'}")
print(f"   Robot       : {'CONNECTED' if mc else 'PRINT ONLY'}")
print("=" * 60)
print("Press Ctrl+C to stop.\n")

last_sent = None
STEPS_PER_CHUNK = 4  # execute first N steps from each 24-step prediction

try:
    iteration = 0
    while True:
        iteration += 1
        frame = get_frame()

        t0 = time.time()
        action_chunk, n_tokens = predict_action(frame, INSTRUCTION)
        dt = time.time() - t0

        print(f"\n[{iteration}] ⏱ {dt:.2f}s | tokens={n_tokens} | chunk={action_chunk.shape}")

        # Execute first N steps
        for t in range(min(STEPS_PER_CHUNK, action_chunk.shape[0])):
            step = action_chunk[t]
            cmd = action_to_m750(step)

            pos_str = f"[{cmd['coords'][0]:7.1f}, {cmd['coords'][1]:7.1f}, {cmd['coords'][2]:7.1f}]"
            rot_str = f"[{cmd['coords'][3]:6.1f}, {cmd['coords'][4]:6.1f}, {cmd['coords'][5]:6.1f}]"
            grip_str = "OPEN" if cmd["gripper"] == 0 else "CLOSE"

            print(f"  t={t}: pos={pos_str}mm rot={rot_str}° grip={grip_str}")

            # Send to robot if connected
            if mc is not None:
                current = mc.get_coords()

                if current is not None and current != -1 and len(current) >= 6:
                    # Smooth interpolation for position
                    send = [
                        current[0] + (cmd["coords"][0] - current[0]) * INTERP_ALPHA,
                        current[1] + (cmd["coords"][1] - current[1]) * INTERP_ALPHA,
                        current[2] + (cmd["coords"][2] - current[2]) * INTERP_ALPHA,
                        cmd["coords"][3],
                        cmd["coords"][4],
                        cmd["coords"][5],
                    ]
                else:
                    send = cmd["coords"]

                # Clamp
                send[0] = np.clip(send[0], *X_RANGE)
                send[1] = np.clip(send[1], *Y_RANGE)
                send[2] = np.clip(send[2], *Z_RANGE)

                # Skip if barely moved
                if last_sent is not None:
                    delta = np.linalg.norm(np.array(send[:3]) - np.array(last_sent[:3]))
                    if delta < 0.5:
                        time.sleep(LOOP_DELAY)
                        continue

                mc.send_coords(send, MOVE_SPEED, 1)
                mc.set_gripper_state(cmd["gripper"], GRIPPER_SPEED)
                last_sent = send

            time.sleep(LOOP_DELAY)

except KeyboardInterrupt:
    print("\n\n🛑 Stopping...")

finally:
    if mc is not None:
        print("🏠 Returning home...")
        mc.sync_send_angles([0, 0, 0, 0, 0, 0, 0], 15)
        time.sleep(3)

    if cap is not None:
        cap.release()

    print("✅ Done.")