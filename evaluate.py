"""
Model Evaluation — check what the model actually learned
Run this BEFORE inference to understand model quality

conda activate maniskill2  
python evaluate_model.py
"""

import numpy as np, zarr, torch, sys, os
sys.path.insert(0, '/home/rishabh/Downloads/umi-pipeline-training')

SAVE_DIR = '/home/rishabh/Downloads/umi-pipeline-training/checkpoints_pose_only'

from poseonly import (
    PoseDiffusionNet, Normalizer,
    OBS_HORIZON, ACTION_HORIZON, ACTION_DIM, OBS_DIM
)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

ZARR   = '/home/rishabh/Downloads/umi-pipeline-training/outputs/sim_replay_buffer3.zarr'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Load model ────────────────────────────────────────────────────
ckpt  = torch.load(f'{SAVE_DIR}/best_model.pt', map_location=DEVICE)
model = PoseDiffusionNet(ACTION_DIM, ACTION_HORIZON, OBS_DIM, OBS_HORIZON).to(DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"Model: epoch={ckpt['epoch']}  loss={ckpt['loss']:.5f}")

obs_norm = Normalizer.load(f'{SAVE_DIR}/obs_normalizer.pt')
act_norm = Normalizer.load(f'{SAVE_DIR}/act_normalizer.pt')

sched = DDIMScheduler(num_train_timesteps=100,
                      beta_schedule='squaredcos_cap_v2',
                      clip_sample=True, prediction_type='epsilon')
sched.set_timesteps(16)

# ── Load data ─────────────────────────────────────────────────────
z      = zarr.open(ZARR, 'r')
pos    = z['data']['robot0_eef_pos'][:]
rot    = z['data']['robot0_eef_rot_axis_angle'][:]
grip   = z['data']['robot0_gripper_width'][:]
ends   = z['meta']['episode_ends'][:]
starts = np.concatenate([[0], ends[:-1]])
states = np.concatenate([pos, rot, grip], axis=-1).astype(np.float32)

print(f"\nDataset: {len(ends)} episodes, {len(states)} frames")
print(f"EEF x: {pos[:,0].min():.3f} → {pos[:,0].max():.3f}")
print(f"EEF y: {pos[:,1].min():.3f} → {pos[:,1].max():.3f}")
print(f"EEF z: {pos[:,2].min():.3f} → {pos[:,2].max():.3f}")
print(f"Grip:  {grip.min():.4f} → {grip.max():.4f}")

def predict(obs_history):
    obs_parts = []
    for s in obs_history:
        t = torch.tensor(s, dtype=torch.float32)
        obs_parts.append(obs_norm.normalize(t))
    obs_flat = torch.cat(obs_parts).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        noisy = torch.randn(1, ACTION_HORIZON, ACTION_DIM, device=DEVICE)
        for t in sched.timesteps:
            ts = torch.tensor([t], device=DEVICE).long()
            noisy = sched.step(model(noisy, ts, obs_flat), t, noisy).prev_sample
        return act_norm.denormalize(noisy[0]).cpu().numpy()

# ── Evaluate on training episodes ────────────────────────────────
print("\n" + "="*60)
print("EVALUATION: Model predictions vs ground truth")
print("="*60)

errors_pos  = []
errors_grip = []
grip_correct = 0
total_checks = 0

# Test on 10 episodes
test_eps = [0, 10, 20, 30, 40, 50, 60, 70, 73, 74]  # ep 74 is key!

for ep in test_eps:
    s, e = int(starts[ep]), int(ends[ep])
    ep_states = states[s:e]
    ep_len    = len(ep_states)

    # Test at 3 points: start, grasp moment, carry moment
    test_frames = [
        OBS_HORIZON,           # start of episode
        ep_len // 3,           # approaching marker
        ep_len // 2,           # around grasp
    ]

    ep_pos_err  = []
    ep_grip_err = []

    for t_rel in test_frames:
        if t_rel < OBS_HORIZON or t_rel + ACTION_HORIZON >= ep_len:
            continue
        t = s + t_rel

        # Ground truth next actions
        gt_actions = states[t : t + ACTION_HORIZON]  # (16, 7)

        # Model prediction
        obs_hist = [states[t - OBS_HORIZON + 1 + i] for i in range(OBS_HORIZON)]
        pred     = predict(obs_hist)                  # (16, 7)

        # Errors
        pos_err  = np.linalg.norm(pred[:, :3] - gt_actions[:, :3], axis=-1).mean()
        grip_err = np.abs(pred[:, 6] - gt_actions[:, 6]).mean()
        ep_pos_err.append(pos_err)
        ep_grip_err.append(grip_err)

        # Check if gripper open/close direction is correct
        gt_grip_dir   = gt_actions[0, 6] < 0.01   # True = closing
        pred_grip_dir = pred[0, 6] < 0.01
        if gt_grip_dir == pred_grip_dir:
            grip_correct += 1
        total_checks += 1

    avg_pos  = np.mean(ep_pos_err)  if ep_pos_err  else 0
    avg_grip = np.mean(ep_grip_err) if ep_grip_err else 0
    errors_pos.append(avg_pos)
    errors_grip.append(avg_grip)

    marker_str = f"ep{ep+1:3d}"
    quality = "✅ GOOD" if avg_pos < 0.05 else ("⚠️  OK" if avg_pos < 0.10 else "❌ BAD")
    print(f"  {marker_str}: pos_err={avg_pos:.4f}m  grip_err={avg_grip:.4f}  {quality}")

print(f"\nOverall:")
print(f"  Mean position error : {np.mean(errors_pos):.4f} m")
print(f"  Mean gripper error  : {np.mean(errors_grip):.4f}")
print(f"  Grip direction acc  : {grip_correct}/{total_checks} = {100*grip_correct/max(total_checks,1):.0f}%")

print("\n" + "="*60)
print("EPISODE 74 DETAILED ANALYSIS")
print("="*60)
ep = 73  # 0-indexed
s, e = int(starts[ep]), int(ends[ep])
ep_states = states[s:e]
print(f"Episode 74: frames {s}→{e}  ({e-s} frames)")
print(f"  Start EEF: {ep_states[0,:3].round(3)}")
print(f"  Min z (grasp): {ep_states[:,2].min():.3f}")
print(f"  Grip min: {ep_states[:,6].min():.4f}  max: {ep_states[:,6].max():.4f}")
print(f"  Grip goes closed at frame: ", end="")
close_frames = np.where(ep_states[:,6] < 0.005)[0]
print(close_frames[:5] if len(close_frames) else "NEVER")

# Predict from episode 74 start
print(f"\n  Model prediction from ep74 start (t=OBS_HORIZON):")
t_rel = OBS_HORIZON
obs_hist = [ep_states[t_rel - OBS_HORIZON + 1 + i] for i in range(OBS_HORIZON)]
pred = predict(obs_hist)
print(f"  Predicted first 5 positions:")
for i in range(5):
    gt = ep_states[t_rel + i, :3]
    pr = pred[i, :3]
    err = np.linalg.norm(pr - gt)
    print(f"    step {i}: pred={pr.round(3)}  gt={gt.round(3)}  err={err:.4f}m")
print(f"  Predicted grip sequence: {pred[:,6].round(4)}")
print(f"  GT grip sequence:        {ep_states[t_rel:t_rel+ACTION_HORIZON,6].round(4)}")

print("\n✅ Evaluation complete!")
print("If pos_err < 0.05m → inference should work")
print("If grip direction acc > 80% → grasping should work")