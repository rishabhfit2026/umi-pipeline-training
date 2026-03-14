"""
Inference on REAL UMI trained diffusion policy
Rolls out model on held-out episodes and visualizes predicted vs actual trajectory.

conda activate maniskill2  
python infer_umi_diffusion.py
"""

import math, numpy as np, zarr, torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

# ── Config ────────────────────────────────────────────────────────
ZARR_PATH = '/home/rishabh/Downloads/umi-pipeline-training/replay_buffer.zarr'
SAVE_DIR  = './checkpoints_umi'
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'

OBS_HORIZON    = 4
ACTION_HORIZON = 16
ACTION_DIM     = 7
STATE_DIM      = 7
GOAL_DIM       = 12   # demo_start_pose(6) + demo_end_pose(6)
OBS_IN         = STATE_DIM * OBS_HORIZON + GOAL_DIM  # 40

# ── Model (must match training arch exactly) ──────────────────────
class Normalizer:
    def normalize(self, x):
        return 2.0*(x - self.min.to(x.device))/self.scale.to(x.device) - 1.0
    def denormalize(self, x):
        return (x+1.0)/2.0*self.scale.to(x.device) + self.min.to(x.device)
    @classmethod
    def load(cls, path):
        n=cls.__new__(cls); d=torch.load(path, map_location='cpu')
        n.min,n.max,n.scale=d['min'],d['max'],d['scale']; return n

class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim): super().__init__(); self.dim=dim
    def forward(self, t):
        half=self.dim//2
        emb=math.log(10000)/(half-1)
        emb=torch.exp(torch.arange(half,device=t.device)*-emb)
        emb=t.float()[:,None]*emb[None,:]
        return torch.cat([emb.sin(),emb.cos()],dim=-1)

class ResBlock(torch.nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.net  = torch.nn.Sequential(torch.nn.Linear(dim,dim),torch.nn.Mish(),torch.nn.Linear(dim,dim))
        self.cond = torch.nn.Linear(cond_dim, dim*2)
        self.norm = torch.nn.LayerNorm(dim)
    def forward(self, x, cond):
        scale,bias = self.cond(cond).chunk(2,dim=-1)
        return x + self.net(self.norm(x)*(scale+1)+bias)

class UMIDiffusionNet(torch.nn.Module):
    def __init__(self, hidden=512, depth=8):
        super().__init__()
        flat_act = ACTION_DIM * ACTION_HORIZON
        cond_dim = 512
        self.obs_emb   = torch.nn.Sequential(
            torch.nn.Linear(OBS_IN,256),torch.nn.Mish(),
            torch.nn.Linear(256,256),torch.nn.Mish(),torch.nn.Linear(256,256))
        self.time_emb  = torch.nn.Sequential(
            SinusoidalPosEmb(128),torch.nn.Linear(128,256),
            torch.nn.Mish(),torch.nn.Linear(256,256))
        self.cond_proj = torch.nn.Sequential(
            torch.nn.Linear(512,cond_dim),torch.nn.Mish(),torch.nn.Linear(cond_dim,cond_dim))
        self.in_proj   = torch.nn.Linear(flat_act, hidden)
        self.blocks    = torch.nn.ModuleList([ResBlock(hidden,cond_dim) for _ in range(depth)])
        self.out_proj  = torch.nn.Sequential(torch.nn.LayerNorm(hidden),torch.nn.Linear(hidden,flat_act))
    def forward(self, noisy, timestep, obs_goal):
        B=noisy.shape[0]; x=noisy.reshape(B,-1)
        cond=self.cond_proj(torch.cat([self.obs_emb(obs_goal),self.time_emb(timestep)],dim=-1))
        x=self.in_proj(x)
        for blk in self.blocks: x=blk(x,cond)
        return self.out_proj(x).reshape(B,ACTION_HORIZON,ACTION_DIM)

# ── Load model + normalizers ──────────────────────────────────────
print(f"Device: {DEVICE}")
ckpt = torch.load(f'{SAVE_DIR}/best_model.pt', map_location=DEVICE)
model = UMIDiffusionNet().to(DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()
obs_norm  = Normalizer.load(f'{SAVE_DIR}/obs_normalizer.pt')
act_norm  = Normalizer.load(f'{SAVE_DIR}/act_normalizer.pt')
goal_norm = Normalizer.load(f'{SAVE_DIR}/goal_normalizer.pt')

sched = DDIMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2',
                      clip_sample=True, prediction_type='epsilon')
sched.set_timesteps(16)
print(f"✅ Model loaded — epoch {ckpt['epoch']}, loss={ckpt['loss']:.5f}")

# ── Load data ─────────────────────────────────────────────────────
z            = zarr.open(ZARR_PATH, 'r')
pos          = z['data']['robot0_eef_pos'][:]
rot          = z['data']['robot0_eef_rot_axis_angle'][:]
grip         = z['data']['robot0_gripper_width'][:]
start_poses  = z['data']['robot0_demo_start_pose'][:]
end_poses    = z['data']['robot0_demo_end_pose'][:]
ends         = z['meta']['episode_ends'][:]
starts       = np.concatenate([[0], ends[:-1]])
states       = np.concatenate([pos, rot, grip], axis=-1).astype(np.float32)

def get_goal(ep_i, s, e):
    if start_poses.shape[0] == len(ends):
        return np.concatenate([start_poses[ep_i], end_poses[ep_i]]).astype(np.float32)
    return np.concatenate([start_poses[s], end_poses[e-1]]).astype(np.float32)

# ── Single episode rollout ────────────────────────────────────────
def rollout_episode(ep_idx, exec_h=8):
    s, e     = int(starts[ep_idx]), int(ends[ep_idx])
    goal_raw = get_goal(ep_idx, s, e)
    goal_t   = goal_norm.normalize(torch.tensor(goal_raw, dtype=torch.float32))

    pred_xyz   = []
    pred_grip  = []
    actual_xyz = pos[s:e]
    actual_grip= grip[s:e, 0]

    t = s + OBS_HORIZON - 1
    while t < e - ACTION_HORIZON:
        obs_parts = [obs_norm.normalize(
                         torch.tensor(states[t-OBS_HORIZON+1+i], dtype=torch.float32))
                     for i in range(OBS_HORIZON)]
        obs_goal = torch.cat(obs_parts + [goal_t]).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            noisy = torch.randn(1, ACTION_HORIZON, ACTION_DIM, device=DEVICE)
            for ts in sched.timesteps:
                tst   = torch.tensor([ts], device=DEVICE).long()
                noisy = sched.step(model(noisy, tst, obs_goal), ts, noisy).prev_sample
            pred = act_norm.denormalize(noisy[0]).cpu().numpy()

        for i in range(exec_h):
            pred_xyz.append(pred[i, :3])
            pred_grip.append(pred[i, 6])
        t += exec_h

    pred_xyz  = np.array(pred_xyz)
    pred_grip = np.array(pred_grip)
    n = min(len(pred_xyz), len(actual_xyz))
    return actual_xyz[:n], pred_xyz[:n], actual_grip[:n], pred_grip[:n], goal_raw

# ── Metrics ───────────────────────────────────────────────────────
def compute_metrics(actual, pred):
    err = np.linalg.norm(actual - pred, axis=-1)
    return {'mean': err.mean(), 'max': err.max(),
            'final': err[-1], 'under2cm': (err < 0.02).mean()}

# ═══════════════════════════════════════════════════════════════════
# PLOT 1 — Trajectory comparison per episode
# ═══════════════════════════════════════════════════════════════════
def plot_trajectory_comparison(n_episodes=6):
    n_test_start = int(len(ends) * 0.8)
    test_eps = list(range(n_test_start, min(n_test_start+n_episodes, len(ends))))

    print(f"\n{'='*55}")
    print(f" TRAJECTORY COMPARISON ({len(test_eps)} held-out episodes)")
    print(f"{'='*55}")

    fig = plt.figure(figsize=(22, 5*len(test_eps)))
    gs  = gridspec.GridSpec(len(test_eps), 5, figure=fig,
                             hspace=0.45, wspace=0.35)
    fig.suptitle(f'UMI Diffusion Policy — Predicted vs Actual\n'
                 f'(epoch={ckpt["epoch"]}, loss={ckpt["loss"]:.5f})',
                 fontsize=13, y=1.01)

    metrics_all = []
    for row, ep_idx in enumerate(test_eps):
        actual, pred, act_g, pred_g, goal = rollout_episode(ep_idx)
        m = compute_metrics(actual, pred)
        metrics_all.append(m)

        print(f"  Ep {ep_idx+1:3d} | frames={len(actual):4d} | "
              f"mean={m['mean']*100:.1f}cm | max={m['max']*100:.1f}cm | "
              f"final={m['final']*100:.1f}cm | "
              f"<2cm={m['under2cm']*100:.0f}%")

        t = np.arange(len(actual))
        axis_labels = ['X (m)', 'Y (m)', 'Z (m)']
        colors      = ['#e74c3c', '#2ecc71', '#3498db']

        for col in range(3):
            ax = fig.add_subplot(gs[row, col])
            ax.plot(t, actual[:, col], color=colors[col], lw=2,
                    label='Actual', alpha=0.9)
            ax.plot(t, pred[:, col],   color=colors[col], lw=1.5,
                    label='Predicted', alpha=0.75, linestyle='--')
            ax.fill_between(t,
                actual[:, col], pred[:, col],
                alpha=0.12, color=colors[col])
            ax.set_title(f'Ep{ep_idx+1} {axis_labels[col]}  '
                         f'err={m["mean"]*100:.1f}cm', fontsize=9)
            ax.set_xlabel('Frame', fontsize=8)
            ax.legend(fontsize=7); ax.grid(alpha=0.3)

        # Gripper
        ax_g = fig.add_subplot(gs[row, 3])
        ax_g.plot(t, act_g,  lw=2,   label='Actual',    color='#9b59b6', alpha=0.9)
        ax_g.plot(t, pred_g, lw=1.5, label='Predicted', color='#9b59b6',
                  alpha=0.75, linestyle='--')
        ax_g.axhline(y=0.02, color='gray', linestyle=':', alpha=0.6, label='Open thresh')
        ax_g.set_title(f'Ep{ep_idx+1} Gripper width', fontsize=9)
        ax_g.set_xlabel('Frame', fontsize=8)
        ax_g.legend(fontsize=7); ax_g.grid(alpha=0.3)

        # 3D trajectory
        ax3 = fig.add_subplot(gs[row, 4], projection='3d')
        ax3.plot(actual[:,0], actual[:,1], actual[:,2],
                 'b-', lw=2, label='Actual', alpha=0.85)
        ax3.plot(pred[:,0],   pred[:,1],   pred[:,2],
                 'r--', lw=1.5, label='Predicted', alpha=0.75)
        ax3.scatter(*actual[0],  c='lime',  s=60, zorder=5, label='Start')
        ax3.scatter(*actual[-1], c='red',   s=60, zorder=5, label='End')
        # plot goal end position
        ax3.scatter(goal[3], goal[4], goal[5],
                    c='orange', s=80, marker='*', zorder=5, label='Goal')
        ax3.set_title(f'Ep{ep_idx+1} 3D', fontsize=9)
        ax3.set_xlabel('X', fontsize=7); ax3.set_ylabel('Y', fontsize=7)
        ax3.set_zlabel('Z', fontsize=7)
        ax3.legend(fontsize=6)

    overall_mean = np.mean([m['mean'] for m in metrics_all]) * 100
    overall_2cm  = np.mean([m['under2cm'] for m in metrics_all]) * 100
    print(f"\n  Overall mean error : {overall_mean:.1f}cm")
    print(f"  Frames within 2cm  : {overall_2cm:.0f}%")

    plt.savefig(f'{SAVE_DIR}/trajectory_comparison.png',
                dpi=130, bbox_inches='tight')
    print(f"\n✅ Saved → {SAVE_DIR}/trajectory_comparison.png")
    plt.show()

# ═══════════════════════════════════════════════════════════════════
# PLOT 2 — Error analysis across all test episodes
# ═══════════════════════════════════════════════════════════════════
def plot_error_analysis(n_episodes=20):
    n_test_start = int(len(ends) * 0.8)
    test_eps = list(range(n_test_start, min(n_test_start+n_episodes, len(ends))))

    print(f"\nRunning error analysis on {len(test_eps)} episodes...")

    all_errors  = []
    mean_errors = []
    for ep_idx in test_eps:
        actual, pred, _, _, _ = rollout_episode(ep_idx)
        err = np.linalg.norm(actual - pred, axis=-1)
        all_errors.append(err)
        mean_errors.append(err.mean())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Error Analysis — UMI Diffusion Policy', fontsize=13)

    # Per-episode error curves
    ax = axes[0, 0]
    for i, err in enumerate(all_errors):
        ax.plot(err * 100, alpha=0.4, lw=1)
    ax.plot(np.mean([e*100 for e in all_errors], axis=0),
            'r-', lw=2.5, label='Mean across eps')
    ax.axhline(y=2.0, color='k', linestyle='--', alpha=0.6, label='2cm')
    ax.axhline(y=5.0, color='r', linestyle='--', alpha=0.6, label='5cm')
    ax.set_title('Per-frame error per episode')
    ax.set_xlabel('Frame'); ax.set_ylabel('Error (cm)')
    ax.legend(); ax.grid(alpha=0.3)

    # Mean error per episode bar
    ax = axes[0, 1]
    colors = ['#2ecc71' if e*100 < 2 else '#f39c12' if e*100 < 5 else '#e74c3c'
              for e in mean_errors]
    ax.bar(range(len(mean_errors)), [e*100 for e in mean_errors],
           color=colors, alpha=0.85)
    ax.axhline(y=np.mean(mean_errors)*100, color='navy', linestyle='--',
               lw=2, label=f'Mean={np.mean(mean_errors)*100:.1f}cm')
    ax.axhline(y=2.0, color='green', linestyle=':', lw=1.5, label='2cm (good)')
    ax.axhline(y=5.0, color='red',   linestyle=':', lw=1.5, label='5cm (bad)')
    ax.set_title('Mean error per episode')
    ax.set_xlabel('Episode index'); ax.set_ylabel('Mean error (cm)')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    # Error histogram
    ax = axes[1, 0]
    all_flat = np.concatenate(all_errors) * 100
    ax.hist(all_flat, bins=60, color='steelblue', alpha=0.8, edgecolor='white')
    ax.axvline(x=2.0, color='green', lw=2, label=f'2cm ({(all_flat<2).mean()*100:.0f}% frames)')
    ax.axvline(x=5.0, color='red',   lw=2, label=f'5cm ({(all_flat<5).mean()*100:.0f}% frames)')
    ax.axvline(x=all_flat.mean(), color='orange', lw=2,
               linestyle='--', label=f'Mean={all_flat.mean():.1f}cm')
    ax.set_title('Error distribution (all frames)')
    ax.set_xlabel('Error (cm)'); ax.set_ylabel('Frame count')
    ax.legend(); ax.grid(alpha=0.3)

    # Error vs episode length
    ax = axes[1, 1]
    ep_lens = [int(ends[ep])-int(starts[ep]) for ep in test_eps[:len(mean_errors)]]
    sc = ax.scatter(ep_lens, [e*100 for e in mean_errors],
                    c=[e*100 for e in mean_errors],
                    cmap='RdYlGn_r', s=80, alpha=0.85, vmin=0, vmax=8)
    plt.colorbar(sc, ax=ax, label='Mean error (cm)')
    ax.set_title('Error vs episode length')
    ax.set_xlabel('Episode length (frames)'); ax.set_ylabel('Mean error (cm)')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/error_analysis.png', dpi=130, bbox_inches='tight')
    print(f"✅ Saved → {SAVE_DIR}/error_analysis.png")
    plt.show()

# ═══════════════════════════════════════════════════════════════════
# PLOT 3 — Animated robot arm movement (2D top-down view)
# ═══════════════════════════════════════════════════════════════════
def animate_episode(ep_idx=None, save_gif=True):
    import matplotlib.animation as animation

    if ep_idx is None:
        ep_idx = int(len(ends) * 0.8)  # first test episode

    actual, pred, act_g, pred_g, goal = rollout_episode(ep_idx, exec_h=4)
    print(f"\nAnimating episode {ep_idx+1} ({len(actual)} frames)...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Episode {ep_idx+1} — Robot EEF Movement\n'
                 f'Blue=Actual  Red=Predicted  ★=Goal', fontsize=12)

    # Top-down XY view
    ax1 = axes[0]
    ax1.set_xlim(actual[:,0].min()-0.05, actual[:,0].max()+0.05)
    ax1.set_ylim(actual[:,1].min()-0.05, actual[:,1].max()+0.05)
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
    ax1.set_title('Top-down view (XY)')
    ax1.grid(alpha=0.3)
    ax1.plot(actual[:,0], actual[:,1], 'b-', alpha=0.2, lw=1)
    ax1.plot(pred[:,0],   pred[:,1],   'r-', alpha=0.2, lw=1)
    ax1.scatter(goal[3], goal[4], c='orange', s=150, marker='*',
                zorder=5, label='Goal end')
    ax1.scatter(actual[0,0], actual[0,1], c='lime', s=100,
                zorder=5, label='Start')
    ax1.legend(fontsize=8)

    actual_dot, = ax1.plot([], [], 'bo', ms=10, zorder=6)
    pred_dot,   = ax1.plot([], [], 'ro', ms=8,  zorder=6, alpha=0.8)
    actual_trail, = ax1.plot([], [], 'b-', lw=2, alpha=0.7)
    pred_trail,   = ax1.plot([], [], 'r--', lw=1.5, alpha=0.6)

    # XZ side view
    ax2 = axes[1]
    ax2.set_xlim(actual[:,0].min()-0.05, actual[:,0].max()+0.05)
    ax2.set_ylim(actual[:,2].min()-0.02, actual[:,2].max()+0.02)
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Z (m)')
    ax2.set_title('Side view (XZ) — height')
    ax2.grid(alpha=0.3)
    ax2.plot(actual[:,0], actual[:,2], 'b-', alpha=0.2, lw=1)
    ax2.plot(pred[:,0],   pred[:,2],   'r-', alpha=0.2, lw=1)

    actual_dot2, = ax2.plot([], [], 'bo', ms=10, zorder=6)
    pred_dot2,   = ax2.plot([], [], 'ro', ms=8,  zorder=6, alpha=0.8)
    trail_len = 30
    err_text  = ax1.text(0.02, 0.97, '', transform=ax1.transAxes,
                          fontsize=9, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    def update(frame):
        lo = max(0, frame-trail_len)
        actual_dot.set_data([actual[frame,0]], [actual[frame,1]])
        pred_dot.set_data([pred[frame,0]],     [pred[frame,1]])
        actual_trail.set_data(actual[lo:frame+1,0], actual[lo:frame+1,1])
        pred_trail.set_data(pred[lo:frame+1,0],     pred[lo:frame+1,1])
        actual_dot2.set_data([actual[frame,0]], [actual[frame,2]])
        pred_dot2.set_data([pred[frame,0]],     [pred[frame,2]])
        err = np.linalg.norm(actual[frame]-pred[frame]) * 100
        grip_state = "CLOSED" if act_g[frame] < 0.02 else "OPEN"
        err_text.set_text(f'Frame: {frame}/{len(actual)}\n'
                          f'Error: {err:.1f}cm\n'
                          f'Gripper: {grip_state}\n'
                          f'Z: {actual[frame,2]:.3f}m')
        return actual_dot, pred_dot, actual_trail, pred_trail, \
               actual_dot2, pred_dot2, err_text

    step   = max(1, len(actual)//200)  # cap at ~200 frames for speed
    frames = range(0, len(actual), step)
    ani    = animation.FuncAnimation(fig, update, frames=frames,
                                     interval=50, blit=True)
    if save_gif:
        out = f'{SAVE_DIR}/episode_{ep_idx+1}_movement.gif'
        ani.save(out, writer='pillow', fps=20)
        print(f"✅ Saved → {out}")
    plt.tight_layout()
    plt.show()

# ── Run all ───────────────────────────────────────────────────────
if __name__ == '__main__':
    plot_trajectory_comparison(n_episodes=6)
    plot_error_analysis(n_episodes=20)
    animate_episode(ep_idx=None, save_gif=True)