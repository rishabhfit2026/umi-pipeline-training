"""
Dataset Visualization — How well did the robot pick and place?
Run this locally:
  conda activate maniskill2
  python visualize_dataset.py
"""

import zarr, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

ZARR = '/home/rishabh/Downloads/umi-pipeline-training/outputs/sim_replay_buffer3.zarr'

# ── Load data ─────────────────────────────────────────────────────
z     = zarr.open(ZARR, 'r')
pos   = z['data']['robot0_eef_pos'][:]           # (72000, 3)
rot   = z['data']['robot0_eef_rot_axis_angle'][:]
grip  = z['data']['robot0_gripper_width'][:][:,0] # (72000,)
ends  = z['meta']['episode_ends'][:]
starts= np.concatenate([[0], ends[:-1]])
N_EP  = len(ends)

print(f"Dataset: {N_EP} episodes, {len(pos)} frames")
print(f"EEF x: {pos[:,0].min():.3f} → {pos[:,0].max():.3f}")
print(f"EEF y: {pos[:,1].min():.3f} → {pos[:,1].max():.3f}")
print(f"EEF z: {pos[:,2].min():.3f} → {pos[:,2].max():.3f}")
print(f"Grip:  {grip.min():.4f} → {grip.max():.4f}")

# ── Reconstruct episode goals (same RNG as training) ──────────────
TX, TY = 0.250, 0.000
rng = np.random.default_rng(42)
ep_markers, ep_boxes = [], []
for _ in range(N_EP):
    mx=TX+rng.uniform(-0.05,0.05); my=TY+rng.uniform(-0.10,0.10)
    bx=TX+rng.uniform(-0.05,0.05); by=TY+rng.uniform( 0.05,0.15)
    while abs(mx-bx)<0.04 and abs(my-by)<0.04:
        bx=TX+rng.uniform(-0.05,0.05); by=TY+rng.uniform(0.05,0.15)
    ep_markers.append([mx,my])
    ep_boxes.append([bx,by])

# ── Analyze every episode ─────────────────────────────────────────
results = []
for i,(s,e) in enumerate(zip(starts,ends)):
    g = grip[s:e]; p = pos[s:e]
    
    # Find grasp and release events
    grasped=False; grasp_f=None; release_f=None; releases=0
    for f in range(len(g)):
        if not grasped and g[f]<0.005:
            grasped=True; grasp_f=f
        if grasped and g[f]>0.015:
            releases+=1; release_f=f; grasped=False
    
    # Estimate if pick+place succeeded
    # "picked" = gripper closed below z=0.15
    picked = grasp_f is not None and p[grasp_f,2] < 0.15 if grasp_f else False
    # "placed" = released near box xy
    placed = False
    if release_f is not None:
        ee_at_release = p[release_f,:2]
        box = np.array(ep_boxes[i])
        dist = np.linalg.norm(ee_at_release - box)
        placed = dist < 0.12

    results.append({
        'ep': i+1,
        'grasp_frame': grasp_f,
        'release_frame': release_f,
        'closed_frames': int((g<0.005).sum()),
        'picked': picked,
        'placed': placed,
        'success': picked and placed,
        'z_min': float(p[:,2].min()),
        'z_max': float(p[:,2].max()),
        'x_range': float(p[:,0].max()-p[:,0].min()),
        'y_range': float(p[:,1].max()-p[:,1].min()),
    })

n_pick   = sum(r['picked']   for r in results)
n_place  = sum(r['placed']   for r in results)
n_success= sum(r['success']  for r in results)
n_grasp  = sum(r['grasp_frame'] is not None for r in results)
print(f"\n{'='*50}")
print(f"Gripper closed:  {n_grasp}/200  ({100*n_grasp/200:.0f}%)")
print(f"Picked (low z):  {n_pick}/200   ({100*n_pick/200:.0f}%)")
print(f"Placed (near box):{n_place}/200  ({100*n_place/200:.0f}%)")
print(f"Full success:    {n_success}/200  ({100*n_success/200:.0f}%)")
print(f"{'='*50}")

# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — Overview dashboard
# ══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#1a1a2e')
gs  = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.4)

# ── Plot 1: Success rate pie ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0,0])
ax1.set_facecolor('#16213e')
sizes  = [n_success, n_pick-n_success, n_grasp-n_pick, 200-n_grasp]
labels = [f'Full success\n{n_success}', f'Picked not placed\n{n_pick-n_success}',
          f'Grasped not picked\n{n_grasp-n_pick}', f'No grasp\n{200-n_grasp}']
colors = ['#00ff88','#ffaa00','#ff6644','#444466']
non_zero = [(s,l,c) for s,l,c in zip(sizes,labels,colors) if s>0]
if non_zero:
    s2,l2,c2 = zip(*non_zero)
    wedges,texts,autotexts = ax1.pie(s2, labels=l2, colors=c2,
                                      autopct='%1.0f%%', startangle=90,
                                      textprops={'color':'white','fontsize':7})
ax1.set_title('Episode Outcomes\n(200 total)', color='white', fontsize=9, pad=8)

# ── Plot 2: Z trajectory for 10 sample episodes ───────────────────
ax2 = fig.add_subplot(gs[0,1:3])
ax2.set_facecolor('#16213e')
sample_eps = np.linspace(0, N_EP-1, 12, dtype=int)
cmap = plt.cm.plasma
for k,i in enumerate(sample_eps):
    s,e = int(starts[i]), int(ends[i])
    z_traj = pos[s:e, 2]
    t = np.linspace(0,1,len(z_traj))
    color = cmap(k/len(sample_eps))
    ax2.plot(t, z_traj, color=color, alpha=0.7, linewidth=1.2, label=f'ep{i+1}')
ax2.axhline(0.076, color='#ff4444', linestyle='--', linewidth=1, alpha=0.8, label='marker z')
ax2.axhline(0.150, color='#ffaa00', linestyle='--', linewidth=1, alpha=0.8, label='grasp z limit')
ax2.set_xlabel('Normalized time', color='white', fontsize=8)
ax2.set_ylabel('EEF Z (m)', color='white', fontsize=8)
ax2.set_title('EEF Height Over Time (12 episodes)', color='white', fontsize=9)
ax2.tick_params(colors='white', labelsize=7)
ax2.spines[:].set_color('#444')
ax2.legend(fontsize=5, ncol=4, loc='upper right',
           facecolor='#1a1a2e', labelcolor='white')

# ── Plot 3: Gripper width over time ───────────────────────────────
ax3 = fig.add_subplot(gs[0,3])
ax3.set_facecolor('#16213e')
for k,i in enumerate(sample_eps[:6]):
    s,e = int(starts[i]), int(ends[i])
    g_traj = grip[s:e]
    t = np.linspace(0,1,len(g_traj))
    ax3.plot(t, g_traj, color=cmap(k/6), alpha=0.8, linewidth=1.2)
ax3.axhline(0.005, color='#ff4444', linestyle='--', linewidth=1, alpha=0.8, label='closed')
ax3.set_xlabel('Norm time', color='white', fontsize=8)
ax3.set_ylabel('Grip width (m)', color='white', fontsize=8)
ax3.set_title('Gripper Width\n(6 episodes)', color='white', fontsize=9)
ax3.tick_params(colors='white', labelsize=7)
ax3.spines[:].set_color('#444')

# ── Plot 4: XY trajectories top-down ─────────────────────────────
ax4 = fig.add_subplot(gs[1,:2])
ax4.set_facecolor('#16213e')
for k,i in enumerate(sample_eps):
    s,e = int(starts[i]), int(ends[i])
    x_t = pos[s:e, 0]; y_t = pos[s:e, 1]
    color = cmap(k/len(sample_eps))
    ax4.plot(x_t, y_t, color=color, alpha=0.5, linewidth=0.9)
    ax4.plot(x_t[0],  y_t[0],  'o', color=color, markersize=4)
    ax4.plot(x_t[-1], y_t[-1], 's', color=color, markersize=4)
    # Draw marker and box
    mx,my = ep_markers[i]; bx,by = ep_boxes[i]
    ax4.plot(mx, my, 'r^', markersize=5, alpha=0.6)
    ax4.plot(bx, by, 'gs', markersize=5, alpha=0.6)
ax4.plot([],[], 'r^', label='Marker'); ax4.plot([],[], 'gs', label='Box')
ax4.plot([],[], 'o', color='gray', label='Start'); ax4.plot([],[], 's', color='gray', label='End')
ax4.set_xlabel('X (m)', color='white', fontsize=8)
ax4.set_ylabel('Y (m)', color='white', fontsize=8)
ax4.set_title('EEF XY Trajectories Top-Down\n(12 episodes)', color='white', fontsize=9)
ax4.tick_params(colors='white', labelsize=7)
ax4.spines[:].set_color('#444')
ax4.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white')
ax4.set_aspect('equal')

# ── Plot 5: Grasp timing distribution ────────────────────────────
ax5 = fig.add_subplot(gs[1,2])
ax5.set_facecolor('#16213e')
grasp_frames = [r['grasp_frame'] for r in results if r['grasp_frame'] is not None]
if grasp_frames:
    ax5.hist(grasp_frames, bins=20, color='#00ff88', edgecolor='#16213e', alpha=0.85)
ax5.set_xlabel('Frame #', color='white', fontsize=8)
ax5.set_ylabel('Count', color='white', fontsize=8)
ax5.set_title(f'Grasp Timing\n({len(grasp_frames)} episodes)', color='white', fontsize=9)
ax5.tick_params(colors='white', labelsize=7)
ax5.spines[:].set_color('#444')

# ── Plot 6: Z min per episode (how low did EE go) ─────────────────
ax6 = fig.add_subplot(gs[1,3])
ax6.set_facecolor('#16213e')
z_mins = [r['z_min'] for r in results]
colors_bar = ['#00ff88' if r['picked'] else '#ff4444' for r in results]
ax6.bar(range(N_EP), z_mins, color=colors_bar, alpha=0.7, width=1.0)
ax6.axhline(0.076, color='#ffff00', linestyle='--', linewidth=1, label='marker z=0.076')
ax6.axhline(0.150, color='#ff8800', linestyle='--', linewidth=1, label='z<0.15=pick')
ax6.set_xlabel('Episode', color='white', fontsize=8)
ax6.set_ylabel('Min EEF Z (m)', color='white', fontsize=8)
ax6.set_title('How Low EE Went\n🟢=picked  🔴=missed', color='white', fontsize=9)
ax6.tick_params(colors='white', labelsize=7)
ax6.spines[:].set_color('#444')
ax6.legend(fontsize=6, facecolor='#1a1a2e', labelcolor='white')

# ── Plot 7: Detailed episode 74 ───────────────────────────────────
ax7 = fig.add_subplot(gs[2,:2])
ax7.set_facecolor('#16213e')
ep74 = 73
s74,e74 = int(starts[ep74]),int(ends[ep74])
frames = np.arange(e74-s74)
ax7_twin = ax7.twinx()
ax7.plot(frames, pos[s74:e74,0], color='#ff6644', linewidth=1.5, label='x')
ax7.plot(frames, pos[s74:e74,1], color='#44aaff', linewidth=1.5, label='y')
ax7.plot(frames, pos[s74:e74,2], color='#44ff88', linewidth=1.5, label='z')
ax7_twin.plot(frames, grip[s74:e74], color='#ffff00', linewidth=2, linestyle='--', label='grip')
r74 = results[ep74]
if r74['grasp_frame']:
    ax7.axvline(r74['grasp_frame'], color='white', linestyle=':', linewidth=1.5, label=f'grasp@{r74["grasp_frame"]}')
if r74['release_frame']:
    ax7.axvline(r74['release_frame'], color='#ff88ff', linestyle=':', linewidth=1.5, label=f'release@{r74["release_frame"]}')
ax7.axhline(0.076, color='#ff4444', linestyle='--', linewidth=0.8, alpha=0.5)
ax7.set_xlabel('Frame', color='white', fontsize=8)
ax7.set_ylabel('Position (m)', color='white', fontsize=8)
ax7_twin.set_ylabel('Grip width', color='#ffff00', fontsize=8)
ax7_twin.tick_params(colors='#ffff00', labelsize=7)
ax7.set_title(f'Episode 74 Detailed  marker=({ep_markers[ep74][0]:.3f},{ep_markers[ep74][1]:.3f})'
              f'  box=({ep_boxes[ep74][0]:.3f},{ep_boxes[ep74][1]:.3f})'
              f'  {"✅ SUCCESS" if results[ep74]["success"] else "❌ FAILED"}',
              color='white', fontsize=9)
ax7.tick_params(colors='white', labelsize=7)
ax7.spines[:].set_color('#444')
lines1,labs1=ax7.get_legend_handles_labels()
lines2,labs2=ax7_twin.get_legend_handles_labels()
ax7.legend(lines1+lines2, labs1+labs2, fontsize=7, facecolor='#1a1a2e', labelcolor='white', ncol=3)

# ── Plot 8: XY range per episode (how much did EE move) ───────────
ax8 = fig.add_subplot(gs[2,2])
ax8.set_facecolor('#16213e')
x_ranges = [r['x_range'] for r in results]
y_ranges = [r['y_range'] for r in results]
ax8.scatter(x_ranges, y_ranges,
            c=['#00ff88' if r['success'] else '#ff4444' for r in results],
            s=20, alpha=0.7)
ax8.set_xlabel('X range (m)', color='white', fontsize=8)
ax8.set_ylabel('Y range (m)', color='white', fontsize=8)
ax8.set_title('EEF Movement Range\n🟢=success  🔴=fail', color='white', fontsize=9)
ax8.tick_params(colors='white', labelsize=7)
ax8.spines[:].set_color('#444')

# ── Plot 9: Summary text ──────────────────────────────────────────
ax9 = fig.add_subplot(gs[2,3])
ax9.set_facecolor('#16213e')
ax9.axis('off')
summary = (
    f"DATASET SUMMARY\n"
    f"{'─'*22}\n"
    f"Total episodes:  200\n"
    f"Frames/episode:  360\n"
    f"Total frames:    72,000\n\n"
    f"PICK & PLACE STATS\n"
    f"{'─'*22}\n"
    f"Gripper closed:  {n_grasp}/200 ({100*n_grasp/200:.0f}%)\n"
    f"Picked (z<0.15): {n_pick}/200 ({100*n_pick/200:.0f}%)\n"
    f"Placed @ box:    {n_place}/200 ({100*n_place/200:.0f}%)\n"
    f"Full success:    {n_success}/200 ({100*n_success/200:.0f}%)\n\n"
    f"EEF WORKSPACE\n"
    f"{'─'*22}\n"
    f"x: 0.093 → 0.473 m\n"
    f"y: -0.222 → 0.284 m\n"
    f"z: 0.009 → 0.783 m\n\n"
    f"Avg grasp frame: {np.mean(grasp_frames):.0f}\n"
    f"(out of 360 total)"
)
ax9.text(0.05, 0.95, summary, transform=ax9.transAxes,
         fontsize=8.5, verticalalignment='top', fontfamily='monospace',
         color='#00ffcc', bbox=dict(boxstyle='round', facecolor='#0d1117', alpha=0.8))

fig.suptitle('Robot Pick & Place Dataset Analysis — sim_replay_buffer3.zarr',
             color='white', fontsize=13, fontweight='bold', y=0.98)

out = '/home/rishabh/Downloads/umi-pipeline-training/dataset_visualization.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print(f"\n✅ Saved → {out}")
plt.close()

# ── Figure 2: Episode-by-episode success grid ─────────────────────
fig2,ax = plt.subplots(figsize=(16,6))
fig2.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#16213e')
grid = np.zeros((10,20))
color_grid = np.zeros((10,20,3))
for i,r in enumerate(results):
    row,col = i//20, i%20
    if r['success']:       color_grid[row,col] = [0.0, 1.0, 0.5]
    elif r['picked']:      color_grid[row,col] = [1.0, 0.7, 0.0]
    elif r['grasp_frame']: color_grid[row,col] = [1.0, 0.4, 0.1]
    else:                  color_grid[row,col] = [0.2, 0.2, 0.3]
ax.imshow(color_grid, aspect='auto', interpolation='nearest')
for i,r in enumerate(results):
    row,col = i//20, i%20
    sym = '✓' if r['success'] else ('P' if r['picked'] else ('G' if r['grasp_frame'] else '✗'))
    ax.text(col, row, sym, ha='center', va='center', fontsize=7,
            color='white', fontweight='bold')
ax.set_xticks(range(20)); ax.set_xticklabels([str(i+1) for i in range(20)], fontsize=7, color='white')
ax.set_yticks(range(10));  ax.set_yticklabels([f'ep{i*20+1}-{i*20+20}' for i in range(10)], fontsize=7, color='white')
ax.set_title('Episode Grid  ✓=full success  P=picked  G=grasped  ✗=nothing',
             color='white', fontsize=11, pad=10)
from matplotlib.patches import Patch
legend_els = [Patch(facecolor=[0.0,1.0,0.5], label='Full pick+place ✓'),
              Patch(facecolor=[1.0,0.7,0.0], label='Picked not placed P'),
              Patch(facecolor=[1.0,0.4,0.1], label='Grasped only G'),
              Patch(facecolor=[0.2,0.2,0.3], label='Nothing ✗')]
ax.legend(handles=legend_els, loc='upper right', bbox_to_anchor=(1,1.18),
          ncol=4, fontsize=8, facecolor='#1a1a2e', labelcolor='white')
out2 = '/home/rishabh/Downloads/umi-pipeline-training/dataset_grid.png'
plt.savefig(out2, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print(f"✅ Saved → {out2}")
plt.close()
print("\nDone! Open the two PNG files to see the visualizations.")