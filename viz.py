import zarr, numpy as np, cv2, os

ZARR = '/home/rishabh/Downloads/umi-pipeline-training/outputs/sim_replay_buffer2.zarr'
OUT  = '/home/rishabh/Downloads/umi-pipeline-training/outputs/viz_frames'
os.makedirs(OUT, exist_ok=True)

z    = zarr.open(ZARR, 'r')
imgs = z['data']['camera0_rgb']
pos  = z['data']['robot0_eef_pos']
grip = z['data']['robot0_gripper_width']
ends = z['meta']['episode_ends']

print(f"Episodes: {len(ends)}  Frames: {len(imgs)}")

# Save 12 frames from episode 0
ep0_end = ends[0]
indices = [int(ep0_end * f) for f in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]]
tiles = []
for idx in indices:
    img = imgs[idx].copy()
    p, g = pos[idx], grip[idx,0]
    cv2.putText(img, f"z={p[2]:.3f} g={g:.3f}", (3,15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
    tiles.append(img)
    cv2.imwrite(f"{OUT}/frame_{idx:05d}.png", img[:,:,::-1])

# Save tiled image
row1 = np.concatenate(tiles[:6], axis=1)
row2 = np.concatenate(tiles[6:], axis=1)
grid = np.concatenate([row1, row2], axis=0)
cv2.imwrite(f"{OUT}/ep0_grid.png", grid[:,:,::-1])
print(f"Saved to {OUT}/ep0_grid.png")
print(f"Frame 0 image mean={imgs[0].mean():.1f} std={imgs[0].std():.1f}")