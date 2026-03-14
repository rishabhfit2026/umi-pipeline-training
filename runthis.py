"""
zarr_explore.py — Run this and paste the full output back to Claude
conda activate maniskill2 && python zarr_explore.py
"""
import zarr, numpy as np

ZARR = '/home/rishabh/Downloads/umi-pipeline-training/outputs/sim_replay_buffer3.zarr'

print("=" * 60)
print("ZARR EXPLORER")
print("=" * 60)

z = zarr.open(ZARR, 'r')

def explore(node, prefix=""):
    if hasattr(node, 'keys'):
        for k in node.keys():
            explore(node[k], prefix + "/" + k)
    else:
        arr = node
        print(f"\n{prefix}")
        print(f"  shape   : {arr.shape}")
        print(f"  dtype   : {arr.dtype}")
        if arr.dtype in [np.float32, np.float64, np.float16]:
            data = arr[:]
            print(f"  min/max : {data.min():.4f} / {data.max():.4f}")
            print(f"  mean    : {data.mean():.4f}")
            print(f"  sample[0]: {data[0]}")
        elif arr.dtype == np.uint8 and len(arr.shape) >= 3:
            print(f"  *** LIKELY IMAGE DATA ***")
            print(f"  sample frame shape: {arr[0].shape}")
        else:
            print(f"  sample[0]: {arr[0]}")

explore(z)

print("\n" + "=" * 60)
print("TOP-LEVEL KEYS:")
for k in z.keys():
    print(f"  {k}: {list(z[k].keys()) if hasattr(z[k], 'keys') else z[k].shape}")

print("\n" + "=" * 60)
print("EPISODE INFO:")
ends = z['meta']['episode_ends'][:]
starts = np.concatenate([[0], ends[:-1]])
lengths = ends - starts
print(f"  Num episodes : {len(ends)}")
print(f"  Total frames : {ends[-1]}")
print(f"  Episode len  : min={lengths.min()} max={lengths.max()} mean={lengths.mean():.1f}")

print("\n" + "=" * 60)
print("ACTION SPACE (what model needs to predict):")
for key in ['robot0_eef_pos', 'robot0_eef_rot_axis_angle', 'robot0_gripper_width']:
    if 'data' in z and key in z['data']:
        d = z['data'][key][:]
        print(f"  {key}: shape={d.shape} min={d.min():.4f} max={d.max():.4f}")

print("\n" + "=" * 60)
print("OBSERVATION SPACE (model inputs):")
obs_keys = [k for k in z['data'].keys() 
            if k not in ['robot0_eef_pos','robot0_eef_rot_axis_angle','robot0_gripper_width']]
if obs_keys:
    for k in obs_keys:
        d = z['data'][k]
        print(f"  {k}: shape={d.shape} dtype={d.dtype}")
else:
    print("  ⚠️  No separate observation keys found!")
    print("  Only action data present — model needs to use past actions as obs")

print("\nDone! Paste this output back to Claude.")