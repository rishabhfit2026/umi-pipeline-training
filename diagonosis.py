"""
Diagnose inference failure:
1. Compare training camera frames vs inference camera frames
2. Check what actions the model is actually predicting
3. Verify IK is solving correctly

python diagnose_inference.py
"""
import sapien, numpy as np, torch, zarr, cv2, os
from scipy.spatial.transform import Rotation
from torchvision import transforms
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import sys
sys.path.insert(0, '/home/rishabh/Downloads/umi-pipeline-training')
from diffusionpolicy import (
    DiffusionPolicy, ActionNormalizer,
    OBS_HORIZON, ACTION_HORIZON, ACTION_DIM, SAVE_DIR
)

URDF  = '/home/rishabh/Downloads/myarm_m750_fixed.urdf'
ZARR  = '/home/rishabh/Downloads/umi-pipeline-training/outputs/sim_replay_buffer3.zarr'
CKPT  = f'{SAVE_DIR}/best_model.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 224
OUT_DIR = '/home/rishabh/Downloads/umi-pipeline-training/diagnose_frames'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────
ckpt = torch.load(CKPT, map_location=DEVICE)
model = DiffusionPolicy(OBS_HORIZON, ACTION_HORIZON, ACTION_DIM).to(DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()
normalizer = ActionNormalizer.load(f'{SAVE_DIR}/normalizer.pt')
noise_scheduler = DDIMScheduler(
    num_train_timesteps=100, beta_schedule='squaredcos_cap_v2',
    clip_sample=True, prediction_type='epsilon')
noise_scheduler.set_timesteps(16)
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# ── Save training frames from zarr ───────────────────────────────
print("Saving training frames from zarr (episode 0)...")
z = zarr.open(ZARR, 'r')
imgs = z['data']['camera0_rgb'][:]
pos  = z['data']['robot0_eef_pos'][:]
grip = z['data']['robot0_gripper_width'][:]
ends = z['meta']['episode_ends'][:]

# Save frames 0, 60, 120, 180, 240, 300 from episode 0
for fi in [0, 60, 120, 180, 240, 300, 359]:
    frame = imgs[fi]
    cv2.imwrite(f'{OUT_DIR}/TRAIN_frame_{fi:03d}_pos{pos[fi,0]:.2f},{pos[fi,1]:.2f},{pos[fi,2]:.2f}_grip{grip[fi,0]:.3f}.png',
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
print(f"  Saved {len([0,60,120,180,240,300,359])} training frames to {OUT_DIR}/TRAIN_*.png")

# ── Setup scene ───────────────────────────────────────────────────
scene = sapien.Scene()
scene.set_timestep(1/120)
scene.set_ambient_light([0.6,0.6,0.6])
scene.add_directional_light([0,1,-1],[1.0,0.95,0.85])
scene.add_ground(altitude=0)

loader = scene.create_urdf_loader(); loader.fix_root_link=True
robot = loader.load(URDF); robot.set_pose(sapien.Pose(p=[0,0,0]))
joints = robot.get_active_joints(); N=len(joints)
links = {l.name:l for l in robot.get_links()}
ee = links['gripper']
ee_idx = ee.get_index(); pm = robot.create_pinocchio_model()
for jt in joints: jt.set_drive_property(stiffness=20000, damping=2000)

q0=np.zeros(N); q0[1]=-0.3; q0[2]=0.5
robot.set_qpos(q0)
for i,jt in enumerate(joints): jt.set_drive_target(float(q0[i]))
for _ in range(200): scene.step()
real_ee = np.array(ee.get_entity_pose().p); TX,TY = real_ee[0],real_ee[1]
print(f"\nEE home: ({TX:.3f},{TY:.3f})")
print(f"Training data EEF x range: {pos[:,0].min():.3f} → {pos[:,0].max():.3f}")
print(f"Training data EEF y range: {pos[:,1].min():.3f} → {pos[:,1].max():.3f}")
print(f"Training data EEF z range: {pos[:,2].min():.3f} → {pos[:,2].max():.3f}")

# ── Scene objects ─────────────────────────────────────────────────
def make_box(half,color,pos_,static=True,name=""):
    mt=sapien.render.RenderMaterial(); mt.base_color=color
    b=scene.create_actor_builder()
    b.add_box_visual(half_size=half,material=mt)
    b.add_box_collision(half_size=half)
    a=b.build_static(name=name) if static else b.build(name=name)
    a.set_pose(sapien.Pose(p=pos_)); return a

make_box([0.30,0.28,0.025],[0.52,0.33,0.15,1.0],[TX,TY,0.025],True,"table")
make_box([0.20,0.18,0.002],[0.96,0.96,0.94,1.0],[TX,TY,0.052],True,"mat")
mr=sapien.render.RenderMaterial(); mr.base_color=[0.95,0.10,0.10,1.0]
bm=scene.create_actor_builder(); bm.add_sphere_visual(radius=0.018,material=mr)
sim_marker=bm.build(name="marker")
mg=sapien.render.RenderMaterial(); mg.base_color=[0.05,0.80,0.15,1.0]
gb=scene.create_actor_builder()
gb.add_box_visual(half_size=[0.055,0.055,0.025],material=mg)
gb.add_box_collision(half_size=[0.055,0.055,0.025])
sim_box=gb.build_static(name="box")

# Place ep0 objects
rng=np.random.default_rng(42)
mx=TX+rng.uniform(-0.05,0.05); my=TY+rng.uniform(-0.10,0.10)
bx=TX+rng.uniform(-0.05,0.05); by=TY+rng.uniform(0.05,0.15)
sim_marker.set_pose(sapien.Pose(p=[mx,my,0.076]))
sim_box.set_pose(sapien.Pose(p=[bx,by,0.075]))
print(f"Episode 0: marker=({mx:.3f},{my:.3f}) box=({bx:.3f},{by:.3f})")

# ── Test MULTIPLE camera positions ────────────────────────────────
print("\n" + "="*60)
print("TESTING DIFFERENT CAMERA POSITIONS")
print("Compare INFER_cam_*.png with TRAIN_*.png visually!")

camera_configs = [
    # name, pos, quaternion
    ("front_angled",  [TX+0.0, TY-0.4, 0.6],  [0.9239, 0.3827, 0, 0]),
    ("top_down",      [TX,     TY,      0.8],  [0.7071, 0.7071, 0, 0]),
    ("side_high",     [TX+0.5, TY,      0.7],  [0.7071, 0,      0.7071, 0]),
    ("current",       [TX,     TY-0.3,  0.8],  [0.9239, 0.3827, 0, 0]),
    ("closer_front",  [TX+0.0, TY-0.5,  0.5],  [0.9659, 0.2588, 0, 0]),
    ("overhead_tilt", [TX-0.1, TY-0.2,  0.9],  [0.8660, 0.5000, 0, 0]),
]

for cam_name, cam_pos, cam_q in camera_configs:
    cam_entity = sapien.Entity()
    cam_comp   = sapien.render.RenderCameraComponent(IMG_SIZE, IMG_SIZE)
    cam_comp.set_fovy(np.deg2rad(60))
    cam_entity.set_pose(sapien.Pose(p=cam_pos, q=cam_q))
    cam_entity.add_component(cam_comp)
    scene.add_entity(cam_entity)
    scene.update_render()
    cam_comp.take_picture()
    rgba = cam_comp.get_picture('Color')
    rgb  = (rgba[:,:,:3]*255).clip(0,255).astype(np.uint8)
    cv2.imwrite(f'{OUT_DIR}/INFER_cam_{cam_name}.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"  Saved INFER_cam_{cam_name}.png  pos={cam_pos}")
    scene.remove_entity(cam_entity)

# ── Check what model predicts with different cameras ──────────────
print("\n" + "="*60)
print("WHAT DOES MODEL PREDICT FOR EACH CAMERA VIEW?")
print("(comparing to training data ground truth)")

print(f"\nTraining ep0 frame0 actual actions:")
print(f"  EEF pos : {pos[0]}")
print(f"  grip    : {grip[0,0]:.4f}")

def predict_from_frame(frame_rgb):
    obs = [frame_rgb, frame_rgb]  # use same frame twice for 2-frame obs
    imgs = [img_transform(f) for f in obs]
    obs_img = torch.cat(imgs, dim=0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        noisy = torch.randn(1, ACTION_HORIZON, ACTION_DIM, device=DEVICE)
        for t in noise_scheduler.timesteps:
            ts = torch.tensor([t], device=DEVICE).long()
            pred_noise = model(obs_img, noisy, ts)
            noisy = noise_scheduler.step(pred_noise, t, noisy).prev_sample
        actions = normalizer.denormalize(noisy[0])
    return actions.cpu().numpy()

print(f"\nModel predictions from TRAINING frames (should match ~{pos[0]}):")
for fi in [0, 60, 120]:
    preds = predict_from_frame(imgs[fi])
    print(f"  train frame {fi:3d} → pred[0] pos=({preds[0,0]:.3f},{preds[0,1]:.3f},{preds[0,2]:.3f}) grip={preds[0,6]:.4f}")
    print(f"                      actual  pos=({pos[fi,0]:.3f},{pos[fi,1]:.3f},{pos[fi,2]:.3f}) grip={grip[fi,0]:.4f}")

print(f"\nDone! Open {OUT_DIR}/ and compare:")
print(f"  TRAIN_frame_*.png   ← what model was trained on")
print(f"  INFER_cam_*.png     ← what model sees during inference")
print(f"  Find the camera that looks MOST SIMILAR to TRAIN frames!")