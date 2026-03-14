"""
Pick & Place — direct joint control, no IK confusion
marker is FROZEN until grasped, then follows EE exactly
"""
import sapien.core as sapien
import sapien
import numpy as np

URDF = '/home/rishabh/Downloads/myarm_m750_fixed.urdf'

scene = sapien.core.Scene()
scene.set_timestep(1/120)
scene.set_ambient_light([0.5,0.5,0.5])
scene.add_directional_light([0,1,-1],[1.0,0.95,0.85])
scene.add_ground(altitude=0)

loader = scene.create_urdf_loader()
loader.fix_root_link = True
robot  = loader.load(URDF)
robot.set_pose(sapien.core.Pose(p=[0, 0, 0]))
joints = robot.get_active_joints()
n      = len(joints)
links  = {l.name: l for l in robot.get_links()}
ee     = links['gripper']
pm     = robot.create_pinocchio_model()
for jt in joints: jt.set_drive_property(stiffness=20000, damping=2000)

def mat(c,r=0.6,m=0.0):
    x=sapien.render.RenderMaterial()
    x.base_color=c; x.roughness=r; x.metallic=m; return x

def sbox(half,mt,pos,name=""):
    b=scene.create_actor_builder()
    b.add_box_visual(half_size=half,material=mt)
    b.add_box_collision(half_size=half)
    a=b.build_static(name=name)
    a.set_pose(sapien.core.Pose(p=pos)); return a

# Environment
sbox([3,3,0.005], mat([0.55,0.50,0.45,1.0],0.9), [0,0,-0.005], "floor")
sbox([3,0.05,2],  mat([0.82,0.80,0.78,1.0],1.0), [0,2.5,1.0],  "wall_b")
sbox([0.05,3,2],  mat([0.82,0.80,0.78,1.0],1.0), [-2,0.0,1.0], "wall_s")

viewer = scene.create_viewer()
viewer.set_camera_xyz(0.8, -0.8, 1.2)
viewer.set_camera_rpy(0, -0.55, 0.65)

# ── First: settle at home and find REAL EE position ──────────────
q_home = np.zeros(n); q_home[1]=-0.3; q_home[2]=0.5
robot.set_qpos(q_home)
for i,jt in enumerate(joints): jt.set_drive_target(float(q_home[i]))
for _ in range(300):
    scene.step(); scene.update_render(); viewer.render()

real_ee = np.array(ee.get_entity_pose().p)
print(f"Real EE home: {np.round(real_ee,4)}")

TX, TY = real_ee[0], real_ee[1]
# From zarr data: EEF z min = 0.076 → robot CAN reach this
# Robot base at z=0, EE home at z=0.77
# We know from step9 data generation this works
TABLE_TOP_Z = 0.050
MARKER_Z    = 0.076
print(f"TABLE_TOP_Z={TABLE_TOP_Z:.3f}  MARKER_Z={MARKER_Z:.3f}")
sbox([0.30,0.28,0.025], mat([0.52,0.33,0.15,1.0],0.7), [TX,TY,TABLE_TOP_Z],    "table")
for lx,ly in [(TX+0.27,TY+0.23),(TX+0.27,TY-0.23),(TX-0.27,TY+0.23),(TX-0.27,TY-0.23)]:
    sbox([0.02,0.02,TABLE_TOP_Z], mat([0.38,0.22,0.10,1.0],0.8), [lx,ly,TABLE_TOP_Z/2], "leg")
sbox([0.20,0.18,0.002], mat([0.96,0.96,0.94,1.0],0.2), [TX,TY,TABLE_TOP_Z+0.027], "mat")

# ── Task objects ─────────────────────────────────────────────────
mb = scene.create_actor_builder()
mb.add_capsule_visual(radius=0.010,half_length=0.045,material=mat([0.92,0.05,0.05,1.0],0.3))
# NO collision — robot passes through marker, no physics pushing
marker = mb.build(name="marker")

gb = scene.create_actor_builder()
gb.add_box_visual(half_size=[0.05,0.05,0.022],material=mat([0.05,0.72,0.12,1.0],0.5))
gb.add_box_collision(half_size=[0.05,0.05,0.022])
tray = gb.build_static(name="tray")

# ── IK ───────────────────────────────────────────────────────────
from scipy.spatial.transform import Rotation

def solve_ik(xyz, grip):
    r=Rotation.from_euler('xyz',[np.pi,0,0]); qv=r.as_quat()
    pose=sapien.core.Pose(p=list(xyz),q=[qv[3],qv[0],qv[1],qv[2]])
    mask=np.ones(n,dtype=np.int32); mask[6:]=0
    q_cur=robot.get_qpos().copy()
    qr,ok,_=pm.compute_inverse_kinematics(ee.get_index(),pose,
        initial_qpos=q_cur.astype(np.float64),active_qmask=mask,
        max_iterations=500,dt=0.1,damp=1e-6)
    q=np.array(qr)
    if n>=7: q[6]=grip
    if n>=8: q[7]=grip
    return q, ok

grasped = False

def step_sim(update_marker=True):
    for _ in range(5): scene.step()
    if grasped and update_marker:
        p = np.array(ee.get_entity_pose().p)
        marker.set_pose(sapien.core.Pose(p=p.tolist()))
    scene.update_render()
    viewer.render()

def move_to(xyz, grip, steps=150):
    q_cur  = robot.get_qpos().copy()
    q_tgt, ok = solve_ik(xyz, grip)
    print(f"    IK {'✅' if ok else '❌'} target={np.round(xyz,3)}")
    for i in range(steps):
        if viewer.closed: return
        t=(i+1)/steps; s=t*t*(3-2*t)
        q_int = q_cur + s*(q_tgt-q_cur)
        for ji,jt in enumerate(joints): jt.set_drive_target(float(q_int[ji]))
        step_sim()

def set_grip(val, steps=60):
    q_cur=robot.get_qpos().copy(); q_tgt=q_cur.copy()
    if n>=7: q_tgt[6]=val
    if n>=8: q_tgt[7]=val
    for i in range(steps):
        if viewer.closed: return
        t=(i+1)/steps
        q_int=q_cur+t*(q_tgt-q_cur)
        for ji,jt in enumerate(joints): jt.set_drive_target(float(q_int[ji]))
        step_sim()

def wait(s=40):
    for _ in range(s):
        if viewer.closed: return
        step_sim()

OPEN=0.0345; CLOSE=0.0

# Episodes: marker at different offsets from EE home xy
offsets=[
    ([0.00, 0.00],[0.00, 0.15]),
    ([0.05, 0.10],[-.05,-0.10]),
    ([-.05, 0.10],[0.05,-0.10]),
    ([0.05,-0.10],[-.05, 0.10]),
    ([0.00,-0.12],[0.00, 0.12]),
]

print(f"\nTable center: ({TX:.3f}, {TY:.3f})")
print(f"Marker z:     {MARKER_Z:.3f}")

for ep_i,(mo,bo) in enumerate(offsets):
    if viewer.closed: break
    grasped=False

    mx=TX+mo[0]; my=TY+mo[1]
    bx=TX+bo[0]; by=TY+bo[1]

    # Place marker and box — STATIC, don't touch until grasped
    marker.set_pose(sapien.core.Pose(p=[mx, my, MARKER_Z]))
    tray.set_pose(sapien.core.Pose(p=[bx, by, TABLE_TOP_Z+0.022]))
    print(f"\n{'='*45}")
    print(f"Episode {ep_i+1}  marker=({mx:.3f},{my:.3f})  box=({bx:.3f},{by:.3f})")

    # Reset
    robot.set_qpos(q_home)
    for i,jt in enumerate(joints): jt.set_drive_target(float(q_home[i]))
    wait(60)

    # 1. Open gripper
    set_grip(OPEN, 40)

    # 2. Hover above marker (same xy, high z)
    print("  [2] Hover above marker")
    move_to([mx, my, MARKER_Z+0.15], OPEN, 150)
    wait(20)

    # 3. Descend slowly to marker
    print("  [3] Descend to marker")
    move_to([mx, my, MARKER_Z], OPEN, 200)
    wait(20)

    # Verify how close EE is to marker
    ee_now = np.array(ee.get_entity_pose().p)
    dist = np.linalg.norm(ee_now - np.array([mx,my,MARKER_Z]))
    print(f"  EE pos: {np.round(ee_now,3)}  dist to marker: {dist:.4f}")

    # 4. Close gripper
    print("  [4] Grasp")
    set_grip(CLOSE, 80)
    # Snap marker to exact EE position
    ee_now = np.array(ee.get_entity_pose().p)
    marker.set_pose(sapien.core.Pose(p=ee_now.tolist()))
    grasped = True
    wait(20)

    # 5. Lift
    print("  [5] Lift")
    move_to([mx, my, MARKER_Z+0.22], CLOSE, 150)
    wait(20)

    # 6. Carry to box
    print("  [6] Carry to box")
    move_to([bx, by, MARKER_Z+0.22], CLOSE, 180)
    wait(20)

    # 7. Lower
    print("  [7] Lower")
    move_to([bx, by, MARKER_Z+0.04], CLOSE, 130)
    wait(20)

    # 8. Release — marker stays at box
    print("  [8] Release")
    set_grip(OPEN, 60)
    grasped=False
    marker.set_pose(sapien.core.Pose(p=[bx, by, MARKER_Z]))
    wait(40)

    # 9. Retreat
    move_to([bx, by, MARKER_Z+0.22], OPEN, 100)
    wait(40)
    print(f"  ✅ Done!")

print("\nAll episodes complete!")
while not viewer.closed:
    scene.update_render(); viewer.render()