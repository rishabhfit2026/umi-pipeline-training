#!/usr/bin/env python3
"""
Test RDT2 Fine-tuned Model with M750 in RViz
"""

import rospy
import numpy as np
import torch
from PIL import Image
import sys, os
from sensor_msgs.msg import JointState

# RDT2 paths
RDT2_DIR = "/home/rishabh/Downloads/umi-pipeline-training/RDT2"
sys.path.insert(0, RDT2_DIR)
sys.path.insert(0, os.path.join(RDT2_DIR, 'vqvae'))
sys.path.insert(0, os.path.join(RDT2_DIR, 'models'))
os.chdir(RDT2_DIR)

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from models.multivqvae import MultiVQVAE
from models.normalizer.normalizer import LinearNormalizer
from utils import batch_predict_action


class RDT2RvizTester:
    def __init__(self):
        rospy.init_node('rdt2_rviz_tester', anonymous=True)
        
        self.checkpoint = "/home/rishabh/Downloads/umi-pipeline-training/outputs/rdt2-finetuned/checkpoint-5000"
        self.normalizer_path = "/home/rishabh/Downloads/umi-pipeline-training/umi_normalizer_official.pt"
        self.instruction = "pick up the marker and place it in the box"
        self.dataset_path = "/home/rishabh/Downloads/umi-pipeline-training/replay_buffer.zarr"
        
        print("="*70)
        print("🧪 TESTING RDT2 MODEL")
        print("="*70)
        
        self.load_models()
        self.js_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
        
        print("✅ Ready!")
    
    def load_models(self):
        os.environ['TRANSFORMERS_ATTN_IMPLEMENTATION'] = 'eager'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        print("[1/5] Loading base model...")
        self.base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "robotics-diffusion-transformer/RDT2-VQ",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map="cuda"
        )
        
        print("[2/5] Loading checkpoint-5000...")
        self.model = PeftModel.from_pretrained(self.base, self.checkpoint)
        self.model.eval()
        
        print("[3/5] Loading processor...")
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)
        
        print("[4/5] Loading VAE...")
        self.vae = MultiVQVAE.from_pretrained(
            "robotics-diffusion-transformer/RVQActionTokenizer"
        ).eval().to(device="cuda", dtype=torch.float32)
        
        print("[5/5] Loading normalizer...")
        self.normalizer = LinearNormalizer.load(self.normalizer_path)
        
        self.valid_action_id_length = self.vae.pos_id_len + self.vae.rot_id_len + self.vae.grip_id_len
        print("✅ Models loaded!")
    
    def rot6d_to_euler(self, rot6d):
        a1 = rot6d[:3]
        a2 = rot6d[3:]
        b1 = a1 / (np.linalg.norm(a1) + 1e-8)
        b2 = a2 - np.dot(b1, a2) * b1
        b2 = b2 / (np.linalg.norm(b2) + 1e-8)
        b3 = np.cross(b1, b2)
        R  = np.stack([b1, b2, b3], axis=-1)
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        if sy > 1e-6:
            rx = np.arctan2(R[2,1], R[2,2])
            ry = np.arctan2(-R[2,0], sy)
            rz = np.arctan2(R[1,0], R[0,0])
        else:
            rx = np.arctan2(-R[1,2], R[1,1])
            ry = np.arctan2(-R[2,0], sy)
            rz = 0.0
        return np.array([rx, ry, rz])
    
    def predict_action(self, image_np):
        img = np.array(Image.fromarray(image_np).resize((384, 384)), dtype=np.uint8)
        img_t = torch.from_numpy(img).unsqueeze(0)
        
        with torch.no_grad():
            result = batch_predict_action(
                self.model, self.processor, self.vae, self.normalizer,
                examples=[{"obs": {"camera0_rgb": img_t}, "meta": {"num_camera": 1}}],
                valid_action_id_length=self.valid_action_id_length,
                apply_jpeg_compression=True,
                instruction=self.instruction
            )
        
        ac = result["action_pred"][0].cpu().detach()
        ac[:, 9] = ac[:, 9] / 0.088 * 0.1
        
        pos = ac[0, 0:3].float().numpy() * 1000
        r6d = ac[0, 3:9].float().numpy()
        grip = float(ac[0, 9].float().item())
        
        rpy = self.rot6d_to_euler(r6d)
        gripper_pct = np.clip(grip / 0.1 * 100, 0, 100)
        
        return pos, rpy, gripper_pct
    
    def xyz_to_joints(self, xyz_mm, rpy_rad):
        x, y, z = xyz_mm / 1000.0
        rx, ry, rz = rpy_rad
        
        j1 = np.arctan2(y, x)
        r = np.sqrt(x**2 + y**2)
        
        L1, L2 = 0.1105, 0.1105
        z_rel = z - 0.12
        r_rel = r - 0.05
        
        D = (r_rel**2 + z_rel**2 - L1**2 - L2**2) / (2 * L1 * L2)
        D = np.clip(D, -1, 1)
        
        j3 = np.arctan2(np.sqrt(1 - D**2), D)
        j2 = np.arctan2(z_rel, r_rel) - np.arctan2(L2*np.sin(j3), L1+L2*np.cos(j3))
        
        return np.array([j1, j2, j3, rx, ry, rz])
    
    def publish_joints(self, angles):
        js = JointState()
        js.header.stamp = rospy.Time.now()
        js.header.frame_id = "base_link"
        js.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        js.position = angles.tolist()
        self.js_pub.publish(js)
    
    def load_images(self):
        import zarr
        root = zarr.open(self.dataset_path, mode='r')
        total = len(root['data']['camera0_rgb'])
        indices = [500, 5000, 10000, 15000, 20000, 25000, 30000]
        
        images = []
        for idx in indices:
            if idx < total:
                images.append(np.array(root['data']['camera0_rgb'][idx], dtype=np.uint8))
        
        print(f"Loaded {len(images)} test images")
        return images
    
    def test(self):
        print("\n" + "="*70)
        print("🧪 TESTING MODEL")
        print("="*70)
        
        images = self.load_images()
        results = []
        
        for i, img in enumerate(images):
            print(f"\nTest {i+1}/{len(images)}:")
            
            xyz, rpy, grip = self.predict_action(img)
            joints = self.xyz_to_joints(xyz, rpy)
            self.publish_joints(joints)
            
            safe = -400 <= xyz[0] <= 400 and -400 <= xyz[1] <= 400 and 50 <= xyz[2] <= 600
            
            print(f"  Position: x={xyz[0]:.1f}, y={xyz[1]:.1f}, z={xyz[2]:.1f} mm")
            print(f"  Gripper: {grip:.0f}%")
            print(f"  {'✅ SAFE' if safe else '❌ UNSAFE'}")
            
            results.append({'xyz': xyz, 'grip': grip, 'safe': safe})
            rospy.sleep(1.0)
        
        # Analysis
        positions = np.array([r['xyz'] for r in results])
        grips = np.array([r['grip'] for r in results])
        
        x_range = positions[:, 0].max() - positions[:, 0].min()
        y_range = positions[:, 1].max() - positions[:, 1].min()
        g_range = grips.max() - grips.min()
        
        print("\n" + "="*70)
        print("📊 RESULTS")
        print("="*70)
        print(f"X range: {x_range:.1f}mm  (good if >50mm)")
        print(f"Y range: {y_range:.1f}mm  (good if >50mm)")
        print(f"Gripper: {g_range:.0f}%  (good if >20%)")
        
        if x_range > 50 and y_range > 50 and g_range > 20:
            print("\n✅ MODEL LEARNED!")
        elif x_range > 20:
            print("\n⚠️  PARTIALLY LEARNED")
        else:
            print("\n❌ DID NOT LEARN - RETRAIN NEEDED")
        print("="*70)
    
    def run(self):
        rospy.sleep(2.0)
        self.test()
        print("\n✅ Done!")


if __name__ == '__main__':
    try:
        tester = RDT2RvizTester()
        tester.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
