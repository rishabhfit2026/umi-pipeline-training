#!/usr/bin/env python3
"""
RDT2 Controller for M750 - Shows robot movement in RViz
Tests if your trained model learned pick-and-place
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import numpy as np
import torch
from PIL import Image
import sys, os
import zarr

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


class RDT2Controller(Node):
    def __init__(self):
        super().__init__('rdt2_controller')
        
        # Model paths - USING YOUR NEW MODEL!
        self.checkpoint = "/home/rishabh/Downloads/umi-pipeline-training/outputs/rdt2-m750-v2/checkpoint-1000"
        self.normalizer_path = "/home/rishabh/Downloads/umi-pipeline-training/umi_normalizer_official.pt"
        self.dataset_path = "/home/rishabh/Downloads/umi-pipeline-training/replay_buffer.zarr"
        self.instruction = "pick up the marker and place it in the box"
        
        self.get_logger().info("="*60)
        self.get_logger().info("🤖 RDT2 M750 CONTROLLER")
        self.get_logger().info("="*60)
        self.get_logger().info(f"Model: rdt2-m750-v2/checkpoint-1000")
        self.get_logger().info(f"Task:  '{self.instruction}'")
        self.get_logger().info("="*60)
        
        # Load models
        self.load_models()
        
        # ROS2 Publisher
        self.joint_pub = self.create_publisher(
            JointState,
            '/joint_states',
            10
        )
        
        # Timer for control loop (10 Hz)
        self.timer = self.create_timer(0.1, self.control_loop)
        self.step_count = 0
        
        # Load test images
        self.load_test_images()
        
        self.get_logger().info("✅ Controller ready! Robot will move in RViz...")
    
    def load_models(self):
        """Load your NEW trained model"""
        os.environ['TRANSFORMERS_ATTN_IMPLEMENTATION'] = 'eager'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        self.get_logger().info("[1/5] Loading base model...")
        self.base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "robotics-diffusion-transformer/RDT2-VQ",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map="cuda"
        )
        
        self.get_logger().info("[2/5] Loading checkpoint-1000...")
        self.model = PeftModel.from_pretrained(self.base, self.checkpoint)
        self.model.eval()
        
        self.get_logger().info("[3/5] Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            use_fast=True
        )
        
        self.get_logger().info("[4/5] Loading VAE...")
        self.vae = MultiVQVAE.from_pretrained(
            "robotics-diffusion-transformer/RVQActionTokenizer"
        ).eval().to(device="cuda", dtype=torch.float32)
        
        self.get_logger().info("[5/5] Loading normalizer...")
        self.normalizer = LinearNormalizer.load(self.normalizer_path)
        
        self.valid_action_id_length = (
            self.vae.pos_id_len +
            self.vae.rot_id_len +
            self.vae.grip_id_len
        )
        
        self.get_logger().info("✅ Models loaded!")
    
    def load_test_images(self):
        """Load images from training dataset"""
        root = zarr.open(self.dataset_path, mode='r')
        total = len(root['data']['camera0_rgb'])
        
        # Load diverse frames for testing
        indices = list(range(0, min(total, 10000), 200))  # Every 200th frame
        
        self.test_images = []
        for idx in indices:
            img = np.array(root['data']['camera0_rgb'][idx], dtype=np.uint8)
            self.test_images.append(img)
        
        self.get_logger().info(f"Loaded {len(self.test_images)} test images")
        self.current_image_idx = 0
    
    def rot6d_to_euler(self, rot6d):
        """Convert 6D rotation to Euler angles"""
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
        """Get action from your model"""
        img = np.array(
            Image.fromarray(image_np).resize((384, 384)),
            dtype=np.uint8
        )
        img_t = torch.from_numpy(img).unsqueeze(0)
        
        with torch.no_grad():
            result = batch_predict_action(
                self.model,
                self.processor,
                self.vae,
                self.normalizer,
                examples=[{
                    "obs": {"camera0_rgb": img_t},
                    "meta": {"num_camera": 1}
                }],
                valid_action_id_length=self.valid_action_id_length,
                apply_jpeg_compression=True,
                instruction=self.instruction
            )
        
        ac = result["action_pred"][0].cpu().detach()
        ac[:, 9] = ac[:, 9] / 0.088 * 0.1
        
        pos = ac[0, 0:3].float().numpy() * 1000  # mm
        r6d = ac[0, 3:9].float().numpy()
        
        rpy = self.rot6d_to_euler(r6d)
        
        # Simple IK (approximate)
        joints = self.xyz_to_joints(pos, rpy)
        
        return joints
    
    def xyz_to_joints(self, xyz_mm, rpy_rad):
        """Simple inverse kinematics"""
        x, y, z = xyz_mm / 1000.0  # Convert to meters
        rx, ry, rz = rpy_rad
        
        # Base rotation
        j1 = np.arctan2(y, x)
        r = np.sqrt(x**2 + y**2)
        
        # Arm lengths
        L1, L2 = 0.1105, 0.1105
        z_rel = z - 0.12
        r_rel = r - 0.05
        
        # 2-link IK
        D = (r_rel**2 + z_rel**2 - L1**2 - L2**2) / (2 * L1 * L2)
        D = np.clip(D, -1, 1)
        
        j3 = np.arctan2(np.sqrt(1 - D**2), D)
        j2 = np.arctan2(z_rel, r_rel) - np.arctan2(
            L2*np.sin(j3), L1+L2*np.cos(j3)
        )
        
        # End effector orientation
        j4 = rx
        j5 = ry
        j6 = rz
        
        return np.array([j1, j2, j3, j4, j5, j6])
    
    def control_loop(self):
        """Main control loop - runs at 10Hz"""
        if self.step_count >= len(self.test_images):
            self.get_logger().info("✅ Test complete!")
            return
        
        # Get current test image
        img = self.test_images[self.step_count]
        
        # Get predicted joint angles
        joints = self.predict_action(img)
        
        # Publish to RViz
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        
        msg.name = [
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
            'joint6'
        ]
        msg.position = joints.tolist()
        
        self.joint_pub.publish(msg)
        
        self.get_logger().info(
            f"Step {self.step_count+1}/{len(self.test_images)}: "
            f"joints={np.degrees(joints).round(1)}"
        )
        
        self.step_count += 1


def main(args=None):
    rclpy.init(args=args)
    controller = RDT2Controller()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

