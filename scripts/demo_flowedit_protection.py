"""
FlowEdit保护性噪声演示脚本

此脚本展示如何:
1. 加载FlowEdit模型和示例图像
2. 应用四种保护性噪声方法
3. 对比保护前后的编辑效果

使用方法:
    python demo_flowedit_protection.py --model_type FLUX --image_path example_images/gas_station.png
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

# 添加项目根目录到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 导入自定义模块
from src.protection.flowedit_protection.protective_noise import (
    FrequencyDomainNoise,
    FeatureSpaceAdversarialNoise,
    MultiScaleTextureNoise,
    VelocityFieldAdversarialNoise,
    ProtectiveNoiseOptimizer,
    NoiseConfig
)
from src.protection.flowedit_protection.loss_functions import (
    EditQualityLoss,
    ImperceptibilityLoss,
    RobustnessLoss,
    FeatureAdversarialLoss,
    FrequencyAdversarialLoss,
    CombinedLoss,
    LossConfig
)


class FlowEditProtectionDemo:
    """FlowEdit保护性噪声演示"""
    
    def __init__(self, model_type: str = 'FLUX', device: str = 'cuda'):
        """
        初始化演示
        
        Args:
            model_type: 模型类型 ('FLUX' 或 'SD3')
            device: 设备 ('cuda' 或 'cpu')
        """
        self.model_type = model_type
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.pipe = None
        self.scheduler = None
    
    def load_image(self, image_path: str, target_size: Tuple[int, int] = None) -> torch.Tensor:
        """
        加载并预处理图像
        
        Args:
            image_path: 图像路径
            target_size: 目标大小 (H, W)，如果为None则使用原始大小
            
        Returns:
            预处理后的图像张量 [1, C, H, W]，值域[0, 1]
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 调整大小
        if target_size is not None:
            image = image.resize(target_size, Image.LANCZOS)
        else:
            # 确保尺寸能被16整除
            w, h = image.size
            w = (w // 16) * 16
            h = (h // 16) * 16
            image = image.crop((0, 0, w, h))
        
        # 转换为张量
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)  # [C, H, W]
        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # [1, C, H, W]
        
        return image_tensor
    
    def save_image(self, image_tensor: torch.Tensor, save_path: str):
        """
        保存图像张量
        
        Args:
            image_tensor: 图像张量 [C, H, W] 或 [1, C, H, W]，值域[0, 1]
            save_path: 保存路径
        """
        # 处理批处理维度
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.squeeze(0)
        
        # 转换为numpy数组
        image_array = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_array = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
        
        # 保存图像
        image = Image.fromarray(image_array)
        image.save(save_path)
        print(f"图像已保存到: {save_path}")
    
    def compute_image_metrics(self, image1: torch.Tensor, 
                             image2: torch.Tensor) -> dict:
        """
        计算图像相似度指标
        
        Args:
            image1: 第一张图像 [B, C, H, W]
            image2: 第二张图像 [B, C, H, W]
            
        Returns:
            指标字典
        """
        metrics = {}
        
        # PSNR (Peak Signal-to-Noise Ratio)
        mse = F.mse_loss(image1, image2)
        psnr = 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse + 1e-8))
        metrics['psnr'] = psnr.item()
        
        # SSIM (Structural Similarity Index) - 简化版
        # 计算均值和方差
        mu1 = F.avg_pool2d(image1, kernel_size=11, padding=5)
        mu2 = F.avg_pool2d(image2, kernel_size=11, padding=5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(image1 ** 2, kernel_size=11, padding=5) - mu1_sq
        sigma2_sq = F.avg_pool2d(image2 ** 2, kernel_size=11, padding=5) - mu2_sq
        sigma12 = F.avg_pool2d(image1 * image2, kernel_size=11, padding=5) - mu1_mu2
        
        # SSIM公式
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        
        metrics['ssim'] = ssim.mean().item()
        
        # L∞距离
        linf = torch.norm(image1 - image2, p=float('inf')).item()
        metrics['linf'] = linf
        
        # L2距离
        l2 = torch.norm(image1 - image2, p=2).item()
        metrics['l2'] = l2
        
        return metrics
    
    def demonstrate_frequency_noise(self, image: torch.Tensor, 
                                   output_dir: str = 'outputs'):
        """
        演示频域对抗噪声
        
        Args:
            image: 输入图像 [1, C, H, W]
            output_dir: 输出目录
        """
        print("\n" + "="*60)
        print("演示1: 频域对抗噪声 (Frequency Domain Adversarial Noise)")
        print("="*60)
        
        # 创建噪声生成器
        freq_noise = FrequencyDomainNoise(freq_cutoff=0.3, amplitude=0.05)
        
        # 生成噪声
        print("生成频域对抗噪声...")
        noisy_image = freq_noise.generate(image)
        
        # 计算指标
        metrics = self.compute_image_metrics(image, noisy_image)
        print(f"图像相似度指标:")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  SSIM: {metrics['ssim']:.4f}")
        print(f"  L∞距离: {metrics['linf']:.6f}")
        print(f"  L2距离: {metrics['l2']:.6f}")
        
        # 保存结果
        os.makedirs(output_dir, exist_ok=True)
        self.save_image(image, f"{output_dir}/01_original.png")
        self.save_image(noisy_image, f"{output_dir}/01_freq_noise.png")
        
        return noisy_image, metrics
    
    def demonstrate_texture_noise(self, image: torch.Tensor,
                                 output_dir: str = 'outputs'):
        """
        演示多尺度纹理对抗噪声
        
        Args:
            image: 输入图像 [1, C, H, W]
            output_dir: 输出目录
        """
        print("\n" + "="*60)
        print("演示2: 多尺度纹理对抗噪声 (Multi-Scale Texture Noise)")
        print("="*60)
        
        # 创建噪声生成器
        texture_noise = MultiScaleTextureNoise(
            num_scales=4,
            num_orientations=4,
            amplitude_range=(0.02, 0.08)
        )
        
        # 生成噪声
        print("生成多尺度纹理对抗噪声...")
        noisy_image = texture_noise.generate(image)
        
        # 计算指标
        metrics = self.compute_image_metrics(image, noisy_image)
        print(f"图像相似度指标:")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  SSIM: {metrics['ssim']:.4f}")
        print(f"  L∞距离: {metrics['linf']:.6f}")
        print(f"  L2距离: {metrics['l2']:.6f}")
        
        # 保存结果
        os.makedirs(output_dir, exist_ok=True)
        self.save_image(noisy_image, f"{output_dir}/02_texture_noise.png")
        
        return noisy_image, metrics
    
    def demonstrate_combined_noise(self, image: torch.Tensor,
                                  output_dir: str = 'outputs'):
        """
        演示综合保护性噪声
        
        Args:
            image: 输入图像 [1, C, H, W]
            output_dir: 输出目录
        """
        print("\n" + "="*60)
        print("演示3: 综合保护性噪声 (Combined Protective Noise)")
        print("="*60)
        
        # 创建配置
        config = NoiseConfig(
            freq_enabled=True,
            texture_enabled=True,
            feature_enabled=False,  # 需要特征提取器
            velocity_enabled=False,  # 需要流模型
            freq_weight=0.5,
            texture_weight=0.5,
            feature_weight=0.0,
            velocity_weight=0.0
        )
        
        # 创建优化器
        optimizer = ProtectiveNoiseOptimizer(config)
        
        # 生成噪声
        print("生成综合保护性噪声...")
        noisy_image = optimizer.generate(image)
        
        # 计算指标
        metrics = self.compute_image_metrics(image, noisy_image)
        print(f"图像相似度指标:")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  SSIM: {metrics['ssim']:.4f}")
        print(f"  L∞距离: {metrics['linf']:.6f}")
        print(f"  L2距离: {metrics['l2']:.6f}")
        
        # 保存结果
        os.makedirs(output_dir, exist_ok=True)
        self.save_image(noisy_image, f"{output_dir}/03_combined_noise.png")
        
        return noisy_image, metrics
    
    def demonstrate_loss_functions(self):
        """
        演示损失函数配置
        """
        print("\n" + "="*60)
        print("演示4: 损失函数配置")
        print("="*60)
        
        # 创建损失函数配置
        loss_config = LossConfig(
            edit_quality_enabled=True,
            imperceptibility_enabled=True,
            robustness_enabled=True,
            feature_adv_enabled=True,
            freq_adv_enabled=True,
            edit_quality_weight=0.6,
            imperceptibility_weight=0.2,
            robustness_weight=0.15,
            feature_adv_weight=0.4,
            freq_adv_weight=0.25
        )
        
        print("损失函数配置:")
        print(f"  EditQualityLoss: 启用={loss_config.edit_quality_enabled}, 权重={loss_config.edit_quality_weight}")
        print(f"  ImperceptibilityLoss: 启用={loss_config.imperceptibility_enabled}, 权重={loss_config.imperceptibility_weight}")
        print(f"  RobustnessLoss: 启用={loss_config.robustness_enabled}, 权重={loss_config.robustness_weight}")
        print(f"  FeatureAdversarialLoss: 启用={loss_config.feature_adv_enabled}, 权重={loss_config.feature_adv_weight}")
        print(f"  FrequencyAdversarialLoss: 启用={loss_config.freq_adv_enabled}, 权重={loss_config.freq_adv_weight}")
        
        # 创建综合损失函数
        combined_loss = CombinedLoss(loss_config)
        print("\n综合损失函数已创建")
        print("所有损失函数都可通过LossConfig灵活启用/禁用")
    
    def run_demo(self, image_path: str, output_dir: str = 'outputs'):
        """
        运行完整演示
        
        Args:
            image_path: 图像路径
            output_dir: 输出目录
        """
        print("\n" + "="*60)
        print("FlowEdit保护性噪声演示")
        print("="*60)
        
        # 加载图像
        print(f"\n加载图像: {image_path}")
        if not os.path.exists(image_path):
            print(f"错误: 找不到图像文件 {image_path}")
            return
        
        image = self.load_image(image_path)
        print(f"图像大小: {image.shape}")
        print(f"图像值域: [{image.min():.4f}, {image.max():.4f}]")
        
        # 演示各种噪声方法
        results = {}
        
        # 演示1: 频域噪声
        noisy_freq, metrics_freq = self.demonstrate_frequency_noise(image, output_dir)
        results['frequency'] = metrics_freq
        
        # 演示2: 纹理噪声
        noisy_texture, metrics_texture = self.demonstrate_texture_noise(image, output_dir)
        results['texture'] = metrics_texture
        
        # 演示3: 综合噪声
        noisy_combined, metrics_combined = self.demonstrate_combined_noise(image, output_dir)
        results['combined'] = metrics_combined
        
        # 演示4: 损失函数配置
        self.demonstrate_loss_functions()
        
        # 打印总结
        print("\n" + "="*60)
        print("演示总结")
        print("="*60)
        print("\n各种噪声方法的性能对比:")
        print(f"{'方法':<20} {'PSNR':<10} {'SSIM':<10} {'L∞':<12} {'L2':<12}")
        print("-" * 64)
        for method, metrics in results.items():
            print(f"{method:<20} {metrics['psnr']:<10.2f} {metrics['ssim']:<10.4f} {metrics['linf']:<12.6f} {metrics['l2']:<12.6f}")
        
        print("\n" + "="*60)
        print("演示完成！")
        print(f"输出文件已保存到: {output_dir}")
        print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='FlowEdit保护性噪声演示')
    parser.add_argument('--model_type', type=str, default='FLUX',
                       choices=['FLUX', 'SD3'],
                       help='模型类型')
    parser.add_argument('--image_path', type=str, 
                       default='modules/FlowEdit/example_images/gas_station.png',
                       help='输入图像路径')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='计算设备')
    
    args = parser.parse_args()
    
    # 创建演示对象
    demo = FlowEditProtectionDemo(model_type=args.model_type, device=args.device)
    
    # 运行演示
    demo.run_demo(args.image_path, args.output_dir)


if __name__ == '__main__':
    main()
