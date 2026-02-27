"""
针对FlowEdit的损失函数模块

包含以下损失函数:
1. EditQualityLoss: 编辑质量退化损失
2. ImperceptibilityLoss: 不可见性损失
3. RobustnessLoss: 鲁棒性损失
4. FeatureAdversarialLoss: 特征空间对抗损失
5. FrequencyAdversarialLoss: 频域对抗损失

所有损失函数都可通过配置启用/禁用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Callable, List, Tuple
from dataclasses import dataclass


@dataclass
class LossConfig:
    """损失函数配置"""
    # 各损失函数的启用状态
    edit_quality_enabled: bool = True
    imperceptibility_enabled: bool = True
    robustness_enabled: bool = True
    feature_adv_enabled: bool = True
    freq_adv_enabled: bool = True
    
    # 各损失函数的权重
    edit_quality_weight: float = 0.6
    imperceptibility_weight: float = 0.2
    robustness_weight: float = 0.15
    feature_adv_weight: float = 0.4
    freq_adv_weight: float = 0.25
    
    # 约束参数
    linf_epsilon: float = 8.0 / 255.0
    lpips_epsilon: float = 0.1
    
    # 其他参数
    num_edit_steps: int = 5  # 编辑质量损失的时步数
    num_augmentations: int = 3  # 鲁棒性损失的增强数量
    num_velocity_timesteps: int = 5  # 速度场损失的时步数


class EditQualityLoss(nn.Module):
    """
    编辑质量退化损失
    
    目标: 最大化受保护图像和干净图像在编辑后的差异
    """
    
    def __init__(self, enabled: bool = True, weight: float = 0.6,
                 num_steps: int = 5):
        """
        Args:
            enabled: 是否启用此损失
            weight: 损失权重
            num_steps: 编辑时步数
        """
        super().__init__()
        self.enabled = enabled
        self.weight = weight
        self.num_steps = num_steps
    
    def forward(self, x_protected: torch.Tensor, x_clean: torch.Tensor,
                edit_fn: Callable, prompts: List[str]) -> torch.Tensor:
        """
        计算编辑质量损失
        
        Args:
            x_protected: 受保护的图像 [B, C, H, W]
            x_clean: 干净的原始图像 [B, C, H, W]
            edit_fn: 编辑函数，接收(image, prompt, timestep)并返回编辑后的图像
            prompts: 目标编辑提示列表
            
        Returns:
            编辑质量损失
        """
        if not self.enabled:
            return torch.tensor(0.0, device=x_protected.device)
        
        if len(prompts) == 0:
            return torch.tensor(0.0, device=x_protected.device)
        
        total_loss = 0.0
        
        # 采样时步
        timesteps = torch.linspace(0, 1, self.num_steps)
        
        for t in timesteps:
            for prompt in prompts:
                try:
                    # 编辑受保护和干净的图像
                    x_edited_protected = edit_fn(x_protected, prompt, t)
                    x_edited_clean = edit_fn(x_clean, prompt, t)
                    
                    # 计算L2距离
                    diff = torch.norm(x_edited_protected - x_edited_clean, p=2)
                    total_loss += diff
                except Exception as e:
                    print(f"编辑质量损失计算错误: {e}")
                    continue
        
        # 最大化差异（取负）
        num_terms = len(timesteps) * len(prompts)
        if num_terms > 0:
            loss = -total_loss / num_terms * self.weight
        else:
            loss = torch.tensor(0.0, device=x_protected.device)
        
        return loss


class ImperceptibilityLoss(nn.Module):
    """
    不可见性损失
    
    目标: 确保添加的噪声在视觉上难以察觉
    包含: L∞范数约束 + LPIPS损失
    """
    
    def __init__(self, enabled: bool = True, weight: float = 0.2,
                 linf_epsilon: float = 8.0/255.0, lpips_epsilon: float = 0.1,
                 lpips_model: Optional[nn.Module] = None):
        """
        Args:
            enabled: 是否启用此损失
            weight: 损失权重
            linf_epsilon: L∞约束
            lpips_epsilon: LPIPS约束
            lpips_model: LPIPS模型（可选）
        """
        super().__init__()
        self.enabled = enabled
        self.weight = weight
        self.linf_epsilon = linf_epsilon
        self.lpips_epsilon = lpips_epsilon
        self.lpips_model = lpips_model
    
    def forward(self, x_protected: torch.Tensor, 
                x_clean: torch.Tensor) -> torch.Tensor:
        """
        计算不可见性损失
        
        Args:
            x_protected: 受保护的图像 [B, C, H, W]
            x_clean: 干净的原始图像 [B, C, H, W]
            
        Returns:
            不可见性损失
        """
        if not self.enabled:
            return torch.tensor(0.0, device=x_protected.device)
        
        # L∞约束
        diff = x_protected - x_clean
        linf_dist = torch.norm(diff, p=float('inf'))
        
        # 如果超过约束，添加惩罚
        if linf_dist > self.linf_epsilon:
            linf_penalty = (linf_dist - self.linf_epsilon) ** 2
        else:
            linf_penalty = torch.tensor(0.0, device=x_protected.device)
        
        # LPIPS损失（如果提供了模型）
        lpips_loss = torch.tensor(0.0, device=x_protected.device)
        if self.lpips_model is not None:
            try:
                lpips_loss = self.lpips_model(x_protected, x_clean)
                # 如果超过约束，添加惩罚
                if lpips_loss > self.lpips_epsilon:
                    lpips_loss = (lpips_loss - self.lpips_epsilon) ** 2
            except Exception as e:
                print(f"LPIPS计算错误: {e}")
        
        total_loss = (linf_penalty + lpips_loss) * self.weight
        
        return total_loss


class RobustnessLoss(nn.Module):
    """
    鲁棒性损失
    
    目标: 确保防护对图像的轻微扰动（如压缩、噪声）保持有效
    """
    
    def __init__(self, enabled: bool = True, weight: float = 0.15,
                 num_augmentations: int = 3):
        """
        Args:
            enabled: 是否启用此损失
            weight: 损失权重
            num_augmentations: 增强的数量
        """
        super().__init__()
        self.enabled = enabled
        self.weight = weight
        self.num_augmentations = num_augmentations
    
    def _apply_augmentation(self, x: torch.Tensor, 
                           aug_type: str = 'random') -> torch.Tensor:
        """
        应用数据增强
        
        Args:
            x: 输入张量 [B, C, H, W]
            aug_type: 增强类型 ('jpeg', 'noise', 'blur', 'random')
            
        Returns:
            增强后的张量
        """
        if aug_type == 'jpeg' or aug_type == 'random':
            # JPEG压缩模拟（简化版）
            quality = np.random.randint(80, 95)
            # 这里可以使用真实的JPEG压缩库
            x = torch.clamp(x, 0, 1)
        
        if aug_type == 'noise' or aug_type == 'random':
            # 高斯噪声
            noise = torch.randn_like(x) * 0.01
            x = x + noise
            x = torch.clamp(x, 0, 1)
        
        if aug_type == 'blur' or aug_type == 'random':
            # 轻微模糊
            kernel = torch.tensor([[1., 2., 1.],
                                  [2., 4., 2.],
                                  [1., 2., 1.]], device=x.device) / 16.0
            kernel = kernel.unsqueeze(0).unsqueeze(0)
            
            # 对每个通道应用卷积
            C = x.shape[1]
            kernel = kernel.repeat(C, 1, 1, 1)
            x = F.conv2d(x, kernel, padding=1, groups=C)
            x = torch.clamp(x, 0, 1)
        
        return x
    
    def forward(self, x_protected: torch.Tensor,
                edit_fn: Callable, prompts: List[str]) -> torch.Tensor:
        """
        计算鲁棒性损失
        
        Args:
            x_protected: 受保护的图像 [B, C, H, W]
            edit_fn: 编辑函数
            prompts: 目标编辑提示列表
            
        Returns:
            鲁棒性损失
        """
        if not self.enabled:
            return torch.tensor(0.0, device=x_protected.device)
        
        if len(prompts) == 0:
            return torch.tensor(0.0, device=x_protected.device)
        
        total_loss = 0.0
        
        for _ in range(self.num_augmentations):
            # 应用增强
            x_aug = self._apply_augmentation(x_protected)
            
            for prompt in prompts:
                try:
                    # 编辑原始和增强的图像
                    x_edited_original = edit_fn(x_protected, prompt, 0.5)
                    x_edited_aug = edit_fn(x_aug, prompt, 0.5)
                    
                    # 计算差异
                    diff = torch.norm(x_edited_original - x_edited_aug, p=2)
                    total_loss += diff
                except Exception as e:
                    print(f"鲁棒性损失计算错误: {e}")
                    continue
        
        num_terms = self.num_augmentations * len(prompts)
        if num_terms > 0:
            loss = total_loss / num_terms * self.weight
        else:
            loss = torch.tensor(0.0, device=x_protected.device)
        
        return loss


class FeatureAdversarialLoss(nn.Module):
    """
    特征空间对抗损失
    
    目标: 最大化源和目标速度场在受保护图像上的差异
    """
    
    def __init__(self, enabled: bool = True, weight: float = 0.4):
        """
        Args:
            enabled: 是否启用此损失
            weight: 损失权重
        """
        super().__init__()
        self.enabled = enabled
        self.weight = weight
    
    def forward(self, x_protected: torch.Tensor,
                velocity_fn: Optional[Callable] = None) -> torch.Tensor:
        """
        计算特征空间对抗损失
        
        Args:
            x_protected: 受保护的图像 [B, C, H, W]
            velocity_fn: 速度场计算函数，接收(image)并返回速度场差异
            
        Returns:
            特征空间对抗损失
        """
        if not self.enabled or velocity_fn is None:
            return torch.tensor(0.0, device=x_protected.device)
        
        try:
            # 计算速度场差异
            vel_diff = velocity_fn(x_protected)
            
            # 最大化速度场差异（取负）
            loss = -vel_diff * self.weight
        except Exception as e:
            print(f"特征空间对抗损失计算错误: {e}")
            loss = torch.tensor(0.0, device=x_protected.device)
        
        return loss


class FrequencyAdversarialLoss(nn.Module):
    """
    频域对抗损失
    
    目标: 在高频域最大化受保护图像和干净图像的差异
    """
    
    def __init__(self, enabled: bool = True, weight: float = 0.25,
                 freq_cutoff: float = 0.3):
        """
        Args:
            enabled: 是否启用此损失
            weight: 损失权重
            freq_cutoff: 频率截断值
        """
        super().__init__()
        self.enabled = enabled
        self.weight = weight
        self.freq_cutoff = freq_cutoff
    
    def _create_frequency_mask(self, height: int, width: int,
                              device: torch.device) -> torch.Tensor:
        """创建频率掩码"""
        freq_y = torch.fft.fftfreq(height, device=device)[:, None]
        freq_x = torch.fft.fftfreq(width, device=device)[None, :]
        freq_dist = torch.sqrt(freq_x**2 + freq_y**2)
        mask = (freq_dist > self.freq_cutoff).float()
        return mask
    
    def forward(self, x_protected: torch.Tensor,
                x_clean: torch.Tensor) -> torch.Tensor:
        """
        计算频域对抗损失
        
        Args:
            x_protected: 受保护的图像 [B, C, H, W]
            x_clean: 干净的原始图像 [B, C, H, W]
            
        Returns:
            频域对抗损失
        """
        if not self.enabled:
            return torch.tensor(0.0, device=x_protected.device)
        
        try:
            B, C, H, W = x_protected.shape
            device = x_protected.device
            
            # 创建频率掩码
            mask = self._create_frequency_mask(H, W, device)
            
            total_loss = 0.0
            
            for b in range(B):
                for c in range(C):
                    # FFT变换
                    fft_protected = torch.fft.fft2(x_protected[b, c])
                    fft_clean = torch.fft.fft2(x_clean[b, c])
                    
                    # 计算幅度谱差异
                    mag_diff = torch.abs(torch.abs(fft_protected) - torch.abs(fft_clean))
                    
                    # 应用频率掩码
                    weighted_diff = mag_diff * mask
                    
                    # 计算加权差异
                    loss = torch.sum(weighted_diff) / (torch.sum(mask) + 1e-8)
                    total_loss += loss
            
            # 最大化频域差异（取负）
            loss = -total_loss / (B * C) * self.weight
        except Exception as e:
            print(f"频域对抗损失计算错误: {e}")
            loss = torch.tensor(0.0, device=x_protected.device)
        
        return loss


class CombinedLoss(nn.Module):
    """
    综合损失函数
    
    组合多个损失函数，支持灵活的配置
    """
    
    def __init__(self, config: LossConfig, 
                 lpips_model: Optional[nn.Module] = None):
        """
        Args:
            config: 损失函数配置
            lpips_model: LPIPS模型（可选）
        """
        super().__init__()
        self.config = config
        
        # 初始化各个损失函数
        self.edit_quality_loss = EditQualityLoss(
            enabled=config.edit_quality_enabled,
            weight=config.edit_quality_weight,
            num_steps=config.num_edit_steps
        )
        
        self.imperceptibility_loss = ImperceptibilityLoss(
            enabled=config.imperceptibility_enabled,
            weight=config.imperceptibility_weight,
            linf_epsilon=config.linf_epsilon,
            lpips_epsilon=config.lpips_epsilon,
            lpips_model=lpips_model
        )
        
        self.robustness_loss = RobustnessLoss(
            enabled=config.robustness_enabled,
            weight=config.robustness_weight,
            num_augmentations=config.num_augmentations
        )
        
        self.feature_adv_loss = FeatureAdversarialLoss(
            enabled=config.feature_adv_enabled,
            weight=config.feature_adv_weight
        )
        
        self.freq_adv_loss = FrequencyAdversarialLoss(
            enabled=config.freq_adv_enabled,
            weight=config.freq_adv_weight
        )
    
    def forward(self, x_protected: torch.Tensor, x_clean: torch.Tensor,
                edit_fn: Optional[Callable] = None,
                prompts: Optional[List[str]] = None,
                velocity_fn: Optional[Callable] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算综合损失
        
        Args:
            x_protected: 受保护的图像
            x_clean: 干净的原始图像
            edit_fn: 编辑函数（可选）
            prompts: 编辑提示列表（可选）
            velocity_fn: 速度场函数（可选）
            
        Returns:
            (总损失, 各损失分量字典)
        """
        if prompts is None:
            prompts = []
        
        losses = {}
        total_loss = torch.tensor(0.0, device=x_protected.device)
        
        # 编辑质量损失
        if self.config.edit_quality_enabled and edit_fn is not None:
            loss = self.edit_quality_loss(x_protected, x_clean, edit_fn, prompts)
            losses['edit_quality'] = loss
            total_loss = total_loss + loss
        
        # 不可见性损失
        if self.config.imperceptibility_enabled:
            loss = self.imperceptibility_loss(x_protected, x_clean)
            losses['imperceptibility'] = loss
            total_loss = total_loss + loss
        
        # 鲁棒性损失
        if self.config.robustness_enabled and edit_fn is not None:
            loss = self.robustness_loss(x_protected, edit_fn, prompts)
            losses['robustness'] = loss
            total_loss = total_loss + loss
        
        # 特征空间对抗损失
        if self.config.feature_adv_enabled and velocity_fn is not None:
            loss = self.feature_adv_loss(x_protected, velocity_fn)
            losses['feature_adv'] = loss
            total_loss = total_loss + loss
        
        # 频域对抗损失
        if self.config.freq_adv_enabled:
            loss = self.freq_adv_loss(x_protected, x_clean)
            losses['freq_adv'] = loss
            total_loss = total_loss + loss
        
        return total_loss, losses


if __name__ == "__main__":
    print("损失函数模块已实现")
    print("\n包含的损失函数:")
    print("1. EditQualityLoss: 编辑质量退化损失")
    print("2. ImperceptibilityLoss: 不可见性损失")
    print("3. RobustnessLoss: 鲁棒性损失")
    print("4. FeatureAdversarialLoss: 特征空间对抗损失")
    print("5. FrequencyAdversarialLoss: 频域对抗损失")
    print("6. CombinedLoss: 综合损失函数")
    print("\n所有损失函数都可通过LossConfig配置启用/禁用")
