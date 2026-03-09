"""
针对FlowEdit的对抗性保护性噪声生成方法 (v2)
包含四种方法和可配置的损失函数

四种方法:
1. FrequencyDomainNoise: 频域对抗噪声
2. FeatureSpaceAdversarialNoise: 特征空间对抗噪声  
3. MultiScaleTextureNoise: 多尺度纹理对抗噪声
4. VelocityFieldAdversarialNoise: 速度场对抗噪声

损失函数配置:
- 所有损失函数都可通过参数启用/禁用
- 支持动态权重调整
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Tuple, Callable
import torchvision.transforms as transforms
from dataclasses import dataclass
from enum import Enum


@dataclass
class NoiseConfig:
    """保护性噪声配置"""
    # 频域噪声参数
    freq_enabled: bool = True
    freq_cutoff: float = 0.3
    freq_amplitude: float = 0.1  # 增大振幅
    
    # 特征空间噪声参数
    feature_enabled: bool = True
    feature_lr: float = 0.01
    feature_iterations: int = 20
    feature_epsilon: float = 0.03
    
    # 纹理噪声参数
    texture_enabled: bool = True
    texture_num_scales: int = 4
    texture_num_orientations: int = 4
    texture_amplitude_range: Tuple[float, float] = (0.05, 0.15)  # 增大振幅
    
    # 速度场噪声参数
    velocity_enabled: bool = False  # 默认禁用，需要流模型
    velocity_lr: float = 0.01
    velocity_iterations: int = 30
    velocity_epsilon: float = 0.03
    velocity_num_timesteps: int = 5
    velocity_loss_strategy: str = 'max_diff'  # 'max_diff', 'orthogonal_source_target', 'orthogonal_guide_diff'
    
    # 综合参数
    freq_weight: float = 1.0  # 增大权重
    vertical_weight: float = 0.5
    feature_weight: float = 0.3
    texture_weight: float = 1.0  # 增大权重
    velocity_weight: float = 0.2
    
    # 约束参数
    linf_epsilon: float = 16.0 / 255.0  # 增大 L∞约束
    lpips_epsilon: float = 0.1  # LPIPS约束
    
    # 损失函数配置
    loss_config: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.loss_config is None:
            self.loss_config = {
                'edit_quality': True,
                'imperceptibility': True,
                'robustness': True,
                'feature_adv': True,
                'freq_adv': True,
            }


class FrequencyDomainNoise:
    """
    频域对抗噪声生成器
    
    原理: 在图像的频域高频部分添加噪声，干扰流模型的特征提取
    优势: 计算高效，人眼不易察觉
    """
    
    def __init__(self, freq_cutoff: float = 0.3, amplitude: float = 0.05):
        """
        Args:
            freq_cutoff: 频率截断值 (0-1)，>freq_cutoff的频率被认为是高频
            amplitude: 噪声幅度 (相对于图像范围)
        """
        self.freq_cutoff = freq_cutoff
        self.amplitude = amplitude
    
    def _create_frequency_mask(self, height: int, width: int, 
                               device: torch.device) -> torch.Tensor:
        """
        创建频率掩码，高频部分为1，低频部分为0
        
        Args:
            height: 图像高度
            width: 图像宽度
            device: 设备
            
        Returns:
            频率掩码 [H, W]
        """
        # 创建频率网格
        freq_y = torch.fft.fftfreq(height, device=device)[:, None]
        freq_x = torch.fft.fftfreq(width, device=device)[None, :]
        
        # 计算距离原点的频率距离
        freq_dist = torch.sqrt(freq_x**2 + freq_y**2)
        
        # 创建掩码：高频部分为1
        mask = (freq_dist > self.freq_cutoff).float()
        
        return mask
    
    def generate(self, image: torch.Tensor) -> torch.Tensor:
        """
        生成频域对抗噪声
        
        Args:
            image: 输入图像 [C, H, W] 或 [B, C, H, W]，值域[0,1]
            
        Returns:
            纯噪声 (不是加噪后的图像) [C, H, W] 或 [B, C, H, W]
        """
        # 处理批处理维度
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, C, H, W = image.shape
        device = image.device
        dtype = image.dtype
        
        # 创建频率掩码 (高频区域为1)
        mask = self._create_frequency_mask(H, W, device)
        
        # 生成高频噪声
        noise = torch.randn(B, C, H, W, device=device, dtype=dtype)
        
        # 对噪声进行FFT
        noise_fft = torch.fft.fft2(noise)
        
        # 只保留高频部分
        noise_fft = noise_fft * mask.unsqueeze(0).unsqueeze(0)
        
        # 逆FFT得到高频噪声
        high_freq_noise = torch.fft.ifft2(noise_fft).real
        
        # 归一化噪声到 [-amplitude, amplitude] 范围
        if high_freq_noise.abs().max() > 1e-8:
            high_freq_noise = high_freq_noise / high_freq_noise.abs().max() * self.amplitude
        
        if squeeze_output:
            high_freq_noise = high_freq_noise.squeeze(0)
        
        return high_freq_noise


class FeatureSpaceAdversarialNoise:
    """
    特征空间对抗噪声生成器
    
    原理: 通过梯度上升在特征空间中生成对抗噪声
    优势: 针对性强，可精确控制编辑失败的方式
    """
    
    def __init__(self, feature_extractor: Optional[nn.Module] = None,
                 target_layer: str = 'layer3',
                 lr: float = 0.01, num_iterations: int = 20, 
                 epsilon: float = 0.03):
        """
        Args:
            feature_extractor: 特征提取网络 (如ResNet)，如果为None则跳过此方法
            target_layer: 目标特征层名称
            lr: 学习率
            num_iterations: 迭代次数
            epsilon: L∞约束
        """
        self.feature_extractor = feature_extractor
        self.target_layer = target_layer
        self.lr = lr
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.features = None
        
        if feature_extractor is not None:
            self._register_hook()
    
    def _register_hook(self):
        """注册钩子以提取特定层的特征"""
        def hook(module, input, output):
            self.features = output.detach()
        
        try:
            # 获取目标层并注册钩子
            target_module = dict(self.feature_extractor.named_modules())[self.target_layer]
            target_module.register_forward_hook(hook)
        except KeyError:
            print(f"警告: 找不到层 {self.target_layer}")
    
    def _feature_loss(self, image: torch.Tensor, 
                     source_features: torch.Tensor) -> torch.Tensor:
        """
        计算特征空间损失
        最大化编辑后图像与源特征的距离
        """
        if self.feature_extractor is None:
            return torch.tensor(0.0, device=image.device)
        
        with torch.enable_grad():
            _ = self.feature_extractor(image)
            current_features = self.features
            
            if current_features is None:
                return torch.tensor(0.0, device=image.device)
            
            # 最大化特征距离
            loss = -torch.norm(current_features - source_features, p=2)
        
        return loss
    
    def generate(self, image: torch.Tensor, 
                source_image: torch.Tensor) -> torch.Tensor:
        """
        生成特征空间对抗噪声
        
        Args:
            image: 输入图像 [C, H, W] 或 [B, C, H, W]
            source_image: 源图像，用于提取源特征
            
        Returns:
            添加噪声后的图像
        """
        if self.feature_extractor is None:
            return image.clone()
        
        # 处理批处理维度
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        device = image.device
        dtype = image.dtype
        
        # 提取源特征
        with torch.no_grad():
            _ = self.feature_extractor(source_image)
            source_features = self.features.clone() if self.features is not None else None
        
        if source_features is None:
            if squeeze_output:
                return image.squeeze(0)
            return image
        
        # 初始化对抗噪声
        delta = torch.zeros_like(image, requires_grad=True, device=device, dtype=dtype)
        optimizer = torch.optim.Adam([delta], lr=self.lr)
        
        # 迭代优化
        for iteration in range(self.num_iterations):
            optimizer.zero_grad()
            
            # 计算对抗图像
            adv_image = image + delta
            adv_image = torch.clamp(adv_image, 0, 1)
            
            # 计算损失
            loss = self._feature_loss(adv_image, source_features)
            
            # 反向传播
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
            
            # 投影到约束空间
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
        
        # 生成最终的对抗图像
        noisy_image = torch.clamp(image + delta.detach(), 0, 1).to(dtype)
        
        if squeeze_output:
            noisy_image = noisy_image.squeeze(0)
        
        return noisy_image


class MultiScaleTextureNoise:
    """
    多尺度纹理对抗噪声生成器
    
    原理: 在多个尺度上添加Gabor滤波器生成的纹理噪声
    优势: 能有效破坏图像的局部和全局结构
    """
    
    def __init__(self, num_scales: int = 4, num_orientations: int = 4,
                 amplitude_range: Tuple[float, float] = (0.02, 0.08)):
        """
        Args:
            num_scales: 金字塔尺度数
            num_orientations: Gabor滤波器方向数
            amplitude_range: 幅度范围 (最小, 最大)
        """
        self.num_scales = num_scales
        self.num_orientations = num_orientations
        self.amplitude_range = amplitude_range
    
    def _create_gabor_filter(self, size: int, wavelength: float, 
                            orientation: float, sigma: float = 3.0,
                            device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        创建Gabor滤波器
        
        Args:
            size: 滤波器大小
            wavelength: 波长
            orientation: 方向 (弧度)
            sigma: 高斯包络的标准差
            device: 设备
            
        Returns:
            Gabor滤波器 [size, size]
        """
        # 创建坐标网格
        # Ensure size is odd for symmetry
        if size % 2 == 0:
            size += 1
            
        # Use linspace to guarantee exact size
        x = torch.linspace(-(size//2), size//2, size, device=device)
        y = torch.linspace(-(size//2), size//2, size, device=device)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        
        # 旋转坐标
        X_theta = X * torch.cos(torch.tensor(orientation, device=device)) + \
                  Y * torch.sin(torch.tensor(orientation, device=device))
        Y_theta = -X * torch.sin(torch.tensor(orientation, device=device)) + \
                   Y * torch.cos(torch.tensor(orientation, device=device))
        
        # Gabor函数
        gabor = torch.exp(-0.5 * (X_theta**2 + Y_theta**2) / (sigma**2)) * \
                torch.cos(2 * np.pi * X_theta / wavelength)
        
        # 归一化
        gabor = gabor / (torch.sum(torch.abs(gabor)) + 1e-8)
        
        return gabor
    
    def _generate_laplacian_pyramid(self, image: torch.Tensor, 
                                    num_levels: int) -> List[torch.Tensor]:
        """
        生成拉普拉斯金字塔
        
        Args:
            image: 输入图像 [C, H, W]
            num_levels: 金字塔级数
            
        Returns:
            拉普拉斯金字塔 [L0, L1, ..., Ln]
        """
        pyramid = []
        current = image.clone()
        
        for _ in range(num_levels - 1):
            # 下采样
            downsampled = F.avg_pool2d(current.unsqueeze(0), kernel_size=2, stride=2).squeeze(0)
            
            # 上采样回原大小
            upsampled = F.interpolate(downsampled.unsqueeze(0), 
                                     size=current.shape[-2:], 
                                     mode='bilinear', align_corners=False).squeeze(0)
            
            # 计算拉普拉斯
            laplacian = current - upsampled
            pyramid.append(laplacian)
            
            current = downsampled
        
        pyramid.append(current)  # 最后一层是高斯金字塔的顶层
        
        return pyramid
    
    def _reconstruct_from_pyramid(self, pyramid: List[torch.Tensor]) -> torch.Tensor:
        """
        从拉普拉斯金字塔重建图像
        
        Args:
            pyramid: 拉普拉斯金字塔
            
        Returns:
            重建的图像
        """
        current = pyramid[-1]
        
        for level in reversed(pyramid[:-1]):
            # 上采样
            current = F.interpolate(current.unsqueeze(0), 
                                   size=level.shape[-2:],
                                   mode='bilinear', align_corners=False).squeeze(0)
            # 添加拉普拉斯
            current = current + level
        
        return current
    
    def generate(self, image: torch.Tensor) -> torch.Tensor:
        """
        生成多尺度纹理对抗噪声
        
        Args:
            image: 输入图像 [C, H, W] 或 [B, C, H, W]
            
        Returns:
            添加噪声后的图像
        """
        # 处理批处理维度
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, C, H, W = image.shape
        device = image.device
        dtype = image.dtype
        
        noisy_image = image.clone()
        
        for b in range(B):
            # 生成拉普拉斯金字塔
            pyramid = self._generate_laplacian_pyramid(image[b], self.num_scales)
            
            # 对每一层添加纹理噪声
            noisy_pyramid = []
            for scale_idx, level in enumerate(pyramid):
                # 计算该尺度的幅度
                min_amp, max_amp = self.amplitude_range
                amplitude = min_amp + (max_amp - min_amp) * (1 - scale_idx / self.num_scales)
                
                # 生成纹理噪声
                noise = torch.zeros_like(level)
                
                level_h, level_w = level.shape[-2:]
                
                for c in range(C):
                    for orient_idx in range(self.num_orientations):
                        # 创建Gabor滤波器
                        orientation = orient_idx * np.pi / self.num_orientations
                        filter_size = min(2 ** (scale_idx + 3), 31)  # 滤波器大小随尺度增加
                        
                        # Ensure filter size is odd for symmetric padding
                        if filter_size % 2 == 0:
                            filter_size += 1
                        
                        gabor = self._create_gabor_filter(filter_size, 
                                                         wavelength=filter_size/2,
                                                         orientation=orientation,
                                                         device=device)
                        
                        # 生成随机相位的响应
                        random_phase = np.random.uniform(0, 2*np.pi)
                        response = amplitude * np.sin(random_phase) * gabor
                        
                        # 应用卷积
                        level_c = level[c:c+1, :, :].unsqueeze(0)  # [1, 1, H, W]
                        response_2d = response.unsqueeze(0).unsqueeze(0)  # [1, 1, size, size]
                        
                        # Use same padding explicitly
                        padding = filter_size // 2
                        filtered = F.conv2d(level_c, response_2d, padding=padding)
                        
                        # Debug info if mismatch
                        if filtered.shape[-2:] != level.shape[-2:]:
                            # Force crop or pad to match
                            fh, fw = filtered.shape[-2:]
                            lh, lw = level.shape[-2:]
                            
                            # Crop if larger
                            if fh > lh: filtered = filtered[..., :lh, :]
                            if fw > lw: filtered = filtered[..., :, :lw]
                            
                            # Pad if smaller
                            if fh < lh or fw < lw:
                                pad_h = max(0, lh - fh)
                                pad_w = max(0, lw - fw)
                                # Pad right and bottom
                                filtered = F.pad(filtered, (0, pad_w, 0, pad_h))
                        
                        noise[c] += filtered.squeeze()[:level_h, :level_w]
                
                noisy_level = level + noise
                noisy_pyramid.append(noisy_level)
            
            # 重建图像
            noisy_img = self._reconstruct_from_pyramid(noisy_pyramid)
            
            # 归一化到[0, 1]
            noisy_img = torch.clamp(noisy_img, 0, 1).to(dtype)
            noisy_image[b] = noisy_img
        
        if squeeze_output:
            noisy_image = noisy_image.squeeze(0)
        
        return noisy_image


class VelocityFieldAdversarialNoise:
    """
    速度场对抗噪声生成器
    
    原理: 直接对抗FlowEdit的核心机制：速度场差异
    优势: 理论基础最强，直接攻击编辑过程的核心
    
    注意: 此方法需要访问流模型的内部结构，实现较复杂
    """
    
    def __init__(self, flow_model: Optional[nn.Module] = None,
                 lr: float = 0.01, num_iterations: int = 30,
                 epsilon: float = 0.03, num_timesteps: int = 5,
                 loss_strategy: str = 'max_diff'):
        """
        Args:
            flow_model: 预训练的流模型 (FLUX或SD3)，如果为None则跳过此方法
            lr: 学习率
            num_iterations: 迭代次数
            epsilon: L∞约束
            num_timesteps: 采样的时步数
            loss_strategy: 优化策略 ('max_diff', 'orthogonal_source_target', 'orthogonal_guide_diff')
        """
        self.flow_model = flow_model
        self.lr = lr
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.num_timesteps = num_timesteps
        self.loss_strategy = loss_strategy
    
    def generate(self, latents: torch.Tensor,
                src_prompt_embeds: Optional[torch.Tensor] = None,
                tar_prompt_embeds: Optional[torch.Tensor] = None,
                src_pooled_prompt_embeds: Optional[torch.Tensor] = None,
                tar_pooled_prompt_embeds: Optional[torch.Tensor] = None,
                src_text_ids: Optional[torch.Tensor] = None,
                tar_text_ids: Optional[torch.Tensor] = None,
                model_type: str = 'sd3') -> torch.Tensor:
        """
        生成速度场对抗噪声
        
        Args:
            latents: 潜在表示 [B, C, H, W]
            src_prompt_embeds: 源提示嵌入 (必需)
            tar_prompt_embeds: 目标提示嵌入 (用于 orthogonal_guide_diff 策略)
            src_pooled_prompt_embeds: 源池化提示嵌入 (SD3/Flux)
            tar_pooled_prompt_embeds: 目标池化提示嵌入 (SD3/Flux)
            src_text_ids: 源文本ID (Flux必需)
            tar_text_ids: 目标文本ID (Flux必需)
            model_type: 模型类型 ('sd3' 或 'flux')
            
        Returns:
            添加噪声后的潜在表示
        """
        if self.flow_model is None:
            return latents.clone()
            
        # 复制潜在表示并启用梯度
        adv_latents = latents.clone().detach().requires_grad_(True)
        clean_latents = latents.clone().detach()
        optimizer = torch.optim.Adam([adv_latents], lr=self.lr)
        
        # 获取时间步调度器
        scheduler = self.flow_model.scheduler
        num_train_timesteps = scheduler.config.num_train_timesteps
        timesteps = np.linspace(0, num_train_timesteps - 1, self.num_timesteps).astype(int)
        timesteps = torch.from_numpy(timesteps).to(latents.device).flip(0) # 从大到小
        
        # Flux 准备工作
        flux_img_ids = None
        flux_params = {}
        if model_type == 'flux':
            # 获取 latent_image_ids
            # 假设 flow_model 是 FluxPipeline
            height, width = latents.shape[-2:]
            # 需要原始高度宽度，因为 latents 是缩放后的
            # 这里假设 pipe.vae_scale_factor 可用
            vae_scale_factor = getattr(self.flow_model, 'vae_scale_factor', 8)
            orig_height = height * vae_scale_factor
            orig_width = width * vae_scale_factor
            
            # 使用 prepare_latents 获取 IDs (不做 VAE encode)
            # 注意：prepare_latents 可能返回 tuple (latents, ids)
            # 我们只需要 ids
            try:
                num_channels_latents = latents.shape[1]
                _, flux_img_ids = self.flow_model.prepare_latents(
                    batch_size=latents.shape[0],
                    num_channels_latents=num_channels_latents,
                    height=orig_height,
                    width=orig_width,
                    dtype=latents.dtype,
                    device=latents.device,
                    generator=None,
                    latents=latents # 传入 latents 以跳过编码
                )
            except Exception as e:
                print(f"Warning: Failed to prepare Flux latents: {e}")
                return latents # 无法继续
                
            flux_params['img_ids'] = flux_img_ids
            flux_params['orig_height'] = orig_height
            flux_params['orig_width'] = orig_width
            
            # Guidance
            # Flux 通常需要 guidance
            # 简单起见，使用固定 guidance scale
            flux_guidance = torch.tensor([1.5], device=latents.device).expand(latents.shape[0])
            flux_params['guidance'] = flux_guidance
            
            if src_text_ids is None:
                print("Warning: Flux model requires src_text_ids")
                return latents
        
        print(f"Running Velocity Attack with strategy: {self.loss_strategy} on {model_type}")
        
        for iteration in range(self.num_iterations):
            optimizer.zero_grad()
            total_loss = 0
            
            # 使用相同的噪声进行采样，确保比较的公平性
            noise = torch.randn_like(adv_latents)
            
            for t in timesteps:
                t_float = t.float()
                # t 归一化到 [0, 1]
                t_norm = t_float / num_train_timesteps
                timestep = t_float.expand(latents.shape[0])
                
                # 1. 计算 Clean Latents 的状态和速度场 (Source V)
                zt_clean = (1 - t_norm) * clean_latents + t_norm * noise
                
                # 只有在不需要梯度时才使用 no_grad，但在对抗攻击中，我们只需要对 adv_latents 求导
                # source_v 不需要梯度
                with torch.no_grad():
                    if model_type == 'sd3':
                        source_v = self.flow_model.transformer(
                            hidden_states=zt_clean,
                            timestep=timestep,
                            encoder_hidden_states=src_prompt_embeds,
                            pooled_projections=src_pooled_prompt_embeds,
                            return_dict=False,
                        )[0]
                        
                        # 如果需要计算编辑方向 (Guide Vector)，还需要计算目标 Prompt 下的 Clean V
                        guide_v = None
                        if self.loss_strategy == 'orthogonal_guide_diff' and tar_prompt_embeds is not None:
                             # SD3 安全检查：确保有 pooled embeddings
                            tar_pooled = tar_pooled_prompt_embeds if tar_pooled_prompt_embeds is not None else src_pooled_prompt_embeds
                            
                            target_v_clean = self.flow_model.transformer(
                                hidden_states=zt_clean,
                                timestep=timestep,
                                encoder_hidden_states=tar_prompt_embeds,
                                pooled_projections=tar_pooled,
                                return_dict=False,
                            )[0]
                            # 编辑方向向量：从源指向目标
                            guide_v = target_v_clean - source_v
                    elif model_type == 'flux':
                        # Flux Forward
                        # 需要先 pack latents
                        num_channels = latents.shape[1]
                        zt_clean_packed = self.flow_model._pack_latents(
                            zt_clean, zt_clean.shape[0], num_channels, 
                            zt_clean.shape[2], zt_clean.shape[3]
                        )
                        
                        source_v_packed = self.flow_model.transformer(
                            hidden_states=zt_clean_packed,
                            timestep=timestep / 1000, # Flux timestep scaling
                            guidance=flux_params['guidance'],
                            encoder_hidden_states=src_prompt_embeds,
                            txt_ids=src_text_ids,
                            img_ids=flux_params['img_ids'],
                            pooled_projections=src_pooled_prompt_embeds,
                            return_dict=False,
                        )[0]
                        
                        # Unpack source_v for consistent loss calculation (optional, but good for L2)
                        # source_v = self.flow_model._unpack_latents(
                        #    source_v_packed, flux_params['orig_height'], flux_params['orig_width'], vae_scale_factor
                        # )
                        # 或者直接在 packed 空间计算 loss，更高效
                        source_v = source_v_packed
                        
                        guide_v = None
                        if self.loss_strategy == 'orthogonal_guide_diff' and tar_prompt_embeds is not None:
                            tar_pooled = tar_pooled_prompt_embeds if tar_pooled_prompt_embeds is not None else src_pooled_prompt_embeds
                            target_v_packed = self.flow_model.transformer(
                                hidden_states=zt_clean_packed,
                                timestep=timestep / 1000,
                                guidance=flux_params['guidance'],
                                encoder_hidden_states=tar_prompt_embeds,
                                txt_ids=tar_text_ids if tar_text_ids is not None else src_text_ids,
                                img_ids=flux_params['img_ids'],
                                pooled_projections=tar_pooled,
                                return_dict=False,
                            )[0]
                            guide_v = target_v_packed - source_v

                # 2. 计算 Adv Latents 的状态和速度场 (Target V / Adv V)
                zt_adv = (1 - t_norm) * adv_latents + t_norm * noise
                
                if model_type == 'sd3':
                    adv_v = self.flow_model.transformer(
                        hidden_states=zt_adv,
                        timestep=timestep,
                        encoder_hidden_states=src_prompt_embeds,
                        pooled_projections=src_pooled_prompt_embeds,
                        return_dict=False,
                    )[0]
                elif model_type == 'flux':
                    # Pack adv latents
                    # 注意：_pack_latents 操作是可微的 (reshape/transpose)
                    zt_adv_packed = self.flow_model._pack_latents(
                        zt_adv, zt_adv.shape[0], num_channels,
                        zt_adv.shape[2], zt_adv.shape[3]
                    )
                    
                    adv_v_packed = self.flow_model.transformer(
                        hidden_states=zt_adv_packed,
                        timestep=timestep / 1000,
                        guidance=flux_params['guidance'],
                        encoder_hidden_states=src_prompt_embeds,
                        txt_ids=src_text_ids,
                        img_ids=flux_params['img_ids'],
                        pooled_projections=src_pooled_prompt_embeds,
                        return_dict=False,
                    )[0]
                    adv_v = adv_v_packed
                
                # 3. 计算 Loss
                loss = torch.tensor(0.0, device=latents.device)
                
                if self.loss_strategy == 'max_diff':
                    # 策略1: 最大化 L2 距离 (让对抗样本的速度场尽可能远离原始速度场)
                    loss = -torch.norm(adv_v - source_v, p=2)
                    
                elif self.loss_strategy == 'orthogonal_source_target':
                    # 策略2: 正交化 (让对抗样本的速度场与原始速度场正交)
                    # Cosine Similarity = (A . B) / (|A| * |B|)
                    # 我们希望 Abs(Cos) 最小 (趋近于0)
                    
                    # Flatten vectors for dot product: [B, C, H, W] -> [B, -1]
                    v1 = adv_v.reshape(adv_v.shape[0], -1)
                    v2 = source_v.reshape(source_v.shape[0], -1)
                    
                    # Normalize
                    v1_norm = F.normalize(v1, p=2, dim=1)
                    v2_norm = F.normalize(v2, p=2, dim=1)
                    
                    # Cosine similarity (Dot product of normalized vectors)
                    cos_sim = torch.sum(v1_norm * v2_norm, dim=1)
                    
                    # Minimize absolute cosine similarity (Orthogonal)
                    loss = torch.mean(torch.abs(cos_sim)) * 10.0  # Scale up
                    
                elif self.loss_strategy == 'orthogonal_guide_diff':
                    # 策略3: 与编辑方向正交 (让对抗样本的速度场与“原始-目标”编辑方向正交)
                    if guide_v is not None:
                        v1 = adv_v.reshape(adv_v.shape[0], -1)
                        v2 = guide_v.reshape(guide_v.shape[0], -1) # Guide vector
                        
                        v1_norm = F.normalize(v1, p=2, dim=1)
                        v2_norm = F.normalize(v2, p=2, dim=1)
                        
                        cos_sim = torch.sum(v1_norm * v2_norm, dim=1)
                        loss = torch.mean(torch.abs(cos_sim)) * 10.0
                    else:
                        # Fallback if no target prompt provided
                        loss = -torch.norm(adv_v - source_v, p=2)
                
                total_loss += loss
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            # 投影
            with torch.no_grad():
                delta = adv_latents - latents
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                adv_latents.data = latents + delta
                
        return adv_latents.detach()


class ProtectiveNoiseOptimizer:
    """
    保护性噪声优化器
    
    支持多种损失函数的灵活配置
    """
    
    def __init__(self, config: NoiseConfig):
        """
        Args:
            config: 噪声配置
        """
        self.config = config
        
        # 初始化噪声生成器
        self.freq_noise = FrequencyDomainNoise(
            freq_cutoff=config.freq_cutoff,
            amplitude=config.freq_amplitude
        ) if config.freq_enabled else None
        
        self.texture_noise = MultiScaleTextureNoise(
            num_scales=config.texture_num_scales,
            num_orientations=config.texture_num_orientations,
            amplitude_range=config.texture_amplitude_range
        ) if config.texture_enabled else None
        
        self.feature_noise = FeatureSpaceAdversarialNoise(
            feature_extractor=None,  # 需要外部提供
            lr=config.feature_lr,
            num_iterations=config.feature_iterations,
            epsilon=config.feature_epsilon
        ) if config.feature_enabled else None
        
        self.velocity_noise = VelocityFieldAdversarialNoise(
            flow_model=None,  # 需要外部提供
            lr=config.velocity_lr,
            num_iterations=config.velocity_iterations,
            epsilon=config.velocity_epsilon,
            num_timesteps=config.velocity_num_timesteps,
            loss_strategy=config.velocity_loss_strategy
        ) if config.velocity_enabled else None
    
    def set_feature_extractor(self, feature_extractor: nn.Module):
        """设置特征提取器"""
        if self.feature_noise is not None:
            self.feature_noise.feature_extractor = feature_extractor
            self.feature_noise._register_hook()
    
    def set_flow_model(self, flow_model: nn.Module):
        """设置流模型"""
        if self.velocity_noise is not None:
            self.velocity_noise.flow_model = flow_model
    
    def generate(self, image: torch.Tensor,
                source_image: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        生成综合保护性噪声
        
        Args:
            image: 输入图像 [C, H, W] 或 [B, C, H, W]
            source_image: 源图像（用于特征空间噪声）
            
        Returns:
            添加综合噪声后的图像
        """
        # 累积噪声
        total_noise = torch.zeros_like(image)
        noise_count = 0
        
        # 频域噪声 - 现在直接返回纯噪声
        if self.config.freq_enabled and self.freq_noise is not None and \
           self.config.freq_weight > 0:
            freq_noise = self.freq_noise.generate(image)
            total_noise = total_noise + freq_noise * self.config.freq_weight
            noise_count += 1
        
        # 纹理噪声 - 还是返回加噪图像，需要提取噪声
        if self.config.texture_enabled and self.texture_noise is not None and \
           self.config.texture_weight > 0:
            texture_noisy = self.texture_noise.generate(image)
            texture_noise = texture_noisy - image  # 提取噪声部分
            total_noise = total_noise + texture_noise * self.config.texture_weight
            noise_count += 1
        
        # 特征空间噪声
        if self.config.feature_enabled and self.feature_noise is not None and \
           self.config.feature_weight > 0 and source_image is not None:
            feature_noisy = self.feature_noise.generate(image, source_image)
            feature_noise = feature_noisy - image  # 提取噪声部分
            total_noise = total_noise + feature_noise * self.config.feature_weight
            noise_count += 1
        
        # 速度场噪声
        if self.config.velocity_enabled and self.velocity_noise is not None and \
           self.config.velocity_weight > 0:
            # 需要潜在空间表示
            # 这是一个简化的假设：我们假设输入的 image 已经是 latent 或者可以被转换为 latent
            # 在实际使用中，我们需要从外部传入 VAE 和 Encoder
            # 目前暂时跳过这个复杂的集成步骤，或者需要重构接口
            print("警告: VelocityFieldAdversarialNoise 需要在潜在空间操作，当前接口暂不支持直接图像输入")
            # velocity_noisy = self.velocity_noise.generate(image)
            # velocity_noise = velocity_noisy - image  # 提取噪声部分
            # total_noise = total_noise + velocity_noise * self.config.velocity_weight
            # noise_count += 1
            pass
        
        # 如果没有任何噪声，直接返回原图
        if noise_count == 0:
            return image
        
        # 应用 L∞ 约束
        eps = self.config.linf_epsilon
        total_noise = torch.clamp(total_noise, -eps, eps)
        
        # 添加噪声到原图
        noisy_image = image + total_noise
        noisy_image = torch.clamp(noisy_image, 0, 1)
        
        return noisy_image


if __name__ == "__main__":
    print("保护性噪声生成方法已实现 (v2)")
    print("\n四种方法:")
    print("1. FrequencyDomainNoise: 频域对抗噪声")
    print("2. FeatureSpaceAdversarialNoise: 特征空间对抗噪声")
    print("3. MultiScaleTextureNoise: 多尺度纹理对抗噪声")
    print("4. VelocityFieldAdversarialNoise: 速度场对抗噪声")
    print("\n所有损失函数都可通过NoiseConfig配置启用/禁用")
