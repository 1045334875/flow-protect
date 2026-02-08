from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class ProtectionMethod(ABC):
    """
    图像保护方法的抽象基类（接口定义）。
    所有具体的保护方法（如 AtkPDM, Diff-Protect, PID）都必须继承这个类并实现 protect 方法。
    Abstract Base Class for Image Protection methods.
    """
    @abstractmethod
    def protect(self, 
                input_image_path: str, 
                output_image_path: str, 
                model_name: str = "sd1.4", 
                prompt: str = "",
                **kwargs) -> Dict[str, Any]:
        """
        对图像应用保护（加噪）。
        这是一个抽象方法，具体逻辑由子类实现。
        
        Args:
            input_image_path: 输入图像路径。
            output_image_path: 保护后图像的保存路径。
            model_name: 用于生成保护的模型 (sd1.4, sd3, flux)。
            prompt: 某些保护方法可能需要的文本提示。
            **kwargs: 其他特定于方法的参数。
            
        Returns:
            Dict: 包含保护过程元数据的字典。
        """
        pass

class EditingMethod(ABC):
    """
    图像编辑方法的抽象基类（接口定义）。
    所有具体的编辑方法（如 FlowEdit, Dreambooth）都必须继承这个类并实现 edit 方法。
    Abstract Base Class for Image Editing methods.
    """
    @abstractmethod
    def edit(self, 
             input_image_path: str, 
             output_image_path: str, 
             source_prompt: str, 
             target_prompt: str, 
             model_name: str = "sd1.4",
             **kwargs) -> Dict[str, Any]:
        """
        根据提示词编辑图像。
        这是一个抽象方法，具体逻辑由子类实现。
        
        Args:
            input_image_path: 要编辑的图像路径（通常是保护后的图像）。
            output_image_path: 编辑后图像的保存路径。
            source_prompt: 原图的描述。
            target_prompt: 期望的编辑效果描述。
            model_name: 用于编辑的模型 (e.g., sd3, flux)。
            **kwargs: 其他参数。
            
        Returns:
            Dict: 包含编辑过程元数据的字典。
        """
        pass

class Evaluator(ABC):
    """
    评估指标的抽象基类（接口定义）。
    定义了评估保护效果和编辑效果的标准接口。
    Abstract Base Class for Evaluation metrics.
    """
    @abstractmethod
    def evaluate_protection(self, original_image_path: str, protected_image_path: str) -> Dict[str, float]:
        """
        评估图像因保护而发生的变化（例如 PSNR, SSIM, LPIPS）。
        衡量“画质损失”或“隐蔽性”。
        """
        pass

    @abstractmethod
    def evaluate_editing(self, original_image_path: str, edited_image_path: str, target_prompt: str) -> Dict[str, float]:
        """
        评估编辑的质量（例如 CLIP score, 图像保真度）。
        衡量“编辑是否成功”以及“是否保留了原图结构”。
        """
        pass
