import os
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional
from ..interfaces import ProtectionMethod
from .flowedit_protection import ProtectiveNoiseOptimizer, NoiseConfig

class FlowEditProtectionWrapper(ProtectionMethod):
    """
    Wrapper for FlowEdit Protection methods (Frequency, Texture, Feature, Velocity).
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def protect(self, 
                input_image_path: str, 
                output_image_path: str, 
                model_name: str = "sd3", 
                prompt: str = "",
                **kwargs) -> Dict[str, Any]:
        """
        Apply FlowEdit protection to an image.
        
        Args:
            input_image_path: Path to input image
            output_image_path: Path to save protected image
            model_name: Model name (not used directly by noise generators, but maybe for feature extractor)
            prompt: Input prompt (not used by current noise generators)
            **kwargs: Configuration for NoiseConfig
                - freq_enabled: bool
                - texture_enabled: bool
                - feature_enabled: bool
                - velocity_enabled: bool
                - freq_weight: float
                - texture_weight: float
                - feature_weight: float
                - velocity_weight: float
                - eps: float (Linf epsilon, default 8.0)
        
        Returns:
            Dict with status and metadata
        """
        try:
            # Load image
            if not os.path.exists(input_image_path):
                return {"status": "failed", "error": f"Input image not found: {input_image_path}"}
            
            image = Image.open(input_image_path).convert('RGB')
            # Preprocess image
            image_array = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # Parse config from kwargs
            eps = kwargs.get('eps', 8.0)
            noise_config = NoiseConfig(
                freq_enabled=kwargs.get('freq_enabled', True),
                texture_enabled=kwargs.get('texture_enabled', True),
                feature_enabled=kwargs.get('feature_enabled', False), # Default false as it needs feature extractor
                velocity_enabled=kwargs.get('velocity_enabled', False), # Default false as it needs flow model
                
                freq_weight=kwargs.get('freq_weight', 0.5),
                texture_weight=kwargs.get('texture_weight', 0.5),
                feature_weight=kwargs.get('feature_weight', 0.0),
                velocity_weight=kwargs.get('velocity_weight', 0.0),
                
                linf_epsilon=eps / 255.0
            )
            
            # Initialize optimizer
            optimizer = ProtectiveNoiseOptimizer(noise_config)
            
            # TODO: If feature/velocity enabled, we need to set feature_extractor/flow_model
            # For now, we assume simple usage without external models unless provided
            
            # Generate protected image
            protected_tensor = optimizer.generate(image_tensor)
            
            # Save image
            protected_array = protected_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            protected_array = (np.clip(protected_array, 0, 1) * 255).astype(np.uint8)
            protected_image = Image.fromarray(protected_array)
            
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            protected_image.save(output_image_path)
            
            return {
                "status": "success",
                "method": "flowedit_protection",
                "config": {k: v for k, v in kwargs.items() if not k.startswith('_')}
            }
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
