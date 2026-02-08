import sys
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

# Add module to path
MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../modules/AtkPDM"))
if MODULE_PATH not in sys.path:
    sys.path.insert(0, MODULE_PATH) # Insert at 0 to prioritize

# Import the function
try:
    # Since we added modules/AtkPDM to sys.path, we can import directly
    from atk_pdm import atk_pdm_protect_image
except ImportError as e:
    print(f"Failed to import atk_pdm_protect_image: {e}")
    atk_pdm_protect_image = None

from ..interfaces import ProtectionMethod

@dataclass
class AtkPDMArgs:
    protected_image_path: str
    save_folder_path: str
    VAE_model_id: str = "runwayml/stable-diffusion-v1-5"
    VAE_unet_size: int = 512
    victim_model_id: str = "runwayml/stable-diffusion-v1-5"
    victim_model_type: str = "sd" # Assuming 'sd' based on usage
    batch_size: int = 1
    step_size: float = 1.0
    local_rank: int = -1
    # Add other necessary args with defaults
    
class AtkPDMProtection(ProtectionMethod):
    def protect(self, 
                input_image_path: str, 
                output_image_path: str, 
                model_name: str = "sd1.4", 
                prompt: str = "",
                **kwargs) -> Dict[str, Any]:
        
        if atk_pdm_protect_image is None:
            raise ImportError("Could not import atk_pdm_protect_image from modules/AtkPDM")

        # Map model_name to HuggingFace ID
        model_map = {
            "sd1.4": "CompVis/stable-diffusion-v1-4",
            "sd1.5": "runwayml/stable-diffusion-v1-5",
            "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",
            "flux": "black-forest-labs/FLUX.1-dev"
        }
        
        hf_model_id = model_map.get(model_name.lower(), model_name)
        
        # Prepare Output Directory (AtkPDM expects a directory)
        output_dir = os.path.dirname(output_image_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Construct Args
        args = AtkPDMArgs(
            protected_image_path=input_image_path,
            save_folder_path=output_dir,
            victim_model_id=hf_model_id,
            VAE_model_id=hf_model_id if "flux" not in model_name else "black-forest-labs/FLUX.1-dev", # Flux VAE might differ
            **kwargs
        )
        
        # Run Protection
        # Note: AtkPDM might print a lot or take time.
        try:
            atk_pdm_protect_image(args)
            
            # AtkPDM saves to {save_folder_path}/protected_image/{img_name}.png
            # We need to ensure the file is at output_image_path
            img_name = os.path.splitext(os.path.basename(input_image_path))[0]
            generated_path = os.path.join(output_dir, "protected_image", f"{img_name}.png")
            
            if os.path.exists(generated_path) and generated_path != output_image_path:
                import shutil
                shutil.move(generated_path, output_image_path)
                
            return {"status": "success", "model": hf_model_id, "output": output_image_path}
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
