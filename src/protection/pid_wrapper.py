import sys
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Dict, Any

# Add module to path
MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../modules/Diffusion-PID-Protection"))
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

try:
    from PID import main as pid_main
except ImportError:
    pid_main = None

from ..interfaces import ProtectionMethod

@dataclass
class PIDArgs:
    pretrained_model_name_or_path: str
    instance_data_dir: str
    output_dir: str
    resolution: int = 512
    center_crop: bool = False
    train_batch_size: int = 1
    dataloader_num_workers: int = 0
    eps: float = 12.75
    step_size: float = 1/255
    attack_type: str = "add"
    seed: int = None
    revision: str = None
    # Add other args as needed
    max_train_steps: int = 100 # Default fallback
    local_rank: int = -1
    enable_xformers_memory_efficient_attention: bool = False

class PIDProtection(ProtectionMethod):
    def protect(self, 
                input_image_path: str, 
                output_image_path: str, 
                model_name: str = "sd1.4", 
                prompt: str = "",
                **kwargs) -> Dict[str, Any]:
        
        if pid_main is None:
            raise ImportError("Could not import main from modules/Diffusion-PID-Protection/PID.py")

        model_map = {
            "sd1.4": "CompVis/stable-diffusion-v1-4",
            "sd1.5": "runwayml/stable-diffusion-v1-5",
            "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",
            "flux": "black-forest-labs/FLUX.1-dev"
        }
        hf_model_id = model_map.get(model_name.lower(), model_name)

        # PID works on a directory of images. Create a temp input dir.
        with tempfile.TemporaryDirectory() as temp_input_dir:
            # Copy input image to temp dir
            img_name = os.path.basename(input_image_path)
            shutil.copy(input_image_path, os.path.join(temp_input_dir, img_name))
            
            # Create temp output dir (or use specific output dir)
            temp_output_dir = os.path.join(os.path.dirname(output_image_path), "pid_temp_out")
            os.makedirs(temp_output_dir, exist_ok=True)

            args = PIDArgs(
                pretrained_model_name_or_path=hf_model_id,
                instance_data_dir=temp_input_dir,
                output_dir=temp_output_dir,
                **kwargs
            )
            
            try:
                # Run PID
                # Note: PID.py might not return anything, just save files.
                # We need to capture where it saves.
                # Looking at PID.py (previously read), it seems to iterate and save.
                # It saves to args.output_dir usually.
                pid_main(args)
                
                # Find result. PID usually saves with some naming convention.
                # If we assume it saves the image in output_dir
                # We need to find the file and move it to output_image_path.
                
                # Check output dir content
                files = os.listdir(temp_output_dir)
                if not files:
                     return {"status": "failed", "error": "No output file generated"}
                
                # Assuming the first file is the result (since we only processed one image)
                generated_file = os.path.join(temp_output_dir, files[0])
                shutil.move(generated_file, output_image_path)
                
                return {"status": "success", "model": hf_model_id}
                
            except Exception as e:
                return {"status": "failed", "error": str(e)}
            finally:
                if os.path.exists(temp_output_dir):
                    shutil.rmtree(temp_output_dir)
