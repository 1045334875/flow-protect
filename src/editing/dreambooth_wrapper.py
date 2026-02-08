import sys
import os
import shutil
import tempfile
import subprocess
from typing import Dict, Any
from ..interfaces import EditingMethod

class DreamboothWrapper(EditingMethod):
    def edit(self, 
             input_image_path: str, 
             output_image_path: str, 
             source_prompt: str, 
             target_prompt: str, 
             model_name: str = "sd1.4",
             **kwargs) -> Dict[str, Any]:
        
        # Dreambooth training script
        db_script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../modules/Diffusion-PID-Protection"))
        db_script = os.path.join(db_script_dir, "train_dreambooth.py")
        
        if not os.path.exists(db_script):
            return {"status": "failed", "error": f"Script not found: {db_script}"}

        # Create temp dir for instance data (input image)
        with tempfile.TemporaryDirectory() as temp_dir:
            instance_dir = os.path.join(temp_dir, "instance_data")
            output_model_dir = os.path.join(temp_dir, "model_out")
            os.makedirs(instance_dir)
            os.makedirs(output_model_dir)
            
            # Copy input image
            shutil.copy(input_image_path, os.path.join(instance_dir, os.path.basename(input_image_path)))
            
            # 1. Train Dreambooth
            # We use subprocess to call train_dreambooth.py
            # Note: This is computationally expensive and takes time.
            
            model_map = {
                "sd1.4": "CompVis/stable-diffusion-v1-4",
                "sd1.5": "runwayml/stable-diffusion-v1-5",
            }
            hf_model_id = model_map.get(model_name.lower(), "runwayml/stable-diffusion-v1-5")

            cmd = [
                sys.executable,
                "train_dreambooth.py",
                f"--pretrained_model_name_or_path={hf_model_id}",
                f"--instance_data_dir={instance_dir}",
                f"--output_dir={output_model_dir}",
                f"--instance_prompt={source_prompt}", # Use source prompt as instance prompt (e.g. "a photo of sks dog")
                "--resolution=512",
                "--train_batch_size=1",
                "--gradient_accumulation_steps=1",
                "--learning_rate=5e-6",
                "--lr_scheduler=constant",
                "--lr_warmup_steps=0",
                "--max_train_steps=400" # Reduced for speed in this example
            ]
            
            try:
                subprocess.run(cmd, cwd=db_script_dir, check=True)
                
                # 2. Generate Image using the trained model
                # We can use diffusers directly here since we have the model path
                
                from diffusers import StableDiffusionPipeline
                import torch
                
                pipe = StableDiffusionPipeline.from_pretrained(
                    output_model_dir, 
                    torch_dtype=torch.float16
                ).to("cuda" if torch.cuda.is_available() else "cpu")
                
                image = pipe(target_prompt, num_inference_steps=50).images[0]
                
                os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
                image.save(output_image_path)
                
                return {"status": "success", "model": "dreambooth_finetuned"}
                
            except Exception as e:
                return {"status": "failed", "error": str(e)}
