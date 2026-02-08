import sys
import os
import torch
from PIL import Image
from typing import Dict, Any

# Add module to path
MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../modules/FlowEdit"))
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

try:
    from FlowEdit_utils import FlowEditSD3, FlowEditFLUX
    from diffusers import StableDiffusion3Pipeline, FluxPipeline
except ImportError:
    FlowEditSD3 = None
    FlowEditFLUX = None

from ..interfaces import EditingMethod

class FlowEditWrapper(EditingMethod):
    def __init__(self):
        self.pipe = None
        self.current_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self, model_name: str):
        if self.pipe is not None and self.current_model == model_name:
            return

        if model_name.lower() == "sd3":
            self.pipe = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3-medium-diffusers", 
                torch_dtype=torch.float16
            ).to(self.device)
            self.current_model = "sd3"
        elif model_name.lower() == "flux":
            self.pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev", 
                torch_dtype=torch.float16
            ).to(self.device)
            self.current_model = "flux"
        else:
            raise ValueError(f"FlowEdit supports 'sd3' or 'flux', got {model_name}")

    def edit(self, 
             input_image_path: str, 
             output_image_path: str, 
             source_prompt: str, 
             target_prompt: str, 
             model_name: str = "sd3",
             **kwargs) -> Dict[str, Any]:
        
        if FlowEditSD3 is None:
             return {"status": "failed", "error": "FlowEdit dependencies missing"}

        self._load_model(model_name)
        
        # Load Image
        image = Image.open(input_image_path).convert("RGB")
        # Resize/Crop as in run_script.py
        width, height = image.size
        new_w = width - width % 16
        new_h = height - height % 16
        image = image.crop((0, 0, new_w, new_h))
        
        # Preprocess
        # This part mimics run_script.py logic
        # Note: We need to check if we can access pipe components like image_processor
        
        # Default Params
        T_steps = kwargs.get("T_steps", 50)
        n_avg = kwargs.get("n_avg", 1)
        src_guidance_scale = kwargs.get("src_guidance_scale", 1.5)
        tar_guidance_scale = kwargs.get("tar_guidance_scale", 5.5)
        n_min = kwargs.get("n_min", 0)
        n_max = kwargs.get("n_max", 24)
        negative_prompt = kwargs.get("negative_prompt", "")

        try:
            # Encode image to latent
            # Simplified logic based on run_script.py
            # Note: This requires access to VAE and internal methods which are available in pipeline
            
            image_src = self.pipe.image_processor.preprocess(image)
            image_src = image_src.to(self.device).half()
            
            with torch.no_grad():
                x0_src_denorm = self.pipe.vae.encode(image_src).latent_dist.mode()
            
            x0_src = (x0_src_denorm - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
            x0_src = x0_src.to(self.device)

            if model_name.lower() == "sd3":
                x0_tar = FlowEditSD3(
                    self.pipe, self.pipe.scheduler, x0_src, 
                    source_prompt, target_prompt, negative_prompt,
                    T_steps, n_avg, src_guidance_scale, tar_guidance_scale, n_min, n_max
                )
            else: # Flux
                x0_tar = FlowEditFLUX(
                    self.pipe, self.pipe.scheduler, x0_src, 
                    source_prompt, target_prompt,
                    T_steps, n_avg, src_guidance_scale, tar_guidance_scale, n_min, n_max
                )

            # Decode
            x0_tar = (x0_tar / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
            with torch.no_grad():
                image_tar = self.pipe.vae.decode(x0_tar, return_dict=False)[0]
            
            image_tar = self.pipe.image_processor.postprocess(image_tar, output_type="pil")[0]
            
            # Save
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            image_tar.save(output_image_path)
            
            return {"status": "success", "model": model_name}

        except Exception as e:
            return {"status": "failed", "error": str(e)}
