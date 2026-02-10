import sys
import os
import shutil
import tempfile
import subprocess
from typing import Dict, Any
from ..interfaces import ProtectionMethod

class DiffProtectWrapper(ProtectionMethod):
    """
    Wrapper for Diff-Protect (Mist) protection methods.
    Supports multiple attack modes: advdm, texture_only, mist, sds, sdsT
    """
    
    def __init__(self):
        self.code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../modules/Diff-Protect/code"))
        self.config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../modules/Diff-Protect/configs"))
        
    def protect(self, 
                input_image_path: str, 
                output_image_path: str, 
                model_name: str = "sd1.4", 
                prompt: str = "",
                attack_mode: str = "mist",  # advdm, texture_only, mist, sds, sdsT
                epsilon: float = 16.0,
                steps: int = 100,
                alpha: float = 1.0,
                input_size: int = 512,
                g_mode: str = "+",  # "+" or "-"
                using_target: bool = False,
                target_rate: int = 5,
                device: int = 0,
                **kwargs) -> Dict[str, Any]:
        
        """
        Apply Diff-Protect protection to an image.
        
        Diff-Protect computes loss through a target_model that combines:
        1. Semantic loss: Measures how well the image matches the text prompt
        2. Textural loss: Measures distance in latent space between input and target
        3. Fused loss: Combines both semantic and textural losses
        
        For SD1.4, the loss computation happens in the target_model.forward() method:
        - Mode 'advdm': Uses semantic loss only (negative gradient direction)
        - Mode 'texture_only': Uses textural loss only (MSE in latent space)
        - Mode 'mist': Combines both losses with configurable weights
        - Mode 'sds': Uses score distillation for faster convergence
        
        Args:
            input_image_path: Path to input image
            output_image_path: Path to save protected image
            model_name: Model name (sd1.4, sd1.5, sd3, flux) - maps to checkpoint
            prompt: Input prompt (used for semantic guidance in latent space)
            attack_mode: Attack mode - advdm, texture_only, mist, sds, sdsT
            epsilon: Perturbation budget (l_inf norm, typically 8-16)
            steps: Number of PGD attack iterations
            alpha: Step size for each PGD iteration
            input_size: Input image size (default 512x512)
            g_mode: Gradient direction ("+" for maximize, "-" for minimize)
            using_target: Whether to use target guidance for targeted attacks
            target_rate: Weight for target loss component (higher = more emphasis on semantic)
            device: GPU device ID
            **kwargs: Additional parameters (diff_pgd, etc.)
            
        Returns:
            Dict with status and metadata including attack parameters used
        """
        
        # Check if Diff-Protect code exists
        diff_mist_script = os.path.join(self.code_dir, "diff_mist.py")
        if not os.path.exists(diff_mist_script):
            return {"status": "failed", "error": f"Diff-Protect script not found: {diff_mist_script}"}
            
        # Map model name to checkpoint path
        model_map = {
            "sd1.4": "ckpt/model.ckpt",
            "sd1.5": "ckpt/sd-v1-5.ckpt",
            "sd3": "ckpt/sd3.ckpt",
            "flux": "ckpt/flux.ckpt"
        }
        ckpt_path = model_map.get(model_name.lower(), "ckpt/model.ckpt")
        
        # Create temporary directories
        with tempfile.TemporaryDirectory() as temp_base_dir:
            temp_input_dir = os.path.join(temp_base_dir, "input")
            temp_output_dir = os.path.join(temp_base_dir, "output")
            os.makedirs(temp_input_dir)
            os.makedirs(temp_output_dir)
            
            # Copy input image to temp directory
            img_name = os.path.basename(input_image_path)
            shutil.copy(input_image_path, os.path.join(temp_input_dir, img_name))
            
            # Build Hydra-style command arguments
            cmd_args = [
                "attack.epsilon", str(epsilon),
                "attack.steps", str(steps),
                "attack.alpha", str(alpha),
                "attack.input_size", str(input_size),
                "attack.mode", attack_mode,
                "attack.g_mode", g_mode,
                "attack.using_target", str(using_target).lower(),
                "attack.target_rate", str(target_rate),
                "attack.device", str(device),
                "attack.img_path", temp_input_dir,
                "attack.output_path", temp_output_dir,
            ]
            
            # Add input prompt if provided
            if prompt:
                cmd_args.extend(["attack.input_prompt", prompt])
                
            # Add additional kwargs
            for key, value in kwargs.items():
                cmd_args.extend([f"attack.{key}", str(value)])
            
            # Construct full command
            cmd = [
                sys.executable,
                "diff_mist.py",
                "attack.epsilon=16",
                "attack.steps=100", 
                "attack.input_size=512",
                f"attack.mode={attack_mode}",
                f"attack.img_path={temp_input_dir}",
                f"attack.output_path={temp_output_dir}",
                f"attack.g_mode={g_mode}",
                f"attack.using_target={str(using_target).lower()}",
                f"attack.target_rate={target_rate}",
                f"attack.device={device}",
            ]
            
            # Add input prompt if provided
            if prompt:
                cmd.extend([f"attack.input_prompt={prompt}"])
                
            # Add additional kwargs
            for key, value in kwargs.items():
                cmd.extend([f"attack.{key}={value}"])
            
            try:
                # Run Diff-Protect
                result = subprocess.run(
                    cmd, 
                    cwd=self.code_dir, 
                    check=True, 
                    capture_output=True, 
                    text=True
                )
                
                # Find output files
                output_files = []
                for root, dirs, files in os.walk(temp_output_dir):
                    for file in files:
                        if file.endswith("attacked.png"):
                            output_files.append(os.path.join(root, file))
                        elif file.endswith("multistep.png"):
                            output_files.append(os.path.join(root, file))
                        elif file.endswith("onestep.png"):
                            output_files.append(os.path.join(root, file))
                
                if not output_files:
                    return {"status": "failed", "error": "No output files found"}
                
                # Use the first attacked image as the main result
                attacked_file = None
                for file in output_files:
                    if "attacked" in file:
                        attacked_file = file
                        break
                
                if not attacked_file and output_files:
                    attacked_file = output_files[0]
                    
                # Copy result to output path
                shutil.copy(attacked_file, output_image_path)
                
                # Also copy additional outputs if they exist
                additional_outputs = {}
                for file in output_files:
                    if file != attacked_file:
                        base_name = os.path.basename(file)
                        additional_path = os.path.join(os.path.dirname(output_image_path), base_name)
                        shutil.copy(file, additional_path)
                        additional_outputs[base_name] = additional_path
                
                return {
                    "status": "success", 
                    "model": f"sd1.4 ({attack_mode})",
                    "output": output_image_path,
                    "method": "diff_protect",
                    "attack_mode": attack_mode,
                    "parameters": {
                        "epsilon": epsilon,
                        "steps": steps,
                        "alpha": alpha,
                        "attack_mode": attack_mode,
                        "g_mode": g_mode
                    },
                    "additional_outputs": additional_outputs
                }

            except subprocess.CalledProcessError as e:
                return {
                    "status": "failed", 
                    "method": "diff_protect",
                    "error": f"Diff-Protect execution failed: {e.stderr if e.stderr else str(e)}"
                }
            except Exception as e:
                return {"status": "failed", "method": "diff_protect", "error": str(e)}

    def get_available_modes(self) -> list:
        """Return list of available attack modes"""
        return ["advdm", "texture_only", "mist", "sds", "sdsT"]
        
    def get_default_params(self, mode: str = "mist") -> dict:
        """Return default parameters for a given mode"""
        defaults = {
            "advdm": {"epsilon": 16, "steps": 100, "g_mode": "+"},
            "texture_only": {"epsilon": 16, "steps": 100, "g_mode": "+"},
            "mist": {"epsilon": 16, "steps": 100, "g_mode": "+", "using_target": False, "target_rate": 5},
            "sds": {"epsilon": 16, "steps": 100, "g_mode": "+", "using_target": False, "target_rate": 5},
            "sdsT": {"epsilon": 16, "steps": 100, "g_mode": "-", "using_target": True, "target_rate": 5},
        }
        return defaults.get(mode, defaults["mist"])