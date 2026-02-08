import sys
import os
import shutil
import tempfile
import subprocess
from typing import Dict, Any
from ..interfaces import ProtectionMethod

class DiffProtectWrapper(ProtectionMethod):
    def protect(self, 
                input_image_path: str, 
                output_image_path: str, 
                model_name: str = "sd1.4", 
                prompt: str = "",
                **kwargs) -> Dict[str, Any]:
        
        # Diff-Protect relies on specific relative paths and config files.
        # We'll run it via subprocess from its own directory.
        
        diff_protect_code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../modules/Diff-Protect/code"))
        diff_protect_script = os.path.join(diff_protect_code_dir, "diff_mist.py")
        
        if not os.path.exists(diff_protect_script):
             return {"status": "failed", "error": f"Script not found: {diff_protect_script}"}

        # Create temp input dir (Diff-Protect expects a directory of images)
        with tempfile.TemporaryDirectory() as temp_input_dir:
            # Create a subdirectory for the image because the script seems to parse paths like rsplit('/', 2)
            # Line 407: file_name = f"/{rsplit_image_path[-2]}/{rsplit_image_path[-1]}/"
            # So we might need nested structure: {temp_input_dir}/subset/image.png
            subset_dir = os.path.join(temp_input_dir, "to_protect")
            os.makedirs(subset_dir)
            
            img_name = os.path.basename(input_image_path)
            shutil.copy(input_image_path, os.path.join(subset_dir, img_name))
            
            # Create temp output dir
            temp_output_dir = os.path.join(os.path.dirname(output_image_path), "diff_protect_temp_out")
            os.makedirs(temp_output_dir, exist_ok=True)
            
            # Construct command
            # Using Hydra syntax for overrides
            cmd = [
                sys.executable,
                "diff_mist.py",
                f"img_path={subset_dir}",
                f"output_path={temp_output_dir}",
                # Default parameters (can be overridden by kwargs)
                "epsilon=16", 
                "steps=50",
                "mode=mist"
            ]
            
            # Add kwargs as overrides
            for k, v in kwargs.items():
                cmd.append(f"{k}={v}")
                
            try:
                subprocess.run(cmd, cwd=diff_protect_code_dir, check=True, capture_output=True)
                
                # Find the output file
                # The script creates subdirectories based on params.
                # We need to recursively search for the result image.
                # Expected suffix: _attacked.png
                
                result_file = None
                for root, dirs, files in os.walk(temp_output_dir):
                    for file in files:
                        if file.endswith("attacked.png") and img_name.split('.')[0] in file:
                            result_file = os.path.join(root, file)
                            break
                    if result_file:
                        break
                
                if result_file:
                    shutil.move(result_file, output_image_path)
                    return {"status": "success", "model": "sd1.4 (Diff-Protect default)"}
                else:
                    return {"status": "failed", "error": "Output file not found in generated directories"}

            except subprocess.CalledProcessError as e:
                return {"status": "failed", "error": f"Subprocess error: {e.stderr.decode() if e.stderr else str(e)}"}
            except Exception as e:
                return {"status": "failed", "error": str(e)}
            finally:
                if os.path.exists(temp_output_dir):
                    shutil.rmtree(temp_output_dir)
