from typing import Dict, Any
from .interfaces import ProtectionMethod, EditingMethod, Evaluator
from .protection import AtkPDMProtection, DiffProtectWrapper, PIDProtection
from .editing import FlowEditWrapper, DreamboothWrapper
from .evaluation import MetricEvaluator

class ImageProtectionPipeline:
    def __init__(self):
        self.protection_methods: Dict[str, ProtectionMethod] = {
            "atk_pdm": AtkPDMProtection(),
            "diff_protect": DiffProtectWrapper(),
            "pid": PIDProtection()
        }
        self.editing_methods: Dict[str, EditingMethod] = {
            "flow_edit": FlowEditWrapper(),
            "dreambooth": DreamboothWrapper()
        }
        self.evaluator = MetricEvaluator()

    def run(self, 
            input_image: str, 
            output_dir: str, 
            protection_method: str,
            protection_model: str,
            editing_method: str,
            source_prompt: str,
            target_prompt: str,
            edit_model: str = "sd3",
            **kwargs) -> Dict[str, Any]:
        
        results = {}
        
        # 1. Protection
        print(f"Running Protection: {protection_method} with {protection_model}...")
        prot_method = self.protection_methods.get(protection_method.lower())
        if not prot_method:
            raise ValueError(f"Unknown protection method: {protection_method}")
            
        protected_image_path = f"{output_dir}/protected.png"
        prot_result = prot_method.protect(
            input_image_path=input_image,
            output_image_path=protected_image_path,
            model_name=protection_model,
            prompt=source_prompt, # Some attacks might need prompt
            **kwargs
        )
        results['protection'] = prot_result
        
        if prot_result.get('status') == 'failed':
            print("Protection failed.")
            return results

        # 2. Evaluation (Protection)
        print("Evaluating Protection...")
        prot_metrics = self.evaluator.evaluate_protection(input_image, protected_image_path)
        results['protection_metrics'] = prot_metrics
        
        # 3. Editing
        print(f"Running Editing: {editing_method} with {edit_model}...")
        edit_method = self.editing_methods.get(editing_method.lower())
        if not edit_method:
            raise ValueError(f"Unknown editing method: {editing_method}")
            
        edited_image_path = f"{output_dir}/edited.png"
        edit_result = edit_method.edit(
            input_image_path=protected_image_path,
            output_image_path=edited_image_path,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            model_name=edit_model,
            **kwargs
        )
        results['editing'] = edit_result
        
        if edit_result.get('status') == 'failed':
            print("Editing failed.")
            return results

        # 4. Evaluation (Editing)
        print("Evaluating Editing...")
        edit_metrics = self.evaluator.evaluate_editing(protected_image_path, edited_image_path, target_prompt)
        results['editing_metrics'] = edit_metrics
        
        return results

