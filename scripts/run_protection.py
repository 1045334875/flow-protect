import argparse
import json
import os
import sys
import yaml
from pathlib import Path

# Add project root to path so we can import src modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import ImageProtectionPipeline


def load_config_from_yaml(yaml_path):
    """Load configuration from FlowEdit format YAML"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # Convert FlowEdit format to our format
    configs = []
    for item in data:
        if not item:  # Skip empty entries
            continue
            
        config = {
            'input_image': item.get('input_img', ''),
            'source_prompt': item.get('source_prompt', ''),
            'target_prompts': item.get('target_prompts', []),
            'target_codes': item.get('target_codes', [])
        }
        configs.append(config)
    
    return configs


def main():
    parser = argparse.ArgumentParser(description="Run protection using FlowEdit YAML config")
    parser.add_argument("--config", type=str, required=True, help="Path to FlowEdit YAML config file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--protection_method", type=str, required=True, 
                       choices=["atk_pdm", "diff_protect", "pid"], 
                       help="Protection method")
    parser.add_argument("--protection_model", type=str, default="sd1.4", 
                       choices=["sd1.4", "sd1.5", "sd3", "flux"], 
                       help="Model used for protection generation")
    parser.add_argument("--evaluate", action="store_true", 
                       help="Compute protection metrics (original vs protected)")
    
    # Method-specific arguments
    parser.add_argument("--optim_steps", type=int, default=100, 
                       help="Number of optimization steps (atk_pdm)")
    parser.add_argument("--epsilon", type=float, default=16.0, 
                       help="Perturbation budget (diff_protect)")
    parser.add_argument("--attack_mode", type=str, default="mist", 
                       choices=["advdm", "texture_only", "mist", "sds", "sdsT"],
                       help="Attack mode for diff_protect")
    parser.add_argument("--max_train_steps", type=int, default=100, 
                       help="Training steps for PID")
    parser.add_argument("--eps", type=float, default=12.75, 
                       help="Epsilon for PID protection")
    
    args = parser.parse_args()

    # Load configurations from YAML
    configs = load_config_from_yaml(args.config)
    
    # Get the directory containing the YAML file for relative paths
    config_dir = Path(args.config).parent
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    pipeline = ImageProtectionPipeline()
    prot_method = pipeline.protection_methods.get(args.protection_method)
    if not prot_method:
        raise ValueError(f"Unknown protection method: {args.protection_method}")

    results = []
    
    for i, config in enumerate(configs):
        if not config['input_image']:
            continue
            
        print(f"\n=== Processing image {i+1}/{len(configs)}: {config['input_image']} ===")
        
        # Handle relative paths
        input_image_path = config['input_image']
        if not os.path.isabs(input_image_path):
            input_image_path = os.path.join(config_dir, input_image_path)
        
        if not os.path.exists(input_image_path):
            print(f"Warning: Input image not found: {input_image_path}")
            continue
            
        # Create output subdirectory for this image
        image_name = Path(input_image_path).stem
        image_output_dir = os.path.join(args.output_dir, f"{image_name}_{args.protection_method}")
        os.makedirs(image_output_dir, exist_ok=True)
        
        protected_image_path = os.path.join(image_output_dir, f"protected_{image_name}.png")
        
        # Prepare method-specific kwargs
        method_kwargs = {}
        if args.protection_method == "atk_pdm":
            method_kwargs.update({
                'optim_steps': args.optim_steps,
                'protection_mode': 'vae',
                'batch_size': 1,
                'step_size': 1.0
            })
        elif args.protection_method == "diff_protect":
            method_kwargs.update({
                'attack_mode': args.attack_mode,
                'epsilon': args.epsilon,
                'steps': 100,
                'alpha': 1.0
            })
        elif args.protection_method == "pid":
            method_kwargs.update({
                'max_train_steps': args.max_train_steps,
                'eps': args.eps,
                'resolution': 512,
                'attack_type': 'add'
            })
        
        # Run protection
        try:
            result = prot_method.protect(
                input_image_path=input_image_path,
                output_image_path=protected_image_path,
                model_name=args.protection_model,
                prompt=config.get('source_prompt', ''),
                **method_kwargs
            )
            
            output_item = {
                "image_index": i,
                "image_name": image_name,
                "input_image": input_image_path,
                "protected_image": protected_image_path,
                "source_prompt": config.get('source_prompt', ''),
                "protection_method": args.protection_method,
                "protection_model": args.protection_model,
                "protection": result
            }
            
            # Add method-specific parameters to output
            if args.protection_method == "atk_pdm":
                output_item["atk_pdm_params"] = {
                    "optim_steps": args.optim_steps
                }
            elif args.protection_method == "diff_protect":
                output_item["diff_protect_params"] = {
                    "attack_mode": args.attack_mode,
                    "epsilon": args.epsilon
                }
            elif args.protection_method == "pid":
                output_item["pid_params"] = {
                    "max_train_steps": args.max_train_steps,
                    "eps": args.eps
                }
            
            if args.evaluate and result.get('status') == 'success':
                print("Computing protection metrics...")
                metrics = pipeline.evaluator.evaluate_protection(input_image_path, protected_image_path)
                output_item["protection_metrics"] = metrics
            
            results.append(output_item)
            
            print(f"Protection result: {result.get('status', 'unknown')}")
            if result.get('status') == 'failed':
                print(f"Error: {result.get('error', 'Unknown error')}")
            else:
                print(f"Protected image saved to: {protected_image_path}")
                
        except Exception as e:
            error_result = {
                "image_index": i,
                "image_name": image_name,
                "input_image": input_image_path,
                "source_prompt": config.get('source_prompt', ''),
                "protection_method": args.protection_method,
                "protection": {"status": "failed", "error": str(e)}
            }
            results.append(error_result)
            print(f"Error processing {input_image_path}: {e}")

    # Save all results
    output_file = os.path.join(args.output_dir, f"protection_results_{args.protection_method}.json")
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump({
            "protection_method": args.protection_method,
            "protection_model": args.protection_model,
            "total_images": len(results),
            "successful_protections": len([r for r in results if r.get('protection', {}).get('status') == 'success']),
            "results": results
        }, f, indent=4, ensure_ascii=False)

    print(f"\n=== Protection Complete ===")
    print(f"Processed {len(results)} images")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
