import argparse
import os
import json
from src.pipeline import ImageProtectionPipeline

def main():
    parser = argparse.ArgumentParser(description="Unified Image Protection and Editing Pipeline")
    
    # Config file support
    parser.add_argument("--config", type=str, help="Path to a JSON config file to load arguments from")
    
    # Input/Output
    parser.add_argument("--input_image", type=str, help="Path to input image")
    parser.add_argument("--output_dir", type=str, help="Directory to save results")
    
    # Protection
    parser.add_argument("--protection_method", type=str, choices=["atk_pdm", "diff_protect", "pid"], help="Protection method")
    parser.add_argument("--protection_model", type=str, default="sd1.4", choices=["sd1.4", "sd3", "flux"], help="Model used for protection generation")
    
    # Editing
    parser.add_argument("--editing_method", type=str, choices=["flow_edit", "dreambooth"], help="Editing method")
    parser.add_argument("--edit_model", type=str, default="sd3", choices=["sd3", "flux", "sd1.4"], help="Model used for editing")
    
    # Prompts
    parser.add_argument("--source_prompt", type=str, help="Description of input image")
    parser.add_argument("--target_prompt", type=str, help="Description of desired edit")
    
    args = parser.parse_args()

    # Load config if provided
    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        with open(args.config, 'r') as f:
            config_args = json.load(f)
            # Override command line defaults with config values if not specified on CLI
            # But here we simplified: if config exists, we use it to populate missing args
            for key, value in config_args.items():
                if getattr(args, key) is None:
                    setattr(args, key, value)
    
    # Validate required args
    required_args = ["input_image", "output_dir", "protection_method", "editing_method", "source_prompt", "target_prompt"]
    for arg in required_args:
        if getattr(args, arg) is None:
            parser.error(f"Argument --{arg} is required (either via CLI or config file)")

    
    os.makedirs(args.output_dir, exist_ok=True)
    
    pipeline = ImageProtectionPipeline()
    
    results = pipeline.run(
        input_image=args.input_image,
        output_dir=args.output_dir,
        protection_method=args.protection_method,
        protection_model=args.protection_model,
        editing_method=args.editing_method,
        source_prompt=args.source_prompt,
        target_prompt=args.target_prompt,
        edit_model=args.edit_model
    )
    
    # Save results summary
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    # Save the config used for this run with timestamp
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    config_save_path = os.path.join(args.output_dir, f"config_{timestamp}.json")
    with open(config_save_path, "w") as f:
        # Reconstruct config from args
        current_config = {
            "input_image": args.input_image,
            "output_dir": args.output_dir,
            "protection_method": args.protection_method,
            "protection_model": args.protection_model,
            "editing_method": args.editing_method,
            "edit_model": args.edit_model,
            "source_prompt": args.source_prompt,
            "target_prompt": args.target_prompt
        }
        json.dump(current_config, f, indent=4)
        
    print(f"Pipeline completed. Results saved to results.json. Config saved to {config_save_path}")

if __name__ == "__main__":
    main()
