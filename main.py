import argparse
import os
import json
from src.pipeline import ImageProtectionPipeline

def main():
    parser = argparse.ArgumentParser(description="Unified Image Protection and Editing Pipeline")
    
    # Config file support
    parser.add_argument("--config", type=str, help="Path to a JSON config file to load arguments from")
    parser.add_argument("--dataset_yaml", type=str, help="Path to a YAML dataset file for batch processing")
    
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
            for key, value in config_args.items():
                if getattr(args, key) is None:
                    setattr(args, key, value)
    
    pipeline = ImageProtectionPipeline()
    
    # Mode 1: Batch Mode (Dataset YAML provided)
    if args.dataset_yaml:
        import yaml
        if not os.path.exists(args.dataset_yaml):
            raise FileNotFoundError(f"Dataset YAML not found: {args.dataset_yaml}")
            
        with open(args.dataset_yaml, 'r') as f:
            dataset = yaml.load(f, Loader=yaml.FullLoader)
            
        print(f"Running in Batch Mode with {len(dataset)} items...")
        
        # Prepare Batch Output Structure
        # results/exp_name/
        #   original/
        #   protected/
        #   edited/
        #   metrics.json
        
        batch_output_dir = args.output_dir if args.output_dir else "results/batch_run"
        os.makedirs(os.path.join(batch_output_dir, "original"), exist_ok=True)
        os.makedirs(os.path.join(batch_output_dir, "protected"), exist_ok=True)
        os.makedirs(os.path.join(batch_output_dir, "edited"), exist_ok=True)
        
        all_metrics = []
        
        for idx, item in enumerate(dataset):
            # Support FlowEdit's 'input_img' key and standard 'input_image' key
            raw_input_path = item.get("input_image") or item.get("input_img")
            
            # Resolve path: if relative, check if it exists relative to the yaml file location
            # This handles the case where edits.yaml references "example_images/..." inside modules/FlowEdit
            if raw_input_path:
                # 1. Check absolute or relative to CWD
                if os.path.exists(raw_input_path):
                    input_path = raw_input_path
                else:
                    # 2. Check relative to the dataset_yaml file
                    yaml_dir = os.path.dirname(os.path.abspath(args.dataset_yaml))
                    potential_path = os.path.join(yaml_dir, raw_input_path)
                    if os.path.exists(potential_path):
                        input_path = potential_path
                    else:
                        # 3. Fallback: maybe it's relative to project root even if yaml is elsewhere
                        input_path = raw_input_path # Will likely fail check below, but keep for logging
            else:
                input_path = None

            source_prompt = item.get("source_prompt")
            target_prompts = item.get("target_prompts", [])
            
            # Handle single target prompt if string
            if isinstance(target_prompts, str):
                target_prompts = [target_prompts]
                
            if not input_path or not os.path.exists(input_path):
                print(f"Skipping item {idx}: Input image not found ({input_path})")
                continue
                
            filename = os.path.basename(input_path)
            basename = os.path.splitext(filename)[0]
            
            # Copy original
            import shutil
            shutil.copy(input_path, os.path.join(batch_output_dir, "original", filename))
            
            # 1. Protection
            # Define protected path
            protected_path = os.path.join(batch_output_dir, "protected", filename)
            
            # We need to construct per-item output dir for pipeline run if we reuse pipeline.run
            # But pipeline.run does everything. Let's use pipeline components directly or call run per item?
            # Calling pipeline.run is easier but might create nested structures.
            # Let's use the pipeline components directly for better control over batch structure.
            
            # A. Protection
            print(f"[{idx+1}/{len(dataset)}] Protecting {filename}...")
            
            # Check if protection is requested
            if args.protection_method:
                # Check if protected file already exists
                if os.path.exists(protected_path):
                    print(f"  Skipping protection: {protected_path} already exists.")
                else:
                    prot_method = pipeline.protection_methods.get(args.protection_method)
                    try:
                        prot_result = prot_method.protect(
                            input_image_path=input_path,
                            output_image_path=protected_path,
                            model_name=args.protection_model,
                            prompt=source_prompt
                        )
                        if prot_result.get('status') == 'failed':
                            print(f"Protection failed for {filename}: {prot_result.get('error')}")
                            continue
                    except Exception as e:
                        print(f"Protection error for {filename}: {e}")
                        continue
            else:
                # No protection, just copy
                shutil.copy(input_path, protected_path)
                
            # Evaluate Protection
            prot_metrics = pipeline.evaluator.evaluate_protection(input_path, protected_path)
            
            # B. Editing (Iterate over target prompts)
            for t_idx, target_prompt in enumerate(target_prompts):
                edit_filename = f"{basename}_edit_{t_idx}.png"
                edited_path = os.path.join(batch_output_dir, "edited", edit_filename)
                
                print(f"  Editing -> {target_prompt}")
                
                if args.editing_method:
                    edit_method = pipeline.editing_methods.get(args.editing_method)
                    try:
                        edit_result = edit_method.edit(
                            input_image_path=protected_path,
                            output_image_path=edited_path,
                            source_prompt=source_prompt,
                            target_prompt=target_prompt,
                            model_name=args.edit_model
                        )
                    except Exception as e:
                         print(f"Editing error for {filename}: {e}")
                         continue
                else:
                     # No editing
                     shutil.copy(protected_path, edited_path)

                # Evaluate Editing
                edit_metrics = pipeline.evaluator.evaluate_editing(protected_path, edited_path, target_prompt)
                
                # Store Metrics
                all_metrics.append({
                    "image": filename,
                    "target_prompt": target_prompt,
                    "protection_metrics": prot_metrics,
                    "editing_metrics": edit_metrics
                })
                
        # Save Aggregate Metrics
        with open(os.path.join(batch_output_dir, "batch_metrics.json"), "w") as f:
            json.dump(all_metrics, f, indent=4)
            
        # Save run command and config
        import sys
        import time
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Save command
        cmd = "python " + " ".join(sys.argv)
        with open(os.path.join(batch_output_dir, f"run_command_{timestamp}.sh"), "w") as f:
            f.write(cmd + "\n")
            
        # Save config
        current_config = {
            "dataset_yaml": args.dataset_yaml,
            "output_dir": args.output_dir,
            "protection_method": args.protection_method,
            "protection_model": args.protection_model,
            "editing_method": args.editing_method,
            "edit_model": args.edit_model
        }
        with open(os.path.join(batch_output_dir, f"config_{timestamp}.json"), "w") as f:
            json.dump(current_config, f, indent=4)

        print(f"Batch processing complete. Results in {batch_output_dir}")
        return

    # Mode 2: Single Image Mode (CLI args)
    # Validate required args
    required_args = ["input_image", "output_dir", "protection_method", "editing_method", "source_prompt", "target_prompt"]
    for arg in required_args:
        if getattr(args, arg) is None:
            parser.error(f"Argument --{arg} is required (either via CLI or config file) when not using --dataset_yaml")
 
     
    os.makedirs(args.output_dir, exist_ok=True)
    
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
