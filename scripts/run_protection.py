import argparse
import json
import os
from src.pipeline import ImageProtectionPipeline


def main():
    parser = argparse.ArgumentParser(description="Run protection only")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--protection_method", type=str, required=True, choices=["atk_pdm", "diff_protect", "pid"], help="Protection method")
    parser.add_argument("--protection_model", type=str, default="sd1.4", choices=["sd1.4", "sd3", "flux"], help="Model used for protection generation")
    parser.add_argument("--source_prompt", type=str, default="", help="Optional prompt for protection")
    parser.add_argument("--evaluate", action="store_true", help="Compute protection metrics (original vs protected)")
    parser.add_argument("--output_image", type=str, help="Optional explicit output image path")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pipeline = ImageProtectionPipeline()
    prot_method = pipeline.protection_methods.get(args.protection_method)
    if not prot_method:
        raise ValueError(f"Unknown protection method: {args.protection_method}")

    protected_image_path = args.output_image or os.path.join(args.output_dir, "protected.png")
    result = prot_method.protect(
        input_image_path=args.input_image,
        output_image_path=protected_image_path,
        model_name=args.protection_model,
        prompt=args.source_prompt,
    )

    output = {"protection": result, "protected_image": protected_image_path}

    if args.evaluate:
        metrics = pipeline.evaluator.evaluate_protection(args.input_image, protected_image_path)
        output["protection_metrics"] = metrics

    with open(os.path.join(args.output_dir, "protection_results.json"), "w") as f:
        json.dump(output, f, indent=4)

    print(f"Protection complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
