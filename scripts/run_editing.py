import argparse
import json
import os

from src.pipeline import ImageProtectionPipeline


def main():
    parser = argparse.ArgumentParser(description="Run editing only")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input (protected) image")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--editing_method", type=str, required=True, choices=["flow_edit", "dreambooth"], help="Editing method")
    parser.add_argument("--edit_model", type=str, default="sd3", choices=["sd3", "flux", "sd1.4"], help="Model used for editing")
    parser.add_argument("--source_prompt", type=str, required=True, help="Description of input image")
    parser.add_argument("--target_prompt", type=str, required=True, help="Description of desired edit")
    parser.add_argument("--evaluate", action="store_true", help="Compute editing metrics (protected vs edited)")
    parser.add_argument("--output_image", type=str, help="Optional explicit output image path")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pipeline = ImageProtectionPipeline()
    edit_method = pipeline.editing_methods.get(args.editing_method)
    if not edit_method:
        raise ValueError(f"Unknown editing method: {args.editing_method}")

    edited_image_path = args.output_image or os.path.join(args.output_dir, "edited.png")
    result = edit_method.edit(
        input_image_path=args.input_image,
        output_image_path=edited_image_path,
        source_prompt=args.source_prompt,
        target_prompt=args.target_prompt,
        model_name=args.edit_model,
    )

    output = {"editing": result, "edited_image": edited_image_path}

    if args.evaluate:
        metrics = pipeline.evaluator.evaluate_editing(args.input_image, edited_image_path, args.target_prompt)
        output["editing_metrics"] = metrics

    with open(os.path.join(args.output_dir, "editing_results.json"), "w") as f:
        json.dump(output, f, indent=4)

    print(f"Editing complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
