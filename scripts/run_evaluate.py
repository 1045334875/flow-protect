import argparse
import json

from src.evaluation import MetricEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate protection + editing for a single image")
    parser.add_argument("--original_image", type=str, required=True, help="Path to original image")
    parser.add_argument("--protected_image", type=str, required=True, help="Path to protected image")
    parser.add_argument("--edited_image", type=str, required=True, help="Path to edited image")
    parser.add_argument("--target_prompt", type=str, required=True, help="Target prompt for CLIP score")
    parser.add_argument("--output_json", type=str, required=True, help="Path to output JSON")
    args = parser.parse_args()

    evaluator = MetricEvaluator()

    protection_metrics = evaluator.evaluate_protection(args.original_image, args.protected_image)
    editing_metrics = evaluator.evaluate_editing(args.protected_image, args.edited_image, args.target_prompt)

    output = {
        "protection_metrics": protection_metrics,
        "editing_metrics": editing_metrics,
    }

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"Evaluation complete. Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
