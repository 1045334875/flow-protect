import argparse
import json
import os
from typing import Dict, List, Optional

import yaml

from src.evaluation import MetricEvaluator


def load_dataset_prompts(dataset_yaml: str) -> Dict[str, List[str]]:
    """
    Load dataset YAML and map image basename -> list of target prompts.
    Supports keys: input_image or input_img.
    """
    with open(dataset_yaml, "r") as f:
        dataset = yaml.load(f, Loader=yaml.FullLoader)

    prompt_map: Dict[str, List[str]] = {}
    for item in dataset:
        raw_input_path = item.get("input_image") or item.get("input_img")
        if not raw_input_path:
            continue
        basename = os.path.splitext(os.path.basename(raw_input_path))[0]
        target_prompts = item.get("target_prompts", [])
        if isinstance(target_prompts, str):
            target_prompts = [target_prompts]
        prompt_map[basename] = target_prompts
    return prompt_map


def parse_edit_filename(filename: str) -> (str, Optional[int]):
    """
    Parse edited filename like {basename}_edit_{idx}.png
    Returns (basename, idx) or (basename, None) if no index found.
    """
    name = os.path.splitext(filename)[0]
    if "_edit_" in name:
        base, idx_str = name.rsplit("_edit_", 1)
        if idx_str.isdigit():
            return base, int(idx_str)
        return base, None
    return name, None


def main():
    parser = argparse.ArgumentParser(description="Evaluate existing results folder")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to results directory (contains original/protected/edited)")
    parser.add_argument("--dataset_yaml", type=str, help="Optional dataset YAML for target prompts")
    parser.add_argument("--output_json", type=str, help="Output JSON path (default: results_dir/batch_metrics.json)")
    args = parser.parse_args()

    results_dir = args.results_dir
    original_dir = os.path.join(results_dir, "original")
    protected_dir = os.path.join(results_dir, "protected")
    edited_dir = os.path.join(results_dir, "edited")

    for d in [original_dir, protected_dir, edited_dir]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Missing directory: {d}")

    prompt_map: Dict[str, List[str]] = {}
    if args.dataset_yaml:
        if not os.path.exists(args.dataset_yaml):
            raise FileNotFoundError(f"Dataset YAML not found: {args.dataset_yaml}")
        prompt_map = load_dataset_prompts(args.dataset_yaml)

    evaluator = MetricEvaluator()

    # Precompute protection metrics per base image
    protection_cache: Dict[str, Dict[str, float]] = {}
    for filename in os.listdir(original_dir):
        orig_path = os.path.join(original_dir, filename)
        if not os.path.isfile(orig_path):
            continue
        base = os.path.splitext(filename)[0]
        prot_path = os.path.join(protected_dir, filename)
        if not os.path.exists(prot_path):
            continue
        protection_cache[base] = evaluator.evaluate_protection(orig_path, prot_path)

    # Evaluate edited images
    all_metrics = []
    for filename in os.listdir(edited_dir):
        edited_path = os.path.join(edited_dir, filename)
        if not os.path.isfile(edited_path):
            continue

        base, edit_idx = parse_edit_filename(filename)

        # Choose protected image as editing reference
        prot_candidate = os.path.join(protected_dir, f"{base}.png")
        if not os.path.exists(prot_candidate):
            # Fallback: try same filename as edited (rare)
            prot_candidate = os.path.join(protected_dir, filename)
            if not os.path.exists(prot_candidate):
                continue

        target_prompt = None
        if base in prompt_map and edit_idx is not None:
            prompts = prompt_map[base]
            if 0 <= edit_idx < len(prompts):
                target_prompt = prompts[edit_idx]

        editing_metrics = {}
        if target_prompt:
            img_edit = evaluator._load_pil(edited_path)
            editing_metrics["clip_score"] = evaluator.calculate_clip_score(img_edit, target_prompt)
        else:
            editing_metrics["clip_score"] = None

        editing_metrics["structure_dist_original"] = evaluator.calculate_lpips(prot_candidate, edited_path)

        all_metrics.append({
            "image": f"{base}.png",
            "edit_index": edit_idx,
            "target_prompt": target_prompt,
            "protection_metrics": protection_cache.get(base),
            "editing_metrics": editing_metrics,
        })

    output_json = args.output_json or os.path.join(results_dir, "batch_metrics.json")
    with open(output_json, "w") as f:
        json.dump(all_metrics, f, indent=4, ensure_ascii=False)

    print(f"Evaluation complete. Results saved to {output_json}")


if __name__ == "__main__":
    main()


# python evaluate_results.py --results_dir results/flowedit_eval_only --dataset_yaml modules/FlowEdit/edits.yaml