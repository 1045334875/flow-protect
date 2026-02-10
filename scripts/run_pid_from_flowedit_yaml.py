import argparse
import json
import os
import shutil
import sys
import tempfile
from typing import Dict, List, Optional

import yaml

# 计算项目根目录（scripts 的父目录）
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# 使用本地 PID 实现（不再依赖 submodule）
from src.protection.pid_core import PIDConfig, run_pid_protection, parse_args as pid_parse_args, main as pid_main
from src.evaluation.metrics import MetricEvaluator


def _resolve_input_path(raw_input_path: str, dataset_yaml: str) -> Optional[str]:
    if not raw_input_path:
        return None
    if os.path.exists(raw_input_path):
        return raw_input_path
    yaml_dir = os.path.dirname(os.path.abspath(dataset_yaml))
    candidate = os.path.join(yaml_dir, raw_input_path)
    if os.path.exists(candidate):
        return candidate
    return raw_input_path


def _select_latest_image(output_dir: str) -> Optional[str]:
    if not os.path.isdir(output_dir):
        return None
    candidates = []
    for name in os.listdir(output_dir):
        if name.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(output_dir, name)
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _model_id(model_name: str) -> str:
    model_map = {
        "sd1.4": "CompVis/stable-diffusion-v1-4",
        "sd1.5": "runwayml/stable-diffusion-v1-5",
        "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",
        "flux": "black-forest-labs/FLUX.1-dev",
    }
    return model_map.get(model_name.lower(), model_name)


def load_dataset(dataset_yaml: str) -> List[Dict]:
    with open(dataset_yaml, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def main():
    parser = argparse.ArgumentParser(description="Run PID protection on FlowEdit YAML dataset")
    parser.add_argument("--dataset_yaml", type=str, default="modules/FlowEdit/edits.yaml", help="Path to FlowEdit edits.yaml")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--protection_model", type=str, default="sd1.4", choices=["sd1.4", "sd1.5", "sd3", "flux"], help="Model for PID")
    parser.add_argument("--max_train_steps", type=int, default=100, help="PID max train steps")
    parser.add_argument("--attack_type", type=str, default="add", choices=["var", "mean", "KL", "add-log", "latent_vector", "add"], help="PID attack type")
    parser.add_argument("--eps", type=float, default=12.75, help="PID eps")
    parser.add_argument("--step_size", type=float, default=1/255, help="PID step size")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--evaluate", action="store_true", help="Compute protection metrics")
    parser.add_argument("--max_items", type=int, default=None, help="Limit number of items for quick tests")

    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="pid", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity")
    parser.add_argument("--wandb_mode", type=str, default=None, help="W&B mode (online/offline/disabled)")
    parser.add_argument("--wandb_log_images", action="store_true", help="Log sample images to W&B")

    args = parser.parse_args()

    if not os.path.exists(args.dataset_yaml):
        raise FileNotFoundError(f"Dataset YAML not found: {args.dataset_yaml}")

    os.makedirs(args.output_dir, exist_ok=True)
    original_dir = os.path.join(args.output_dir, "original")
    protected_dir = os.path.join(args.output_dir, "protected")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(protected_dir, exist_ok=True)

    evaluator = MetricEvaluator() if args.evaluate else None

    dataset = load_dataset(args.dataset_yaml)
    results = []

    for idx, item in enumerate(dataset):
        if args.max_items is not None and idx >= args.max_items:
            break

        raw_input_path = item.get("input_image") or item.get("input_img")
        input_path = _resolve_input_path(raw_input_path, args.dataset_yaml)
        if not input_path or not os.path.exists(input_path):
            results.append({"index": idx, "status": "skipped", "error": f"Input not found: {input_path}"})
            continue

        filename = os.path.basename(input_path)
        basename = os.path.splitext(filename)[0]
        shutil.copy(input_path, os.path.join(original_dir, filename))

        with tempfile.TemporaryDirectory() as temp_input_dir, tempfile.TemporaryDirectory() as temp_output_dir:
            shutil.copy(input_path, os.path.join(temp_input_dir, filename))

            pid_args_list = [
                "--pretrained_model_name_or_path", _model_id(args.protection_model),
                "--instance_data_dir", temp_input_dir,
                "--output_dir", temp_output_dir,
                "--max_train_steps", str(args.max_train_steps),
                "--attack_type", args.attack_type,
                "--eps", str(args.eps),
                "--step_size", str(args.step_size),
            ]
            if args.seed is not None:
                pid_args_list += ["--seed", str(args.seed)]
            if args.use_wandb:
                pid_args_list.append("--use_wandb")
                pid_args_list += ["--wandb_project", args.wandb_project]
                if args.wandb_run_name:
                    pid_args_list += ["--wandb_run_name", args.wandb_run_name]
                if args.wandb_entity:
                    pid_args_list += ["--wandb_entity", args.wandb_entity]
                if args.wandb_mode:
                    pid_args_list += ["--wandb_mode", args.wandb_mode]
                if args.wandb_log_images:
                    pid_args_list.append("--wandb_log_images")

            try:
                parsed_args = pid_parse_args(pid_args_list)
                pid_main(parsed_args)
            except Exception as e:
                results.append({"image": filename, "status": "failed", "error": str(e)})
                continue

            latest = _select_latest_image(temp_output_dir)
            if not latest:
                results.append({"image": filename, "status": "failed", "error": "PID produced no output"})
                continue

            protected_path = os.path.join(protected_dir, filename)
            shutil.move(latest, protected_path)

            record = {"image": filename, "status": "success"}
            if evaluator:
                record["protection_metrics"] = evaluator.evaluate_protection(input_path, protected_path)
            results.append(record)

    output_json = os.path.join(args.output_dir, "pid_metrics.json")
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"PID run complete. Results saved to {output_json}")


if __name__ == "__main__":
    main()