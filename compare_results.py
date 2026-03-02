import os
import json
import argparse
from collections import defaultdict
import glob

def load_metrics(results_dir):
    """
    遍历 results 目录下所有子目录，寻找 batch_metrics.json
    返回结构:
    {
        "exp_name": {
            "config": {...},
            "metrics": [ ... ]
        }
    }
    """
    experiments = {}
    
    # 查找所有 batch_metrics.json
    metric_files = glob.glob(os.path.join(results_dir, "*", "batch_metrics.json"))
    
    for metric_file in metric_files:
        exp_dir = os.path.dirname(metric_file)
        exp_name = os.path.basename(exp_dir)
        
        # 加载 metrics
        try:
            with open(metric_file, 'r') as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"Error loading metrics for {exp_name}: {e}")
            continue
            
        # 尝试加载 config (通常以 config_ 开头)
        config = {}
        config_files = glob.glob(os.path.join(exp_dir, "config_*.json"))
        if config_files:
            # 取最新的 config
            latest_config = max(config_files, key=os.path.getctime)
            try:
                with open(latest_config, 'r') as f:
                    config = json.load(f)
            except:
                pass
        
        experiments[exp_name] = {
            "metrics": metrics,
            "config": config
        }
        
    return experiments

def aggregate_metrics(experiments):
    """
    计算每个实验的平均指标
    """
    summary = []
    
    for exp_name, data in experiments.items():
        metrics_list = data["metrics"]
        config = data["config"]
        
        if not metrics_list:
            continue
            
        # 初始化累加器
        agg = defaultdict(list)
        
        for item in metrics_list:
            # Protection Metrics (如 SSIM, PSNR, LPIPS)
            prot_metrics = item.get("protection_metrics", {})
            for k, v in prot_metrics.items():
                if isinstance(v, (int, float)):
                    agg[f"prot_{k}"].append(v)
            
            # Editing Metrics (如 CLIP Score)
            edit_metrics = item.get("editing_metrics", {})
            for k, v in edit_metrics.items():
                if isinstance(v, (int, float)):
                    agg[f"edit_{k}"].append(v)
        
        # 计算平均值
        avg_metrics = {k: sum(v)/len(v) for k, v in agg.items()}
        
        # 提取关键配置信息
        prot_method = config.get("protection_method", "N/A")
        edit_method = config.get("editing_method", "N/A")
        prot_model = config.get("protection_model", "N/A")
        
        # 构造汇总行
        row = {
            "Experiment": exp_name,
            "Protection": f"{prot_method} ({prot_model})",
            "Editing": edit_method,
            **avg_metrics
        }
        summary.append(row)
        
    return summary

def print_table(summary):
    if not summary:
        print("No results found.")
        return

    # 确定所有可能的列名
    all_keys = set()
    for row in summary:
        all_keys.update(row.keys())
    
    # 定义列顺序
    # 优先展示实验信息，然后是保护指标，最后是编辑指标
    headers = ["Experiment", "Protection", "Editing"]
    prot_headers = sorted([k for k in all_keys if k.startswith("prot_")])
    edit_headers = sorted([k for k in all_keys if k.startswith("edit_")])
    
    final_headers = headers + prot_headers + edit_headers
    
    # 打印表头
    header_str = " | ".join(f"{h:<20}" for h in final_headers)
    print("-" * len(header_str))
    print(header_str)
    print("-" * len(header_str))
    
    # 打印数据行
    for row in summary:
        row_str = []
        for h in final_headers:
            val = row.get(h, "N/A")
            if isinstance(val, float):
                row_str.append(f"{val:<20.4f}")
            else:
                row_str.append(f"{str(val):<20}")
        print(" | ".join(row_str))
    print("-" * len(header_str))

def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument("--results_dir", type=str, default="results", help="Path to results directory")
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Results directory not found: {args.results_dir}")
        return

    print(f"Loading results from: {args.results_dir}")
    experiments = load_metrics(args.results_dir)
    summary = aggregate_metrics(experiments)
    print_table(summary)

if __name__ == "__main__":
    main()
