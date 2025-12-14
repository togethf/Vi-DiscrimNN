import subprocess
import csv
import re
import itertools
import os
import sys
from typing import List, Dict, Tuple

# ================= 配置区域 =================

# 1. 你的运行脚本名称
TARGET_SCRIPT = "double-branch.py"

# 2. 基础模型定义 (对应索引 0, 1, 2, 3...)
# 请确保这里的顺序与 config_plus.py 中的 det_pool 一致
MODEL_NAMES = {
    2: "Baseline",
    0: "SHC-YOLO",
    3: "VS-YOLO",
    1: "YOLO-iEMA",
}

# 3. 手动录入的效率指标 (因为 double-branch.py 可能不输出这个)
# Key: 模型索引, Value: (Latency_ms, Params_M)
BASE_STATS = {
    0: (22.5, 25.3),
    1: (24.1, 28.7),
    2: (26.3, 27.2),
    3: (21.8, 24.5),
}

# 4. 输出文件
OUTPUT_CSV = "experiments/results/ensemble_results_final.csv"

# ============================================

def get_efficiency_stats(indices: List[int]) -> Tuple[float, float]:
    """计算组合的延迟和参数量（简单求和）"""
    total_latency = 0.0
    total_params = 0.0
    for i in indices:
        lat, param = BASE_STATS.get(i, (0.0, 0.0))
        total_latency += lat
        total_params += param
    return total_latency, total_params

def parse_metrics_from_output(output_text: str) -> Dict[str, float]:
    """
    根据你提供的具体日志格式提取指标：
    1. mAP@0.5:0.95: 0.7541
    2. all ... P R mAP50 F1
    """
    metrics = {
        'mAP50': 0.0,
        'mAP5095': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'F1': 0.0
    }
    
    # 1. 提取 mAP@0.5:0.95
    # 目标行: "mAP@0.5:0.95: 0.7541"
    match_map5095 = re.search(r'mAP@0\.5:0\.95:\s+([0-9.]+)', output_text)
    if match_map5095:
        metrics['mAP5095'] = float(match_map5095.group(1))

    # 2. 提取表格中 'all' 这一行的数据
    # 目标行: "all     1801      2392      0.673      0.925      0.859      0.773"
    # 对应的列顺序是: Class, Images, Instances, P, R, mAP50, F1
    
    # 正则解释: 
    # all -> 空格 -> 数字(Images) -> 空格 -> 数字(Instances) -> 空格 
    # -> 捕获(P) -> 空格 -> 捕获(R) -> 空格 -> 捕获(mAP50) -> 空格 -> 捕获(F1)
    match_all_row = re.search(r'all\s+\d+\s+\d+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)', output_text)
    
    if match_all_row:
        metrics['precision'] = float(match_all_row.group(1)) # Group 1: P
        metrics['recall']    = float(match_all_row.group(2)) # Group 2: R
        metrics['mAP50']     = float(match_all_row.group(3)) # Group 3: mAP50
        metrics['F1']        = float(match_all_row.group(4)) # Group 4: F1
    
    return metrics

def run_experiment(indices: List[int]):
    """调用命令行运行 double-branch.py"""
    
    # 构建模型索引字符串 "0,2,3"
    idx_str = ",".join(map(str, indices))
    
    # 构建命令
    cmd = [
        sys.executable,  # 当前 python 解释器路径
        TARGET_SCRIPT,
        "--ensemble",
        "--ensemble_models", idx_str
    ]
    
    print(f"\n🚀 Running combination: {indices} -> Command: {' '.join(cmd)}")
    
    try:
        # 运行命令并捕获输出
        # stderr=subprocess.STDOUT 确保如果报错也能被 capture 到 stdout 中
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running combination {indices}:")
        # 打印部分错误信息方便调试
        print(e.stderr[:500] if e.stderr else e.stdout[:500])
        return None

def main():
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    # 准备 CSV Header
    header = ['combination', 'mAP50', 'mAP5095', 'precision', 'recall', 'F1', 'latency_ms', 'params_m']
    
    results_data = []
    
    # 生成所有组合 (1到N的所有排列组合)
    num_models = len(MODEL_NAMES)
    all_indices = list(range(num_models))
    
    combinations_list = []
    for r in range(1, num_models + 1):
        combinations_list.extend(itertools.combinations(all_indices, r))
        
    print(f"Total combinations to run: {len(combinations_list)}")
    
    for combo in combinations_list:
        indices = list(combo)
        
        # 1. 获取组合名称和效率指标
        parts = sorted([MODEL_NAMES[i] for i in indices])
        combination_name = "+".join(parts)
        latency, params = get_efficiency_stats(indices)
        
        # 2. 运行脚本获取性能指标
        stdout = run_experiment(indices)
        
        if stdout:
            # 3. 解析输出
            metrics = parse_metrics_from_output(stdout)
            
            # 如果解析失败（mAP50还是0），并且 stdout 其实有内容，说明正则可能还没对上
            if metrics['mAP50'] == 0.0 and "mAP" in stdout:
                 print(f"⚠️ Warning: Parsed 0.0 but output seems to exist. Check regex.")
            else:
                 print(f"✅ Success: {combination_name} -> mAP50: {metrics['mAP50']}, mAP50-95: {metrics['mAP5095']}")

            # 4. 收集数据
            row = [
                combination_name,
                f"{metrics['mAP50']:.4f}",
                f"{metrics['mAP5095']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['F1']:.4f}",
                f"{latency:.1f}",
                f"{params:.1f}"
            ]
            results_data.append(row)
        else:
            print(f"⚠️ Skipping {combination_name} due to runtime error.")

    # 写入 CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results_data)
        
    print(f"\n🎉 All done! Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()