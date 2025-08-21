#!/usr/bin/env python3
"""
visualization_analysis2.py
专门用于重新计算单一实验或批量实验的coverage rate（使用方法2: 正类比例相似度计算）
并更新metrics_summary.json中的coverage rate数值
"""

import os
import json
import pandas as pd
import numpy as np
import argparse
import glob

def recalculate_coverage_rate(experiment_path):
    """
    重新计算coverage rate使用方法2: 正类比例相似度计算
    
    Args:
        experiment_path (str): 实验结果路径
    
    Returns:
        float: 新的coverage rate值
    """
    # 检查必要文件是否存在
    required_files = ["best_y.csv", "y_target.csv", "metrics_summary.json"]
    for file in required_files:
        file_path = os.path.join(experiment_path, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    # 加载数据
    best_y_path = os.path.join(experiment_path, "best_y.csv")
    y_target_path = os.path.join(experiment_path, "y_target.csv")
    
    best_y_df = pd.read_csv(best_y_path)
    y_target_df = pd.read_csv(y_target_path)
    
    # 处理数据格式
    counterfactual_y = np.array((best_y_df > 0.5).astype(int)).flatten()
    y_target = y_target_df['y_target'].values.flatten()
    
    # 方法2: 正类比例相似度计算
    cf_positive_ratio = np.mean(counterfactual_y > 0.5)  # 反事实正类比例
    target_positive_ratio = np.mean(y_target > 0.5)     # 目标正类比例
    coverage_rate_new = 1 - abs(cf_positive_ratio - target_positive_ratio)
    
    return coverage_rate_new, cf_positive_ratio, target_positive_ratio

def update_metrics_summary(experiment_path, new_coverage_rate):
    """
    更新metrics_summary.json中的coverage rate数值
    
    Args:
        experiment_path (str): 实验结果路径
        new_coverage_rate (float): 新的coverage rate值
    """
    metrics_file = os.path.join(experiment_path, "metrics_summary.json")
    
    # 读取原始metrics
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # 备份原始值
    original_coverage_rate = metrics.get('Coverage Rate', 'Not found')
    
    # 更新coverage rate
    metrics['Coverage Rate'] = new_coverage_rate
    
    # 保存更新后的metrics
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return original_coverage_rate

def find_experiment_paths(root_path):
    """
    查找根路径下所有包含metrics_summary.json的实验路径
    
    Args:
        root_path (str): 根路径
    
    Returns:
        list: 实验路径列表
    """
    experiment_paths = []
    
    # 使用glob递归查找所有metrics_summary.json文件
    pattern = os.path.join(root_path, "**/metrics_summary.json")
    metrics_files = glob.glob(pattern, recursive=True)
    
    # 提取实验路径（去掉文件名）
    for metrics_file in metrics_files:
        experiment_path = os.path.dirname(metrics_file)
        experiment_paths.append(experiment_path)
    
    return experiment_paths

def process_single_experiment(experiment_path, dry_run=False):
    """
    处理单个实验
    
    Args:
        experiment_path (str): 实验路径
        dry_run (bool): 是否为dry-run模式
    
    Returns:
        dict: 处理结果
    """
    result = {
        'path': experiment_path,
        'success': False,
        'error': None,
        'original_coverage_rate': None,
        'new_coverage_rate': None,
        'cf_ratio': None,
        'target_ratio': None
    }
    
    try:
        # 重新计算coverage rate
        new_coverage_rate, cf_ratio, target_ratio = recalculate_coverage_rate(experiment_path)
        
        result['new_coverage_rate'] = new_coverage_rate
        result['cf_ratio'] = cf_ratio
        result['target_ratio'] = target_ratio
        
        if not dry_run:
            # 更新metrics_summary.json
            original_coverage_rate = update_metrics_summary(experiment_path, new_coverage_rate)
            result['original_coverage_rate'] = original_coverage_rate
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result

def main():
    parser = argparse.ArgumentParser(description='重新计算单一实验或批量实验的coverage rate（正类比例相似度方法）')
    parser.add_argument('path', help='实验结果路径（单一实验）或文件夹路径（批量处理）')
    parser.add_argument('--dry-run', action='store_true', help='只计算不修改文件')
    parser.add_argument('--batch', action='store_true', help='批量处理模式（处理指定文件夹下所有实验）')
    
    args = parser.parse_args()
    
    # 检查路径是否存在
    if not os.path.exists(args.path):
        print(f"❌ 路径不存在: {args.path}")
        return
    
    if args.batch:
        # 批量处理模式
        print(f"📁 批量处理文件夹: {args.path}")
        
        # 查找所有实验路径
        experiment_paths = find_experiment_paths(args.path)
        
        if not experiment_paths:
            print(f"❌ 在指定路径下未找到任何包含metrics_summary.json的实验")
            return
        
        print(f"🔍 找到 {len(experiment_paths)} 个实验")
        
        # 处理统计
        success_count = 0
        error_count = 0
        results = []
        
        # 逐个处理实验
        for i, experiment_path in enumerate(experiment_paths, 1):
            print(f"\n[{i}/{len(experiment_paths)}] 处理: {os.path.basename(experiment_path)}")
            
            result = process_single_experiment(experiment_path, args.dry_run)
            results.append(result)
            
            if result['success']:
                success_count += 1
                print(f"   ✅ 成功 - Coverage Rate: {result['original_coverage_rate']} → {result['new_coverage_rate']:.3f}")
                print(f"      反事实正类比例: {result['cf_ratio']:.3f}, 目标正类比例: {result['target_ratio']:.3f}")
            else:
                error_count += 1
                print(f"   ❌ 失败: {result['error']}")
        
        # 总结
        print(f"\n📊 批量处理完成:")
        print(f"   总实验数: {len(experiment_paths)}")
        print(f"   成功: {success_count}")
        print(f"   失败: {error_count}")
        
        if args.dry_run:
            print(f"   🔍 Dry-run模式，未修改任何文件")
        else:
            print(f"   💾 已更新 {success_count} 个实验的metrics_summary.json")
    
    else:
        # 单一实验处理模式
        print(f"📁 处理实验路径: {args.path}")
        
        result = process_single_experiment(args.path, args.dry_run)
        
        if result['success']:
            print(f"📊 Coverage Rate计算结果:")
            print(f"   反事实正类比例: {result['cf_ratio']:.3f} ({result['cf_ratio']*100:.1f}%)")
            print(f"   目标正类比例: {result['target_ratio']:.3f} ({result['target_ratio']*100:.1f}%)")
            print(f"   新Coverage Rate (比例相似度): {result['new_coverage_rate']:.3f} ({result['new_coverage_rate']*100:.1f}%)")
            
            if not args.dry_run:
                print(f"✅ metrics_summary.json已更新:")
                print(f"   原始Coverage Rate: {result['original_coverage_rate']}")
                print(f"   新Coverage Rate: {result['new_coverage_rate']:.3f}")
                print(f"   文件路径: {os.path.join(args.path, 'metrics_summary.json')}")
            else:
                print("🔍 Dry-run模式，未修改文件")
        else:
            print(f"❌ 处理失败: {result['error']}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()