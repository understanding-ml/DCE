"""
重新计算DCE_Results中每个实验的metrics
基于best_x, best_y, x_true, y_target重新计算所有指标
参考baseline_experiments_compas.ipynb, baseline_experiments_heloc.ipynb和demo_new.ipynb

Usage:
    python recalculate_dce_metrics_v2.py [--root DCE_Results] [--out DCE_Results]

Output:
    - 每个数据集一个Excel文件: recalculated_metrics_{dataset}.xlsx
    - 所有数据集合并文件: recalculated_metrics_all.xlsx
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from scipy.stats import gaussian_kde, entropy
from numpy.linalg import LinAlgError
from tqdm import tqdm

# Import distance functions
try:
    from explainers.distances import SlicedWassersteinDivergence, WassersteinDivergence
    TORCH_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not import distance functions: {e}")
    TORCH_AVAILABLE = False


# ==================== Dataset-specific configurations ====================

DATASET_CONFIGS = {
    'compas': {
        'continuous_features': ['Priors_Count', 'Time_Served'],
        'n_features': 15
    },
    'heloc': {
        'continuous_features': ['ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen',
                               'AverageMInFile', 'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec',
                               'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq', 'MSinceMostRecentDelq',
                               'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumTotalTrades',
                               'NumTradesOpeninLast12M', 'PercentInstallTrades', 'MSinceMostRecentInqexcl7days',
                               'NumInqLast6M', 'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden',
                               'NetFractionInstallBurden', 'NumRevolvingTradesWBalance',
                               'NumInstallTradesWBalance', 'NumBank2NatlTradesWHighUtilization',
                               'PercentTradesWBalance'],
        'n_features': 23
    },
    'german_credit': {
        'continuous_features': ['Duration-in-Month', 'Credit-Amount', 'Age'],
        'n_features': 20
    },
    'hotel_booking': {
        'continuous_features': ['lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month',
                               'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children',
                               'babies', 'previous_cancellations', 'previous_bookings_not_canceled',
                               'booking_changes', 'days_in_waiting_list', 'adr',
                               'required_car_parking_spaces', 'total_of_special_requests'],
        'n_features': 31
    }
}


# ==================== Metric Calculation Functions ====================

def compute_ot_distance(X_s: np.ndarray, X_t: np.ndarray) -> float:
    """
    计算OT距离 (Sliced Wasserstein Distance)
    使用标准化数据，参考baseline notebooks
    """
    if not TORCH_AVAILABLE:
        # Fallback to simple per-feature wasserstein
        from scipy.stats import wasserstein_distance
        dists = [wasserstein_distance(X_s[:, j], X_t[:, j]) for j in range(X_s.shape[1])]
        return float(np.mean(dists)) if dists else np.nan

    try:
        # Convert to torch tensors
        X_s_tensor = torch.FloatTensor(X_s)
        X_t_tensor = torch.FloatTensor(X_t)

        if X_s_tensor.ndim == 1 or X_s_tensor.shape[1] == 1:
            # 1D case: use Wasserstein Distance
            wd = WassersteinDivergence()
            distance, _ = wd.distance(X_s_tensor.view(-1), X_t_tensor.view(-1), delta=0.1)
        else:
            # Multi-dimensional: use Sliced Wasserstein Distance
            swd = SlicedWassersteinDivergence(dim=X_s_tensor.shape[1], n_proj=5000)
            distance, _ = swd.distance(X_s_tensor, X_t_tensor, delta=0.1)

        return float(distance.item())
    except Exception as e:
        print(f"    OT distance error: {e}")
        return np.nan


def gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """高斯核函数"""
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))


def compute_mmd(X_s: np.ndarray, X_t: np.ndarray, sigma: float = 1.0) -> float:
    """
    计算MMD (Maximum Mean Discrepancy)
    使用标准化数据，参考baseline notebooks
    """
    try:
        n = X_s.shape[0]
        m = X_t.shape[0]

        XX = np.sum([gaussian_kernel(X_s[i], X_s[j], sigma) for i in range(n) for j in range(n)])
        YY = np.sum([gaussian_kernel(X_t[i], X_t[j], sigma) for i in range(m) for j in range(m)])
        XY = np.sum([gaussian_kernel(X_s[i], X_t[j], sigma) for i in range(n) for j in range(m)])

        return float(XX / (n ** 2) + YY / (m ** 2) - 2 * XY / (n * m))
    except Exception as e:
        print(f"    MMD error: {e}")
        return np.nan


def compute_kl_divergence(X_s: np.ndarray, X_t: np.ndarray) -> float:
    """
    计算KL散度，使用KDE估计分布
    对每个特征计算KL散度，然后求和
    """
    kl_divergences = []
    for i in range(X_s.shape[1]):
        try:
            kde_s = gaussian_kde(X_s[:, i])
            kde_t = gaussian_kde(X_t[:, i])

            x_min = min(X_s[:, i].min(), X_t[:, i].min())
            x_max = max(X_s[:, i].max(), X_t[:, i].max())
            x = np.linspace(x_min, x_max, 1000)

            kl_div = entropy(kde_s(x), kde_t(x))
        except (LinAlgError, ValueError):
            kl_div = np.inf

        kl_divergences.append(kl_div)

    return float(np.sum(kl_divergences))


def build_cost_vector(df_standardized: pd.DataFrame, mean_vals: pd.Series,
                      std_vals: pd.Series) -> np.ndarray:
    """
    构建AReS成本向量
    基于原始数据的range计算（参考demo_new实现）

    Parameters:
    - df_standardized: 标准化后的数据
    - mean_vals: 原始数据的均值
    - std_vals: 原始数据的标准差

    Returns:
    - costs_vector: 特征成本向量
    """
    # 恢复到原始尺度
    df_original = df_standardized * std_vals + mean_vals

    cost_list = []
    for col in df_original.columns:
        values = df_original[col].values
        unique_count = np.unique(values).size
        ratio = unique_count / len(values)

        if unique_count <= 10 and ratio < 0.5:
            # 分类特征 → 固定成本
            cost_list.append(0.5)
        else:
            # 连续特征 → range的倒数
            val_range = values.max() - values.min()
            cost_list.append(1.0 / val_range if val_range > 0 else 1.0)

    return np.array(cost_list)


def compute_ares_cost(delta: np.ndarray, costs_vector: np.ndarray) -> float:
    """
    计算AReS成本 (加权L2范数)
    delta: 标准化数据的差异
    costs_vector: 特征成本向量
    """
    try:
        return float(np.linalg.norm(delta @ np.diag(costs_vector)))
    except Exception as e:
        print(f"    Cost error: {e}")
        return np.nan


def compute_categorical_difference(X_cf: pd.DataFrame, X_f: pd.DataFrame,
                                   categorical_columns: List[str]) -> float:
    """
    计算分类特征的平均绝对差异
    """
    try:
        diff_list = []
        for column in categorical_columns:
            if column in X_cf.columns and column in X_f.columns:
                diff_list.append((X_cf[column] - X_f[column]).abs().mean())

        return float(np.nanmean(diff_list)) if diff_list else np.nan
    except Exception as e:
        print(f"    Categorical diff error: {e}")
        return np.nan


def compute_percentile_difference(X_cf: np.ndarray, X_f: np.ndarray,
                                  percentiles: np.ndarray) -> float:
    """
    计算百分位差异
    使用标准化数据
    采用demo_new的方式：对接近0的分母做特殊处理，避免极大值
    """
    try:
        diff_list = []
        for j in range(X_cf.shape[1]):
            perc_cf = np.percentile(X_cf[:, j], percentiles)
            perc_f = np.percentile(X_f[:, j], percentiles)

            # 使用demo_new的方式处理零值
            # 对每个百分位值判断是否接近0
            for k in range(len(percentiles)):
                if abs(perc_f[k]) > 1e-8:
                    # 正常计算相对差异
                    diff = abs(perc_cf[k] - perc_f[k]) / abs(perc_f[k]) * 100.0
                else:
                    # 当分母接近0时，返回0或100而不是极大值
                    diff = 0.0 if abs(perc_cf[k]) < 1e-8 else 100.0

                diff_list.append(diff)

        if not diff_list:
            return np.nan

        # 返回所有特征所有百分位的平均差异
        return float(np.mean(diff_list))
    except Exception as e:
        print(f"    Percentile diff error: {e}")
        return np.nan


def compute_statistic_difference(X_cf: pd.DataFrame, X_f: pd.DataFrame,
                                metric: str, columns: List[str]) -> float:
    """
    计算统计指标差异 (mean或std)
    使用原始数据
    """
    try:
        diff_list = []
        for column in columns:
            if column in X_cf.columns and column in X_f.columns:
                val_cf = X_cf[column].agg(metric)
                val_f = X_f[column].agg(metric)
                if abs(val_f) > 1e-10:
                    diff_list.append(abs(val_cf - val_f) / abs(val_f) * 100)

        return float(np.nanmean(diff_list)) if diff_list else np.nan
    except Exception as e:
        print(f"    {metric} diff error: {e}")
        return np.nan


def compute_diversity(X: np.ndarray) -> float:
    """
    计算平均成对距离 (Diversity)
    使用原始数据
    """
    try:
        n = len(X)
        if n <= 1:
            return 0.0

        total_distance = 0.0
        count = 0

        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(X[i] - X[j])
                total_distance += dist
                count += 1

        return float(total_distance / count) if count > 0 else 0.0
    except Exception as e:
        print(f"    Diversity error: {e}")
        return np.nan


def compute_effective_diversity(diversity: float, ot_distance: float, coverage: float) -> float:
    """
    计算有效多样性
    effective_diversity = (diversity / ot_distance) * coverage
    """
    try:
        if ot_distance > 0 and not np.isnan(ot_distance) and not np.isinf(ot_distance):
            return float((diversity / ot_distance) * coverage)
        else:
            return np.nan
    except Exception as e:
        print(f"    Effective diversity error: {e}")
        return np.nan


# ==================== Data Loading and Processing ====================

def load_json(path: Path) -> Dict[str, Any]:
    """加载JSON文件"""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}")
        return {}


def load_csv_array(filepath: Path) -> Optional[np.ndarray]:
    """加载CSV文件并转换为numpy数组，自动检测和跳过表头"""
    if not filepath.exists():
        return None
    try:
        # 先读取第一行检查是否有表头
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()

        # 检查第一行是否包含非数字字符（除了逗号、点、负号）
        # 如果包含字母，则认为是表头
        has_header = any(c.isalpha() for c in first_line)

        # 根据是否有表头来读取
        if has_header:
            data = pd.read_csv(filepath, header=0)
        else:
            data = pd.read_csv(filepath, header=None)

        # 转换为float数组
        return data.values.astype(float)
    except Exception as e:
        # 如果转换失败，尝试跳过问题行
        try:
            data = pd.read_csv(filepath, header=0, skiprows=lambda x: x > 0 and not all(c.replace('.', '').replace('-', '').replace(',', '').isdigit() or c.isspace() for c in str(x)))
            return data.values.astype(float)
        except:
            print(f"    Error loading {filepath}: {e}")
            return None


def load_experiment_data(run_dir: Path) -> Tuple[Optional[np.ndarray], ...]:
    """
    加载实验数据
    返回: x_true, y_true, y_target, best_x, best_y
    """
    x_true = load_csv_array(run_dir / "x_true.csv")
    y_true = load_csv_array(run_dir / "y_true.csv")
    y_target = load_csv_array(run_dir / "y_target.csv")

    # 优先使用best_x/best_y，如果不存在则使用final_x/final_y
    best_x = load_csv_array(run_dir / "best_x.csv")
    if best_x is None:
        best_x = load_csv_array(run_dir / "final_x.csv")

    best_y = load_csv_array(run_dir / "best_y.csv")
    if best_y is None:
        best_y = load_csv_array(run_dir / "final_y.csv")

    return x_true, y_true, y_target, best_x, best_y


def extract_experiment_info(run_dir: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """提取实验信息"""
    parts = run_dir.parts

    # 从路径中提取信息
    try:
        dce_results_idx = parts.index("DCE_Results")
        dataset = parts[dce_results_idx + 1]
        model = parts[dce_results_idx + 2]
    except (ValueError, IndexError):
        dataset = metadata.get("dataset_name", "unknown")
        model = metadata.get("model_name", "unknown")

    strategy = metadata.get("strategy", "unknown")
    seed = metadata.get("seed", "unknown")

    info = {
        'dataset': dataset,
        'model': model,
        'strategy': strategy,
        'seed': seed,
        'run_dir': str(run_dir),
        'timestamp': metadata.get('timestamp', ''),
    }

    # 添加DCE参数
    dce_params = metadata.get("dce_parameters", {})
    for key, value in dce_params.items():
        info[f'dce_{key}'] = value

    # 添加策略参数
    strategy_params = metadata.get("strategy_parameters", {})
    for key, value in strategy_params.items():
        info[f'strategy_{key}'] = value

    return info


def recalculate_metrics(run_dir: Path) -> Dict[str, Any]:
    """
    重新计算所有metrics

    返回包含所有metrics的字典
    """
    # 加载元数据
    metadata = load_json(run_dir / "metadata.json")

    # 加载实验数据
    x_true, y_true, y_target, best_x, best_y = load_experiment_data(run_dir)

    # 检查是否找到可行解
    found_feasible = metadata.get("found_feasible_solution", False)

    # 初始化结果字典
    metrics = {}
    metrics['found_feasible_solution'] = found_feasible

    # 如果没有best_x或best_y，标记为失败
    if best_x is None or best_y is None:
        metrics['is_feasible'] = False
        metrics['error'] = 'missing best_x or best_y'
        return metrics

    metrics['is_feasible'] = True

    # 获取数据集配置
    dataset_name = metadata.get("dataset_name", "compas")
    config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS['compas'])
    continuous_features = config['continuous_features']

    # ==================== 1. Coverage Rate ====================
    metrics['coverage_rate'] = float(np.mean(best_y > 0.5))

    # ==================== 如果缺少x_true，只计算覆盖率 ====================
    if x_true is None:
        metrics['error'] = 'missing x_true'
        return metrics

    # 检查形状是否匹配
    if best_x.shape[1] != x_true.shape[1]:
        metrics['error'] = f'shape mismatch: best_x {best_x.shape} vs x_true {x_true.shape}'
        return metrics

    n_features = best_x.shape[1]
    feature_names = [f'feature_{i}' for i in range(n_features)]

    # 创建DataFrames（原始尺度数据）
    df_cf_original = pd.DataFrame(best_x, columns=feature_names)
    df_f_original = pd.DataFrame(x_true, columns=feature_names)

    # 识别分类特征（除了连续特征以外的）
    categorical_features = [f for f in feature_names if f not in continuous_features]

    # 计算均值和标准差（基于x_true）用于标准化
    mean_vals = pd.Series(x_true.mean(axis=0), index=feature_names)
    std_vals = pd.Series(x_true.std(axis=0), index=feature_names)
    std_vals = std_vals.replace(0, 1.0)  # 防止除以0

    # 标准化数据（OT、MMD需要）
    best_x_std = (best_x - mean_vals.values) / std_vals.values
    x_true_std = (x_true - mean_vals.values) / std_vals.values

    # ==================== 2. OT Distance (使用标准化数据) ====================
    metrics['ot_distance'] = compute_ot_distance(best_x_std, x_true_std)

    # ==================== 3. MMD (使用标准化数据) ====================
    metrics['mmd'] = compute_mmd(best_x_std, x_true_std, sigma=1.0)

    # ==================== 4. KL Divergence (使用标准化数据) ====================
    metrics['kl_divergence'] = compute_kl_divergence(best_x_std, x_true_std)

    # ==================== 5. AReS Cost ====================
    # 基于原始尺度数据构建cost向量（demo_new方式）
    try:
        costs_vector = build_cost_vector(df_f_original, mean_vals, std_vals)
    except Exception as e:
        print(f"    Error building cost vector: {e}, using unit costs")
        costs_vector = np.ones(n_features)

    # delta在标准化数据上计算
    delta = best_x_std - x_true_std
    metrics['ares_cost'] = compute_ares_cost(delta, costs_vector)

    # ==================== 6. Categorical Difference ====================
    metrics['categorical_difference'] = compute_categorical_difference(df_cf_original, df_f_original, categorical_features)

    # ==================== 7. Percentile Differences (使用标准化数据) ====================
    percentile_ranges = [
        (np.arange(0, 15, 0.1), '0-15%'),
        (np.arange(15, 30, 1), '15-30%'),
        (np.arange(30, 70, 1), '30-70%'),
        (np.arange(70, 85, 1), '70-85%'),
        (np.arange(85, 100, 1), '85-100%')
    ]

    for percentiles, label in percentile_ranges:
        key = f'percentile_diff_{label}'
        metrics[key] = compute_percentile_difference(best_x_std, x_true_std, percentiles)

    # ==================== 8. Statistical Differences (连续特征，使用原始数据) ====================
    continuous_cols = [f for f in feature_names if f in continuous_features]

    metrics['mean_diff_pct'] = compute_statistic_difference(df_cf_original, df_f_original, 'mean', continuous_cols)
    metrics['std_diff_pct'] = compute_statistic_difference(df_cf_original, df_f_original, 'std', continuous_cols)

    # ==================== 9. Diversity (使用原始数据) ====================
    metrics['diversity'] = compute_diversity(best_x)

    # ==================== 10. Effective Diversity ====================
    metrics['effective_diversity'] = compute_effective_diversity(
        metrics['diversity'],
        metrics['ot_distance'],
        metrics['coverage_rate']
    )

    # ==================== 11. 其他元数据 ====================
    metrics['best_iteration'] = metadata.get('best_iter')
    metrics['total_iterations'] = metadata.get('total_iterations')
    metrics['optimization_time_seconds'] = metadata.get('optimization_time_seconds')
    metrics['n_samples'] = len(best_x)

    return metrics


# ==================== Main Processing ====================

def collect_all_experiments(root_dir: Path) -> List[Dict[str, Any]]:
    """
    收集所有实验结果并计算metrics
    """
    all_results = []

    # 查找所有seed_*目录
    seed_dirs = list(root_dir.rglob("seed_*"))

    print(f"\nFound {len(seed_dirs)} experiments to process")
    print("=" * 80)

    for run_dir in tqdm(seed_dirs, desc="Processing experiments"):
        # 加载元数据
        metadata = load_json(run_dir / "metadata.json")

        # 提取实验信息
        exp_info = extract_experiment_info(run_dir, metadata)

        # 重新计算metrics
        try:
            metrics = recalculate_metrics(run_dir)

            # 合并信息和metrics
            result = {**exp_info, **metrics}
            all_results.append(result)

        except Exception as e:
            print(f"\n  Error processing {run_dir}: {e}")
            continue

    print(f"\n\nSuccessfully processed {len(all_results)} experiments")
    return all_results


def save_results(results: List[Dict[str, Any]], output_dir: Path):
    """
    保存结果到Excel文件（如果openpyxl可用）或CSV文件
    - 每个数据集一个文件
    - 所有数据集合并到一个文件（多个sheet或一个CSV）
    """
    if not results:
        print("No results to save!")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 转换为DataFrame
    df = pd.DataFrame(results)

    # 按数据集分组
    datasets = df['dataset'].unique()

    # 检查是否可用Excel格式
    try:
        import openpyxl
        use_excel = True
        file_ext = 'xlsx'
        print(f"\nSaving results to Excel files...")
    except ImportError:
        use_excel = False
        file_ext = 'csv'
        print(f"\nopenpyxl not available, saving results to CSV files...")

    print("=" * 80)

    # 保存每个数据集单独的文件
    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset].copy()
        dataset_df = dataset_df.sort_values(['model', 'strategy', 'seed'])

        output_file = output_dir / f'recalculated_metrics_{dataset}.{file_ext}'

        if use_excel:
            dataset_df.to_excel(output_file, index=False, sheet_name=dataset)
        else:
            dataset_df.to_csv(output_file, index=False)

        print(f"  Saved {dataset}: {len(dataset_df)} experiments → {output_file}")

    # 保存合并文件
    if use_excel:
        # Excel格式：所有数据集，每个数据集一个sheet
        combined_file = output_dir / f'recalculated_metrics_all.{file_ext}'

        with pd.ExcelWriter(combined_file, engine='openpyxl') as writer:
            for dataset in datasets:
                dataset_df = df[df['dataset'] == dataset].copy()
                dataset_df = dataset_df.sort_values(['model', 'strategy', 'seed'])
                dataset_df.to_excel(writer, sheet_name=dataset, index=False)

        print(f"  Saved combined: {len(df)} experiments → {combined_file}")
    else:
        # CSV格式：单个文件包含所有数据集
        combined_file = output_dir / f'recalculated_metrics_all.{file_ext}'
        df_sorted = df.sort_values(['dataset', 'model', 'strategy', 'seed'])
        df_sorted.to_csv(combined_file, index=False)

        print(f"  Saved combined: {len(df)} experiments → {combined_file}")

    print("\n" + "=" * 80)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Recalculate metrics for DCE experiments"
    )
    parser.add_argument(
        '--root',
        type=Path,
        default=Path('DCE_Results'),
        help='Root directory containing DCE experiment results'
    )
    parser.add_argument(
        '--out',
        type=Path,
        default=Path('DCE_Results'),
        help='Output directory for Excel files'
    )
    return parser.parse_args()


# ==================== Main Entry Point ====================

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("DCE Results Metrics Recalculation (v2)")
    print("=" * 80)

    args = parse_args()

    # 收集所有实验结果
    results = collect_all_experiments(args.root)

    if results:
        # 保存结果
        save_results(results, args.out)

        # 打印摘要统计
        df = pd.DataFrame(results)
        print("\nResults Summary:")
        print(df.groupby(['dataset', 'model', 'strategy']).size())
        print(f"\nFeasible solutions: {df['is_feasible'].sum()} / {len(df)}")

        print("\n✅ All metrics recalculated and saved successfully!")
    else:
        print("\n⚠️  No valid results found!")
