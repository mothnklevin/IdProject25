import numpy as np
import math

import os
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import defaultdict


# ------------------------ 可视化函数 visualization function------------------------
# 原始指标值合并图
def plot_raw_indicators(df, save_dir):
    # 选择基线
    baseline_row = df[df['config_name'] == '0_基准'].iloc[0]
    metric_cols = ['bias', 'rmse', 'variance', 'coverage_rate', 'rejection_rate', 'mean_estimate']

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = plt.cm.tab10.colors
    cfg_labels = df['config_name']

    for idx, metric in enumerate(metric_cols):
        ax = axes[idx // 3, idx % 3]
        vals = df[metric].values
        bars = ax.bar(cfg_labels, vals, color=[colors[i % 10] for i in range(len(cfg_labels))])

        # 数值标注
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.4f}", ha='center', va='bottom', fontsize=8)

        # 基线虚线
        base_v = baseline_row[metric]
        ax.axhline(base_v, color='black', linewidth=0.9, linestyle='--', label='baseline')

        ax.set_title(metric)
        ax.set_xticks(range(len(cfg_labels)))
        ax.set_xticklabels(cfg_labels, rotation=15, fontsize=9)
        ax.legend(fontsize=8, frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "indicator_Raw.png"))
    plt.close()


def plot_relative_differences(df, save_dir):
    # baseline = df[df['config_name'].str.contains("基准")].iloc[0]
    baseline = df[df['config_name'] == '0_基准'].iloc[0]
    metric_cols = ['bias', 'rmse', 'variance', 'coverage_rate', 'rejection_rate', 'mean_estimate']

    # # 单图输出
    # for metric in metric_cols:
    #     plt.figure(figsize=(8, 5))
    #     diffs = df[metric] - baseline[metric]
    #     colors = ['red' if val > 0 else 'blue' for val in diffs]
    #     config_labels = df['config_name']
    #     plt.bar(config_labels, diffs, color=colors)
    #     for i, val in enumerate(diffs):
    #         plt.text(i, val, f"{val:.4f}", ha='center', va='bottom' if val > 0 else 'top')
    #     plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    #     plt.ylabel(f"{metric} 变化量")
    #     plt.title(f"相对于基准的 {metric} 变化")
    #     plt.xticks(rotation=15)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(save_dir, f"{metric}.png"))
    #     plt.close()

    # 总图输出
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for idx, metric in enumerate(metric_cols):
        ax = axes[idx // 3, idx % 3]
        diffs = df[metric] - baseline[metric]
        colors = ['red' if val > 0 else 'blue' for val in diffs]
        config_labels = df['config_name']
        bars = ax.bar(config_labels, diffs, color=colors)
        for i, val in enumerate(diffs):
            ax.text(i, val, f"{val:.4f}", ha='center', va='bottom' if val > 0 else 'top', fontsize=8)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title(metric)
        ax.set_xticks(range(len(config_labels)))
        ax.set_xticklabels(config_labels, rotation=15, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "indicator_Summary.png"))
    plt.close()

# ------------------------ QQ图绘制函数 QQ graph drawing function------------------------
def plot_qq_distribution(all_estimates, save_dir):
    if not all_estimates:  # 无成功结果就直接跳过
        print("No estimates collected; skip QQ plot.")
        return

    grouped = defaultdict(list)
    all_z_vals = []
    for cfg, theta, se, true_effect in all_estimates:
        z = (theta - true_effect) / se
        grouped[cfg].append(z)
        all_z_vals.append(z)

    # 动态范围设置
    z_min, z_max = min(all_z_vals), max(all_z_vals)
    margin = 0.2
    x_min = z_min - margin
    x_max = z_max + margin

    plt.figure(figsize=(7, 7))
    colors = plt.cm.tab10.colors  # 最多支持10种配置颜色

    for i, (cfg, vals) in enumerate(grouped.items()):
        osm, osr = stats.probplot(vals, dist="norm", fit=False)
        plt.scatter(osm, osr, label=cfg, color=colors[i % 10])

    # plt.plot([-2, 2], [-2, 2], 'r--', label='y=x')
    plt.plot([x_min, x_max], [x_min, x_max], 'r--', label='y=x')
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Ordered Values")
    plt.title("Quantile-Quantile Plot of Theta Estimates by Config")
    plt.legend()


    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Quantile-Quantile.png"))
    plt.close()

def plot_hist_distribution(all_estimates, save_dir, bins=12):
    # 按配置绘制 z = (theta_hat - true_effect) / se 的直方图。
    # 输出为一张图（多子图），同轴对比更清晰；并叠加标准正态密度曲线作参考。
    if not all_estimates:
        print("No estimates collected; skip histogram.")
        return

    # 把估计值标准化为 z 统计量, 分组并计算 z
    grouped = defaultdict(list)
    for cfg, theta, se, true_effect in all_estimates:
        if se == 0 or np.isnan(se):
            continue
        z = (theta - true_effect) / se
        # 过滤掉 se=0、NaN 或无穷值
        if np.isfinite(z):
            grouped[cfg].append(z)

    if not grouped:
        print("No valid z values; skip histogram.")
        return

    # 统一所有配置的直方图 bin 范围
    all_z = np.concatenate([np.asarray(v) for v in grouped.values()])
    z_min, z_max = np.min(all_z), np.max(all_z) # 查找全局最小/最大值
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min == z_max: # 若范围无效
        z_min, z_max = -3.5, 3.5 # 强制使用默认值
    edges = np.linspace(z_min, z_max, bins + 1) # 直方图分箱边界
    x_grid = np.linspace(z_min, z_max, 400) # 用于绘制参考曲线和核密度

    # 子图布局（单张图）
    cfg_names = list(grouped.keys())
    n_cfg = len(cfg_names)
    n_cols = min(3, n_cfg)
    n_rows = math.ceil(n_cfg / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False)
    colors = plt.cm.tab10.colors

    # 标准正态密度
    norm_pdf = stats.norm.pdf(x_grid, loc=0, scale=1)

    for i, cfg in enumerate(cfg_names):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]
        zs = np.asarray(grouped[cfg], dtype=float)

        # 直方图（密度归一）
        ax.hist(zs, bins=edges, density=True, alpha=0.65, label=f"{cfg} (n={len(zs)})",
                color=colors[i % 10], edgecolor='white', linewidth=0.6)

        # 叠加核密度（可选：样本少时会较抖动）
        if len(zs) >= 5:
            try:
                kde = stats.gaussian_kde(zs)
                ax.plot(x_grid, kde(x_grid), linewidth=1.4, alpha=0.9, color=colors[i % 10])
            except Exception:
                pass

        # 叠加标准正态参考
        ax.plot(x_grid, norm_pdf, 'k--', linewidth=1.0, label='N(0,1)')

        ax.set_title(cfg)
        ax.set_xlabel("z = (theta_hat - true)/se")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, frameon=False)

    # 清空多余子图
    for j in range(n_cfg, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        fig.delaxes(axes[r][c])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Histograms_by_Config.png"))
    plt.close()

import pandas as pd
def main():
    # ======== 1. 设置实验结果目录 ========
    save_dir = "./exp_6"

    # ======== 2. 读取已保存的 CSV ========
    df = pd.read_csv(os.path.join(save_dir, "dml_experiment_summary.csv"))
    est_df = pd.read_csv(os.path.join(save_dir, "dml_all_estimates.csv"))
    # 转成 [(cfg, theta, se, true_effect), ...] 列表
    all_estimates = list(zip(
        est_df['config_name'],
        est_df['theta_hat'],
        est_df['se'],
        est_df['true_effect']
    ))

    # 完整配置表（此处暂时不直接用于作图，但可用作检查）
    cfg_df = pd.read_csv(os.path.join(save_dir, "dml_full_configs.csv"))
    print("Loaded configs:")
    print(cfg_df.head())

    # ======== 3. 调用已有的可视化函数 ========
    plot_relative_differences(df, save_dir)  # 相对基准变化
    plot_qq_distribution(all_estimates, save_dir)  # QQ 图
    plot_hist_distribution(all_estimates, save_dir)  # z 值直方图

    print(f"复现图像已保存到: {save_dir}")

# 当以脚本方式运行时，调用主函数
if __name__ == "__main__":
    results = main()
