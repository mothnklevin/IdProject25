import numpy as np
import math
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import defaultdict

from EVAL_3 import evaluate_dml_results
import warnings
# 忽略警告信息，保持输出整洁
warnings.filterwarnings('ignore')


# ------------------------ 可视化函数 visualization function------------------------
# 原始指标值合并图
def plot_raw_indicators(df, save_dir):
    # 选择基线
    baseline_row = df[df['config_name'] == '0_base'].iloc[0]
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
    baseline = df[df['config_name'] == '0_base'].iloc[0]
    metric_cols = ['bias', 'rmse', 'variance', 'coverage_rate', 'rejection_rate', 'mean_estimate']

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


def plot_normality_reject_rate_curve(
    df_rates: pd.DataFrame,
    out_dir: str,
    facet_by: list = ['config_name'],
    x: str = 'n_samples',
    ys: tuple = ('reject_rate_shapiro', 'reject_rate_ad', 'reject_rate_both')
):

    # 绘制正态性拒绝率曲线（Shapiro / AD / Both）。
    # - df_rates: 由 _6 计算得到的聚合结果，每行对应一个分组在某个 n_samples 下的拒绝率
    #   预期包含列：x（默认 n_samples）、ys（默认三个拒绝率列）以及 facet_by 指定的列
    # - out_dir: 输出目录（通常为 exp_i/DataAnalysis/）
    # - facet_by: 分面字段列表；每个唯一组合生成一张图（默认每个 config_name 一张）
    # - x: 横轴（样本量）；会尝试按数值排序
    # - ys: 要绘制的 y 列名元组

    # 基本检查
    if df_rates is None or df_rates.empty:
        print("[plot_normality_reject_rate_curve] 输入为空，跳过绘图。")
        return

    for col in [x, *ys, *facet_by]:
        if col not in df_rates.columns:
            print(f"[plot_normality_reject_rate_curve] 缺少必要列：{col}，跳过绘图。")
            return

    os.makedirs(out_dir, exist_ok=True)

    # 统一处理：将 x 转为数值便于排序（容忍无法转换的值）
    def _to_numeric_safe(s):
        try:
            return pd.to_numeric(s, errors='coerce')
        except Exception:
            return s

    df = df_rates.copy()
    df[x] = _to_numeric_safe(df[x])

    # 若 facet_by 为空或无效，则退化为单图
    if not facet_by:
        facet_by = []

    # 分面分组
    if facet_by:
        groups = df.groupby(facet_by, dropna=False, sort=False)
    else:
        # 统一成可迭代接口
        groups = [((), df)]

    for facet_vals, g in groups:
        # 排序
        g_sorted = g.sort_values(by=x, kind='mergesort')
        xs = g_sorted[x].values

        # 建图
        plt.figure(figsize=(7.5, 5))
        for y_col in ys:
            if y_col in g_sorted.columns:
                plt.plot(xs, g_sorted[y_col].values, marker='o', label=y_col)

        plt.ylim(0, 1)
        plt.xlabel(x)
        plt.ylabel("Reject Rate")
        # 标题与文件名后缀
        if facet_by:
            kv = [f"{k}={v}" for k, v in zip(facet_by, (facet_vals if isinstance(facet_vals, tuple) else (facet_vals,)))]
            title_suffix = " | ".join(kv)
            fname_suffix = "__" + "_".join([f"{k}-{str(v)}" for k, v in zip(facet_by, (facet_vals if isinstance(facet_vals, tuple) else (facet_vals,)))])
        else:
            title_suffix = "all"
            fname_suffix = "__all"

        plt.title(f"Normality Reject Rate Curves ({title_suffix})")
        plt.grid(alpha=0.25)
        plt.legend(frameon=False, fontsize=9)
        plt.tight_layout()

        safe_name = re.sub(r"[^\w\-_.\u4e00-\u9fa5]", "_", fname_suffix)
        out_path = os.path.join(out_dir, f"NormalityRejectCurve{safe_name}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"[Plot] {out_path}")




def main(root_dir, model_num = 1):
    # 设定实验根目录
    # root_dir = "./exp_4"
    print("model_num = ", model_num)

    # 1) 收集需要处理的目标目录
    targets = []
    # 情况A：根目录本身就有CSV（老结构）
    if os.path.exists(os.path.join(root_dir, "dml_experiment_summary.csv")):
        print(f"目录 {root_dir} 未清理所有CSV文件。")
        return
    else:
        # 情况B：新结构，遍历子文件夹
        if not os.path.isdir(root_dir):
            print(f"未找到目录：{root_dir}")
            return
        for name in sorted(os.listdir(root_dir)):
            sub = os.path.join(root_dir, name)
            if not os.path.isdir(sub):
                continue
            has_summary = os.path.exists(os.path.join(sub, "dml_experiment_summary.csv"))
            has_est = os.path.exists(os.path.join(sub, "dml_all_estimates.csv"))
            if has_summary and has_est:
                targets.append(sub)

    if not targets:
        print(f"在 {root_dir} 下未找到任何包含结果CSV的目录。")
        return

    if model_num == 1:
        # 2) 逐目录读取并作图
        for save_dir in targets:
            print(f"\n[可视化] 处理目录：{save_dir}")
            df = pd.read_csv(os.path.join(save_dir, "dml_experiment_summary.csv"))
            est_df = pd.read_csv(os.path.join(save_dir, "dml_all_estimates.csv"))

            all_estimates = list(zip(
                est_df['config_name'],
                est_df['theta_hat'],
                est_df['se'],
                est_df['true_effect']
            ))

            cfg_path = os.path.join(save_dir, "dml_configs.csv")
            if os.path.exists(cfg_path):
                cfg_df = pd.read_csv(cfg_path)
                print("Loaded configs (head):")
                print(cfg_df.head())
            else:
                print("提示：未发现 dml_configs.csv（不影响绘图）。")

            # 调用现有的四个可视化函数（输出仍保存在各自子目录）
            plot_raw_indicators(df, save_dir)
            plot_relative_differences(df, save_dir)
            plot_qq_distribution(all_estimates, save_dir)
            plot_hist_distribution(all_estimates, save_dir)

            print(f"图像已保存到: {save_dir}")
    elif model_num == 2:
        for save_dir in targets:
            print(f"\n[可视化] 处理目录：{save_dir}")

            # 2.1 读取逐 run 结果（all_estimates）
            est_path = os.path.join(save_dir, "dml_all_estimates.csv")
            est_df = pd.read_csv(est_path)

            # 2.2 动态识别参数列（排除逐次/中间列，其余视为参数常量列）
            true_col = 'true_theta' if 'true_theta' in est_df.columns else 'true_effect'

            exclude_cols = {
                'config_name', 'theta_hat', 'se', true_col, 'z',
                'run_id', 'seed_used'
            }
            param_cols = [c for c in est_df.columns if c not in exclude_cols]

            # 2.3 以 config_name 分组重算 summary
            rows = []
            for cfg, g in est_df.groupby('config_name', sort=False):
                estimates = g['theta_hat'].to_numpy()
                se = g['se'].to_numpy()

                true_theta = float(g[true_col].mean())

                metrics = evaluate_dml_results(estimates, se, true_theta=true_theta)

                # 组内参数常量：取第一行
                first = g.iloc[0]
                row = {'config_name': cfg}
                for c in param_cols:
                    row[c] = first[c] if c in g.columns else None
                row.update(metrics)
                rows.append(row)

            df = pd.DataFrame(rows)

            # 2.4 覆盖写回新的 summary（供 Raw/Relative 作图使用）
            sum_path = os.path.join(save_dir, "dml_experiment_summary.csv")
            df.to_csv(sum_path, index=False)

            # 2.5 可选：读取完整配置（如存在，仅打印头部）
            cfg_path = os.path.join(save_dir, "dml_configs.csv")
            if os.path.exists(cfg_path):
                cfg_df = pd.read_csv(cfg_path)
                print("Loaded configs (head):")
                print(cfg_df.head())
            else:
                print("提示：未发现 dml_configs.csv（不影响绘图）。")

            # 2.6 构造 all_estimates（用于 QQ / 直方图）
            all_estimates = list(zip(
                est_df['config_name'],
                est_df['theta_hat'],
                est_df['se'],
                est_df[true_col]
            ))

            # 2.7 调用现有的四个可视化函数（输出仍保存在各自子目录）
            plot_raw_indicators(df, save_dir)  # 基于新 summary
            plot_relative_differences(df, save_dir)  # 基于新 summary
            plot_qq_distribution(all_estimates, save_dir)  # 基于逐 run 动态 z
            plot_hist_distribution(all_estimates, save_dir)

            print(f"图像已保存到: {save_dir}")
    else:
        print(" wrong model_num ")




# 当以脚本方式运行时，调用主函数
# model_num =1： 直接读取并绘图； model_num =2： 重新计算并绘图
if __name__ == "__main__":
    results = main(root_dir = "./exp_5", model_num=1)
