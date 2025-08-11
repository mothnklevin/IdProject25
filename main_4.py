import os
import re
import copy
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import scipy.stats as stats
import gc
from collections import defaultdict

from DGP_1 import generate_dgp

from DML_2 import run_doubleml_plr_rf
from EVAL_3 import evaluate_dml_results


# 全局默认参数设置
N_SAMPLES = 800   # 生成数据的样本量（X 的行数）
D_DIM = 10        # 特征维度（X 的列数）
DGP_NUM = 0       # 选择结构：0=通用二元；1=兴趣-容忍度；2=CDDDHNR2018（连续处理）

# ------------------------ 单次运行函数 ------------------------
# -------------------- Single-run function -------------------
def run_single_setting(config_dict: dict, dgp_num: int = DGP_NUM):
    # 去掉仅用于标识/打印的字段
    seed = int(config_dict.get('random_seed', 42))
    cfg = {k: v for k, v in config_dict.items() if k not in ['config_name', 'random_seed']}
    # 生成数据
    X, D, Y = generate_dgp(n=N_SAMPLES, d=D_DIM, dgp_num=dgp_num, cfg=cfg, seed=seed)
    # 跑 DML 并返回字典结果
    return run_doubleml_plr_rf(X, D, Y)

# ------------------------ 可视化函数 visualization function------------------------
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

# ------------------------ 实验配置函数  configuration function ------------------------
def get_experiment_configs():
    default_config = { # baseline
        'nonlinearity': 0.0,  # g(X) 非线性强度
        'interaction': 0.0,  # D·X 交互强度
        'sparse_k': 0,  # 稀疏线性项个数（0=全维）
        'skewness': 0.3,  # 倾向分数 γ 的标准差（仅 dgp_num=0 使用）
        'heterogeneous': 0.0,  # θ 的异质性强度
        'true_effect': 1.0,  # θ 的基线值，设定真实因果效应
        'noise_std': 1.0,  # 结果噪声ε的标准差
        'random_seed': 60,  # 复现用种子（每次 run 会 +run 累加）
        # 可选索引（不设则默认 0）
        # 'interaction_idx': 0,
        # 'hetero_idx': 0,
        # v1 结构可选背景列范围（不设则走默认 [2,5)）
        # 'bg_start': 2,
        # 'bg_end': 5,
        # v2 结构可选参数（不设则走默认）
        # 'rho': 0.7, 's1': 1.0, 'a0': 1.0, 'a1': 0.25, 'b0': 1.0, 'b1': 0.25,
    }

    named_configs = [
        ("0_基准", {}),# baseline
        ("1_非线性", {'nonlinearity': 1.0}),  # 加强 g(X) 的非线性项
        ("2_交互", {'interaction': 0.5}),  # 加入 D·X 的交互项（强度 0.5）
        ("3_稀疏性", {'sparse_k': 5}),  # 仅前 5 个 β 非零
        ("4_偏态", {'skewness': 2.0}),  # 更极端的倾向分布
        ("5_异质性", {'heterogeneous': 0.3}),  # θ = θ0 + 0.3·X
        # ,
        # ("6_非线性+异质性", {
        #     'nonlinearity': True,
        #     'heterogeneous': True
        # }),
        # ("7_稀疏+偏态", {
        #     'sparse_beta': True,
        #     'skewness_level': 2.0
        # }),
        # ("8_非线性+交互", {
        #     'nonlinearity': True,
        #     'interaction': True
        # })
    ]

    merged_configs = []
    for i, (cfg_name, cfg_update) in enumerate(named_configs):
        config = copy.deepcopy(default_config)
        config.update(cfg_update)
        config['random_seed'] += i
        config['config_name'] = cfg_name
        merged_configs.append(config)

    return merged_configs

# ------------------------ 实验执行函数 execution function------------------------
def run_experiments(configs, dgp_num: int = DGP_NUM):
    all_summary = []
    all_estimates = []  # 用于绘制QQ图

    for config in configs:
        estimates = []
        std_errors = []
        for run in range(50):
            try:
                config_run = copy.deepcopy(config)
                config_run['random_seed'] = config['random_seed'] + run
                result = run_single_setting(config_run, dgp_num=dgp_num)
                estimates.append(result['theta_hat'])
                std_errors.append(result['se'])
            except Exception as e:
                print(f"配置 {config['config_name']} 第 {run} 次运行失败：{e}")
                continue

        metrics = evaluate_dml_results(estimates, std_errors, true_theta=config['true_effect'])

        summary = {k: v for k, v in config.items() if k != 'random_seed'}
        summary.update(metrics)
        all_summary.append(summary)

        for est, se in zip(estimates, std_errors): # 有颜色分组，添加标准化参数
            all_estimates.append((config['config_name'], est, se, config['true_effect']))

        # 输出配置结果
        print(f"\n=== {config['config_name']} ===")
        print("Bias: {:.4f}".format(metrics['bias']))
        print("RMSE: {:.4f}".format(metrics['rmse']))
        print("Variance: {:.4f}".format(metrics['variance']))
        print("Coverage Rate: {:.2%}".format(metrics['coverage_rate']))
        print("Rejection Rate: {:.2%}".format(metrics['rejection_rate']))
        print("Mean Estimate: {:.4f}".format(metrics['mean_estimate']))

        # 手动释放内存
        gc.collect()

    df = pd.DataFrame(all_summary)

    # 指定输出列顺序
    cols = ['config_name', 'nonlinearity', 'interaction', 'sparse_k', 'skewness', 'heterogeneous',
            'true_effect', 'noise_std', 'bias', 'rmse', 'variance', 'coverage_rate', 'rejection_rate', 'mean_estimate']
    df = df[cols]
    return df, all_estimates

# ------------------------ 主函数：运行实验，保存结果并打印输出 ------------------------
# -- main function: Run, save the results and print the output--
def main():
    # 自动创建结果文件夹 exp_i
    existing = [int(re.findall(r'exp_(\d+)', d)[0]) for d in os.listdir('.') if re.match(r'exp_\d+', d)]
    exp_id = max(existing)+1 if existing else 1
    save_dir = f"exp_{exp_id}"
    os.makedirs(save_dir, exist_ok=True)

    # 选择结构：0/1/2；也可以在 CLI/环境变量里外部注入
    # dgp_num = DGP_NUM
    dgp_num = 2

    # 载入配置
    configs = get_experiment_configs()

    # 运行实验
    df, all_estimates = run_experiments(configs, dgp_num=dgp_num)
    df.to_csv(os.path.join(save_dir, "dml_experiment_summary.csv"), index=False)

    # 可视化
    plot_relative_differences(df, save_dir)
    plot_qq_distribution(all_estimates, save_dir)
    print(f"\n结果与图像已保存至文件夹：{save_dir}")

# ------------------------ 调用主函数 ------------------------
# 忽略警告信息，保持输出整洁
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    main()
