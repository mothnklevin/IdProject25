import os
import re
import copy
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import scipy.stats as stats
import gc
from collections import defaultdict


from DGP_1 import load_twins_X, generate_controlled_dgp, manual_1_dgp, generate_controlled_dgp_1
from DML_2 import run_doubleml_plr_rf
from EVAL_3 import evaluate_dml_results

# ------------------------ 单次运行函数 ------------------------
# -------------------- Single-run function -------------------
def run_single_setting(X_real, config_dict,USE_MANUAL=False):
    # config_dict = {k: v for k, v in config_dict.items() if
    #                k in ['nonlinearity', 'interaction', 'sparse_beta', 'skewness_level', 'heterogeneous', 'true_effect',
    #                      'noise_std', 'random_seed']}
    # 直接删除 config_name 字段
    dgp_config  = {k: v for k, v in config_dict.items() if k != 'config_name'}
    if USE_MANUAL == True:
        X, D, Y = generate_controlled_dgp_1(X_real=X_real, **dgp_config)
    else:
        X, D, Y = generate_controlled_dgp(X_real=X_real, **dgp_config)
    result = run_doubleml_plr_rf(X, D, Y)
    return result

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
    grouped = defaultdict(list)
    all_z_vals = []
    # for cfg, val in all_estimates:
    #     grouped[cfg].append(val)
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
        'nonlinearity': False,  # f(X) 为线性
        'interaction': False,  # 无 D·X 交互项
        'sparse_beta': False,  # 所有特征均参与生成 Y
        'skewness_level': 0.3,  # 倾向评分中 γ 方差较小 → D 分布较均匀
        'heterogeneous': False,  # θ 为常数 1.0
        'true_effect': 1.0,  # 设定真实因果效应
        'noise_std': 1.0,  # ε 的标准差
        'random_seed': 60  # 保证可复现性
    }

    named_configs = [
        ("0_基准", {}),# baseline
        ("1_非线性", {'nonlinearity': True}),# f(X) = X₀² + sin(X₁)
        ("2_交互", {'interaction': True}),# 加入 D·X₀ 交互项
        ("3_稀疏性", {'sparse_beta': True}),# 仅前 5 个 β 非零
        ("4_偏态", {'skewness_level': 2.0}),# D 明显偏态（logit(X@γ)极端）
        ("5_异质性", {'heterogeneous': True})# θ = 1.0 + 0.3·X₀
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
def run_experiments(X_real, configs,USE_MANUAL):
    all_summary = []
    all_estimates = []  # 用于绘制QQ图

    for config in configs:
        estimates = []
        std_errors = []
        for run in range(50):
            try:
                config_run = copy.deepcopy(config)
                config_run['random_seed'] = config['random_seed'] + run
                result = run_single_setting(X_real, config_run,USE_MANUAL)
                estimates.append(result['theta_hat'])
                std_errors.append(result['se'])
            except Exception as e:
                print(f"配置 {config['config_name']} 第 {run} 次运行失败：{e}")
                continue

        metrics = evaluate_dml_results(estimates, std_errors, true_theta=config['true_effect'])

        summary = {k: v for k, v in config.items() if k != 'random_seed'}
        summary.update(metrics)
        all_summary.append(summary)
        # all_estimates.extend(estimates) # 无颜色分组
        # for est in estimates: # 有颜色分组
        #     all_estimates.append((config['config_name'], est))
        for est, se in zip(estimates, std_errors): # 添加标准化参数
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
    cols = ['config_name', 'nonlinearity', 'interaction', 'sparse_beta', 'skewness_level', 'heterogeneous',
            'true_effect', 'noise_std',
            'bias', 'rmse', 'variance', 'coverage_rate', 'rejection_rate', 'mean_estimate']
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

    USE_MANUAL = True  # 切换数据加载模式
    # 载入数据
    if USE_MANUAL:
        X_real = manual_1_dgp(n_samples=800, seed=60)
    else:
        X_real = load_twins_X('./assets/twins/twin_pairs_X_3years_samesex.csv', n_samples=800, seed=60)
    # 载入配置
    configs = get_experiment_configs()

    # 运行实验
    df, all_estimates = run_experiments(X_real, configs, USE_MANUAL)
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
