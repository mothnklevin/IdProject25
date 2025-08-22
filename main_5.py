import os
import re
import copy
import pandas as pd
import warnings
import gc

from itertools import product

from DGP_1 import generate_dgp
from DML_2 import run_doubleml_plr_rf
from EVAL_3 import evaluate_dml_results
from VISUAL_4 import plot_relative_differences, plot_qq_distribution, plot_hist_distribution, plot_raw_indicators



# ------------------------ 单次运行函数 ------------------------
# -------------------- Single-run function -------------------
# def run_single_setting(config_dict: dict, dgp_num: int = 0 ):
def run_single_setting(config_dict: dict, dgp_num: int = 0, n_samples: int = 800, d_dim: int = 10):
    # 去掉仅用于标识/打印的字段
    seed = int(config_dict.get('random_seed', 60))
    cfg = {k: v for k, v in config_dict.items() if k not in ['config_name', 'random_seed']}
    # 生成数据
    X, D, Y = generate_dgp(n=n_samples, d=d_dim, dgp_num=dgp_num, cfg=cfg, seed=seed)
    # 跑 DML 并返回字典结果
    return run_doubleml_plr_rf(X, D, Y)

# ------------------------ 实验配置函数  configuration function ------------------------
def get_experiment_configs():
    default_config = { # baseline
        # 0/1
        'nonlinearity': 0.0,  # g(X) 非线性强度
        'sparse_k': 0,  # 稀疏线性项个数（0=全维）
        'skewness': 0.3,  # 倾向分数 γ 的标准差（仅 dgp_num=0 使用）
        # 2/3
        'interaction': 0.0,  # D·X 交互强度
        'heterogeneous': 0.0,  # θ 的异质性强度
        'noise_std': 1.0,  # 结果噪声ε的标准差

        'true_effect': 1.0,  # θ 的基线值，设定真实因果效应
        'random_seed': 60,  # 复现用种子（每次 run 会 +run 累加）
        # 可选索引（不设则默认 0）
        # 'interaction_idx': 0,
        # 'hetero_idx': 0,
        # v1 结构可选背景列范围（不设则走默认 [2,5)）
        # 'bg_start': 2,
        # 'bg_end': 5,
        # v2 结构可选参数（不设则走默认）
        'rho': 0.7,
        's1': 1.0,
        'a0': 1.0,
        'a1': 0.25,
        'b0': 1.0,
        'b1': 0.25,
    }

    named_configs = [
        ("0_基准", {}),# baseline
        # ("1_非线性", {'nonlinearity': 1.0}),  # 加强 g(X) 的非线性项
        # ("2_交互", {'interaction': 0.5}),  # 加入 D·X 的交互项（强度 0.5）
        # ("3_稀疏性", {'sparse_k': 5}),  # 仅前 5 个 β 非零
        # ("4_偏态", {'skewness': 2.0}),  # 更极端的倾向分布
        # ("5_异质性", {'heterogeneous': 0.3}),  # θ = θ0 + 0.3·X

        # 交互强度（D·X）
        ("inter_0.25", {'interaction': 0.25}),
        ("inter_0.5", {'interaction': 0.5}),
        ("inter_1.0", {'interaction': 1.0}),

        # 异质性 θ(X) = θ0 + h·X[:,idx]
        ("hete_0.15", {'heterogeneous': 0.15}),
        ("hete_0.3", {'heterogeneous': 0.3}),
        ("hete_0.5", {'heterogeneous': 0.5}),

        # 结果噪声 ε 标准差
        ("noise_0.5", {'noise_std': 0.5}),
        ("noise_1.0", {'noise_std': 1.0}),
        ("noise_2.0", {'noise_std': 2.0}),

        # 相关结构参数：Toeplitz 相关系数 rho（影响 X 相关性）
        ("rho_0.3", {'rho': 0.3}),
        ("rho_0.5", {'rho': 0.5}),
        ("rho_0.7", {'rho': 0.7}),  # 同基线

        # 连续处理强度：D = m0(X) + s1·v
        ("s1_0.5", {'s1': 0.5}),
        ("s1_1.0", {'s1': 1.0}),  # 同基线
        ("s1_2.0", {'s1': 2.0}),

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
# def run_experiments(configs, dgp_num: int = 0, N_RUNS = 5):
def run_experiments(configs, dgp_num: int = 0, n_samples: int = 800, d_dim: int = 10, n_runs: int = 50):

    all_summary = []
    all_estimates = []  # 用于绘制QQ图

    for config in configs:
        estimates = []
        std_errors = []
        for run in range(n_runs):
            try:
                config_run = copy.deepcopy(config)
                config_run['random_seed'] = config['random_seed'] + run
                result = run_single_setting(config_run, dgp_num=dgp_num, n_samples=n_samples, d_dim=d_dim)
                estimates.append(result['theta_hat'])
                std_errors.append(result['se'])
            except Exception as e:
                print(f"配置 {config['config_name']} 第 {run} 次运行失败：{e}")
                continue

        metrics = evaluate_dml_results(estimates, std_errors, true_theta=config['true_effect'])

        # ---- summary：保留config全量参数 + 全局维度 + 指标 ----
        summary = {k: v for k, v in config.items() if k != 'random_seed'}
        summary.update({
            'n_samples': n_samples,
            'd_dim': d_dim,
            'dgp_num': dgp_num,
            'n_runs': n_runs,
        })
        summary.update(metrics)
        all_summary.append(summary)

        # all_estimates：逐run保存全量参数
        for run, (est, se) in enumerate(zip(estimates, std_errors)):
            row = {
                'config_name': config['config_name'],
                'theta_hat': est,
                'se': se,
                'true_effect': config['true_effect'],
                'z': (est - config['true_effect']) / se if se not in (0, None) else float('nan'),
                'run_id': run,
                'seed_used': config['random_seed'] + run,
                # 全局维度
                'n_samples': n_samples,
                'd_dim': d_dim,
                'dgp_num': dgp_num,
                'n_runs': n_runs,
            }
            # 合并此配置的所有参数（含默认/更新后的完整DGP参数）
            for k, v in config.items():
                if k != 'random_seed' and k not in row:  # 避免覆盖已有键
                    row[k] = v
            all_estimates.append(row)

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

    # 指定输出列
    metric_cols = ['bias', 'rmse', 'variance', 'coverage_rate', 'rejection_rate', 'mean_estimate']
    front = ['config_name']
    param_cols = [c for c in df.columns if c not in front + metric_cols]

    preferred = [
        'config_name',
        'n_samples', 'd_dim', 'dgp_num', 'n_runs',
        'nonlinearity', 'interaction', 'sparse_k', 'skewness',
        'heterogeneous', 'true_effect', 'noise_std',
        'rho', 's1', 'a0', 'a1', 'b0', 'b1',
        'interaction_idx', 'hetero_idx', 'bg_start', 'bg_end',
        'bias', 'rmse', 'variance', 'coverage_rate', 'rejection_rate', 'mean_estimate'
    ]
    df = df[[c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]]
    return df, all_estimates


# 全局默认参数设置
N_SAMPLES = 800   # 生成数据的样本量（X 的行数）
D_DIM = 10        # 特征维度（X 的列数）
N_RUNS = 50
DGP_NUM = 2

# 批量实验的默认参数网格
DEFAULT_GRID = {
    'N_SAMPLES': [50, 100, 200, 400, 800],
    'D_DIM': [10, 20, 30], # 0,1结构需要>=3
    'DGP_NUM': [2, 3],
    'N_RUNS': [100]
}
# 计算实验编号
existing = [int(re.findall(r'exp_(\d+)', d)[0]) for d in os.listdir('.') if re.match(r'exp_\d+', d)]
EXP_ID = max(existing) + 1 if existing else 1

# 工具函数
# 单次实验运行
def run_one_experiment(n_samples: int, d_dim: int, dgp_num: int, n_runs: int,
                       save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    # 载入配置
    configs = get_experiment_configs()
    # 运行实验
    df, all_estimates = run_experiments(
        configs, dgp_num=dgp_num,
        n_samples=n_samples, d_dim=d_dim, n_runs=n_runs
    )
    df.to_csv(os.path.join(save_dir, "dml_experiment_summary.csv"), index=False)

    # ================= 保存中间结果与配置 =================
    est_df = pd.DataFrame(all_estimates)
    est_df['z'] = (est_df['theta_hat'] - est_df['true_effect']) / est_df['se']
    est_df.to_csv(os.path.join(save_dir, "dml_all_estimates.csv"), index=False)

    cfg_df = pd.DataFrame(configs).copy()
    cfg_df['n_samples'] = n_samples
    cfg_df['d_dim']     = d_dim
    cfg_df['dgp_num']   = dgp_num
    cfg_df['n_runs']    = n_runs
    cfg_df.to_csv(os.path.join(save_dir, "dml_configs.csv"), index=False)
    # =======================================================================

    # 可视化（完全复用你现有的四个图）
    plot_raw_indicators(df, save_dir)
    plot_relative_differences(df, save_dir)
    # plot_qq_distribution(all_estimates, save_dir)
    # plot_hist_distribution(all_estimates, save_dir)
    _ae_legacy = [(r['config_name'], r['theta_hat'], r['se'], r['true_effect']) for r in all_estimates]
    plot_qq_distribution(_ae_legacy, save_dir)
    plot_hist_distribution(_ae_legacy, save_dir)

    print(f"结果与图像已保存至：{save_dir}")
    return df

# ------------------------ 主函数：运行实验，保存结果并打印输出 ------------------------
# -- main function: Run, save the results and print the output--
def main_single():
    root_dir  = f"exp_{EXP_ID}"
    run_tag = f"n{N_SAMPLES}_d{D_DIM}_g{DGP_NUM}_r{N_RUNS}"
    save_dir = os.path.join(root_dir , run_tag)

    run_one_experiment(
        n_samples=N_SAMPLES, d_dim=D_DIM, dgp_num=DGP_NUM, n_runs=N_RUNS,
        save_dir=save_dir
    )

def main_grid():
    root_dir = f"exp_{EXP_ID}"
    os.makedirs(root_dir, exist_ok=True)

    keys = ['N_SAMPLES', 'D_DIM', 'DGP_NUM', 'N_RUNS']
    values = [DEFAULT_GRID.get(k, [globals()[k]]) for k in keys]
    combos = list(product(*values))

    for n_samples, d_dim, dgp_num, n_runs in combos:
        if dgp_num not in (0, 1, 2, 3):
            print(f"[跳过] dgp_num={dgp_num} 暂未实现；当前仅支持 0/1/2/3。")
            continue
        subdir = os.path.join(root_dir, f"n{n_samples}_d{d_dim}_g{dgp_num}_r{n_runs}")
        print(f"\n运行组合: N={n_samples}, d={d_dim}, dgp={dgp_num}, runs={n_runs}")
        run_one_experiment(
            n_samples=n_samples, d_dim=d_dim, dgp_num=dgp_num, n_runs=n_runs,
            save_dir=subdir
        )

    print(f"\n全部批量实验完成。根目录：{root_dir}")

# ------------------------ 调用主函数 ------------------------
# 忽略警告信息，保持输出整洁
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    main_grid()
