import os
import re
import pandas as pd
import numpy as np
from scipy import stats



import matplotlib.pyplot as plt
from VISUAL_4 import plot_normality_reject_rate_curve

# 从路径解析 n/d/g/r
RX_BLOCK = re.compile(r"n(?P<n>\d+)_d(?P<d>\d+)_g(?P<g>\d+)_r(?P<r>\d+)")
RX_CFGIDX = re.compile(r"^(?P<idx>\d+)_")  # "0_基准" -- 0

# DML 参数列 & 性能指标列
DML_PARAM_COLS = [
    "n_samples", "d_dim", "dgp_num", "n_runs",
    "nonlinearity", "interaction", "sparse_k", "skewness",
    "heterogeneous", "true_effect", "noise_std"
]

METRIC_COLS = [
    "bias", "rmse", "variance", "coverage_rate", "rejection_rate", "mean_estimate"
]

PARAM_ORDER = DML_PARAM_COLS.copy()

def collect_and_merge_results(exp_folder: str):
    exp_root = os.path.abspath(exp_folder)

    # 1) 收集所有 dml_experiment_summary.csv
    summary_files = []
    for root, _, files in os.walk(exp_root):
        for f in files:
            if f == "dml_experiment_summary.csv":
                summary_files.append(os.path.join(root, f))

    if not summary_files:
        print(f"在 {exp_folder} 中未找到任何 dml_experiment_summary.csv 文件。")
        return

    # 2) 读取并拼接为一张总表（同时补充 n/d/g/r 与来源路径）
    rows = []
    for path in summary_files:
        parent = os.path.basename(os.path.dirname(path))  # n100_d10_g2_r50
        m = RX_BLOCK.match(parent)
        if not m:
            print(f"[跳过] 未匹配到 n/d/g/r：{path}")
            continue
        n_samples = int(m.group("n"))
        d_dim     = int(m.group("d"))
        dgp_num   = int(m.group("g"))
        n_runs    = int(m.group("r"))

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[跳过] 读取失败：{path} -> {e}")
            continue

        # 增加元信息列
        df["n_samples"]  = n_samples
        df["d_dim"]      = d_dim
        df["dgp_num"]    = dgp_num
        df["n_runs"]     = n_runs
        df["source_file"] = path  # 后续写出前会删除

        rows.append(df)

    if not rows:
        print("未获得可用数据。")
        return

    big = pd.concat(rows, ignore_index=True)

    # 3) 提取“参数配置编号 config_index”以便分组输出
    if "config_name" not in big.columns:
        big["config_name"] = "unknown"

    # 4) 输出目录：<指定实验文件夹>/DataAnalysis
    out_dir = os.path.join(exp_root, "DataAnalysis")
    os.makedirs(out_dir, exist_ok=True)

    # 4.1 先写出“总表” all_results.csv（不包含 source_file）
    preferred_all = [
        "config_index", "config_name",
        "n_samples", "d_dim", "dgp_num", "n_runs",
        "nonlinearity", "interaction", "sparse_k", "skewness",
        "heterogeneous", "true_effect", "true_theta", "noise_std",
        "bias", "rmse", "variance", "coverage_rate", "rejection_rate", "mean_estimate",
    ]
    big_out = big.drop(columns=["source_file"], errors="ignore").copy()
    cols_all = [c for c in preferred_all if c in big_out.columns] + \
               [c for c in big_out.columns if c not in preferred_all]
    big_out = big_out[cols_all]
    all_path = os.path.join(out_dir, "all_results.csv")
    big_out.to_csv(all_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 输出总表：{all_path} （{len(big_out)} 行，"
          f"{big_out[['n_samples','d_dim','dgp_num','n_runs']].drop_duplicates().shape[0]} 个 DML 组合）")

    # 5) 按“参数配置”拆分写出：X 个文件（同样不包含 source_file）
    all_cfg_names = sorted(big["config_name"].dropna().unique().tolist())
    for name in all_cfg_names:
        sub = big[big["config_name"] == name].drop(columns=["source_file"], errors="ignore").copy()

        # 列顺序（保持你原先的 preferred_cols）
        preferred_cols = [
            "config_name",
            "n_samples", "d_dim", "dgp_num", "n_runs",
            "nonlinearity", "interaction", "sparse_k", "skewness",
            "heterogeneous", "true_effect", "noise_std",
            "bias", "rmse", "variance", "coverage_rate", "rejection_rate", "mean_estimate",
        ]
        cols = [c for c in preferred_cols if c in sub.columns] + \
               [c for c in sub.columns if c not in preferred_cols]
        sub = sub[cols]

        safe = re.sub(r"[^\w\-_.\u4e00-\u9fa5]", "_", str(name))
        out_path = os.path.join(out_dir, f"config_{safe}.csv")
        sub.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"输出：{out_path} (共 {len(sub)} 行，覆盖 "
              f"{sub[['n_samples', 'd_dim', 'dgp_num', 'n_runs']].drop_duplicates().shape[0]} 个 DML 组合)")


def get_columns_from_agg(agg_csv_path: str):
    df = pd.read_csv(agg_csv_path)
    return [c for c in df.columns if c not in ("config_index", "source_file")]

def load_all_estimates(exp_folder: str) -> pd.DataFrame:
    # 汇总各实验子目录中的 dml_all_estimates.csv 为一张表，便于正态性检验。
    # 关键列：config_name, theta_hat, se, true_theta, z, n_samples, d_dim, dgp_num, n_runs, 以及各参数列
    exp_root = os.path.abspath(exp_folder)
    est_files = []
    for root, _, files in os.walk(exp_root):
        for f in files:
            if f == "dml_all_estimates.csv":
                est_files.append(os.path.join(root, f))
    if not est_files:
        print(f"[load_all_estimates] 在 {exp_folder} 未找到 dml_all_estimates.csv")
        return pd.DataFrame()

    rows = []
    for p in est_files:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[load_all_estimates] 读取失败：{p} -> {e}")
            continue

        # 若缺 z，补算：z = (theta_hat - true_theta)/se ；优先逐 run 的 true_theta
        need_z = 'z' not in df.columns and {'theta_hat', 'se'}.issubset(df.columns)
        if need_z:
            true_col = 'true_theta' if 'true_theta' in df.columns else 'true_effect'
            if true_col in df.columns:
                df['z'] = (df['theta_hat'] - df[true_col]) / df['se']
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    big = pd.concat(rows, ignore_index=True)

    # 统一必要列存在性
    for col in ['config_name', 'n_samples', 'd_dim', 'dgp_num', 'n_runs']:
        if col not in big.columns:
            print(f"[load_all_estimates] 警告：缺少列 {col}，后续分组可能受影响。")

    return big

def compute_normality_tests(df_est: pd.DataFrame,
                            group_keys: list[str]) -> pd.DataFrame:
    # 对逐 run 的 z 值做 Shapiro–Wilk 和 Anderson–Darling(正态)检验。
    # 返回：每个 group_keys 组合一行，含统计量与是否拒绝的布尔标记。

    if df_est is None or df_est.empty:
        return pd.DataFrame()

    if 'z' not in df_est.columns:
        raise ValueError("compute_normality_tests 需要列 'z'（标准化统计量）。")

    # 只保留有限 z
    df = df_est[np.isfinite(df_est['z'])].copy()
    if df.empty:
        return pd.DataFrame()

    out_rows = []
    for keys, g in df.groupby(group_keys, dropna=False, sort=False):
        zs = g['z'].dropna().to_numpy(dtype=float)
        n = len(zs)
        if n < 3:
            # Shapiro 需要 n>=3；太小则跳过但保留一行占位
            row = {k: v for k, v in zip(group_keys, keys if isinstance(keys, tuple) else (keys,))}
            row.update(dict(n=n, W=np.nan, p_shapiro=np.nan, A2=np.nan,
                            reject_shapiro=False, reject_ad=False, reject_both=False))
            out_rows.append(row)
            continue

        # Shapiro–Wilk
        try:
            W, p_shapiro = stats.shapiro(zs)
            reject_shapiro = bool(p_shapiro < 0.05)
        except Exception:
            W, p_shapiro, reject_shapiro = np.nan, np.nan, False

        # Anderson–Darling for normal
        try:
            ad = stats.anderson(zs, dist='norm')  # ad.statistic, ad.critical_values, ad.significance_level
            # 以 5% 临界值判定
            # 找到最接近 5% 的阈值
            sig_levels = np.asarray(ad.significance_level)
            idx = int(np.argmin(np.abs(sig_levels - 5.0)))
            crit_5 = float(ad.critical_values[idx])
            A2 = float(ad.statistic)
            reject_ad = bool(A2 > crit_5)
        except Exception:
            A2, reject_ad = np.nan, False

        row = {k: v for k, v in zip(group_keys, keys if isinstance(keys, tuple) else (keys,))}
        row.update(dict(n=n, W=W, p_shapiro=p_shapiro, A2=A2,
                        reject_shapiro=reject_shapiro,
                        reject_ad=reject_ad,
                        reject_both=(reject_shapiro and reject_ad)))
        out_rows.append(row)

    return pd.DataFrame(out_rows)

def aggregate_normality_reject_rate(test_df: pd.DataFrame,
                                    agg_keys: list[str]) -> pd.DataFrame:
    # 将 group-level 的正态性检验结果聚合为“拒绝率曲线”：
    # 对每个 agg_keys + n_samples 组合，计算 reject_shapiro/ad/both 的均值。

    if test_df is None or test_df.empty:
        return pd.DataFrame()

    need_cols = set(agg_keys + ['n_samples', 'reject_shapiro', 'reject_ad', 'reject_both'])
    missing = [c for c in need_cols if c not in test_df.columns]
    if missing:
        raise ValueError(f"aggregate_normality_reject_rate 缺少列：{missing}")

    # 聚合分组键，去重（防止 agg_keys 已含 n_samples 导致重复分组键触发 pandas 插列冲突）
    gcols_raw = agg_keys + ['n_samples']
    seen = set()
    gcols = [c for c in gcols_raw if not (c in seen or seen.add(c))]

    out = (
        test_df
        .groupby(gcols, dropna=False, sort=False, as_index=False)
        .agg(
            reject_rate_shapiro=('reject_shapiro', 'mean'),
            reject_rate_ad=('reject_ad', 'mean'),
            reject_rate_both=('reject_both', 'mean'),
        )
    )
    return out


# def compute_stability_threshold(all_results: pd.DataFrame,
#                                 normality_rates: pd.DataFrame,
#                                 by_keys: list[str],
#                                 coverage_thr: float = 0.90,
#                                 normality_thr: float = 0.20) -> pd.DataFrame:
#     # 条件：
#     #   A) coverage_rate >= coverage_thr
#     #   B) reject_rate_both <= normality_thr
#     # 在 by_keys（不含 n_samples）定义的每个“配置组”内，沿 n_samples 升序寻找第一个同时满足 A&B 的 n，记为 N_star。
#     # 输出：每组一行，含 N_star 及该点处的关键指标。
#
#     if all_results is None or all_results.empty:
#         return pd.DataFrame()
#     if normality_rates is None or normality_rates.empty:
#         return pd.DataFrame()
#
#     # 统一用于合并的键
#     merge_keys = list(set(by_keys + ['n_samples']))
#     merged = pd.merge(all_results, normality_rates, how='inner', on=merge_keys)
#
#     for c in merge_keys:
#         if c not in all_results.columns:
#             print(f"[compute_stability_threshold] all_results 缺少列 {c}")
#         if c not in normality_rates.columns:
#             print(f"[compute_stability_threshold] normality_rates 缺少列 {c}")
#
#     # 对每个 by_keys 组合，按 n_samples 升序查找 N*
#     records = []
#     for keys, g in merged.groupby(by_keys, dropna=False, sort=False):
#         g2 = g.sort_values('n_samples', kind='mergesort')
#
#         met = g2[(g2['coverage_rate'] >= coverage_thr) &
#                  (g2['reject_rate_both'] <= normality_thr)]
#
#         row = {k: v for k, v in zip(by_keys, keys if isinstance(keys, tuple) else (keys,))}
#         if met.empty:
#             # 诊断：最接近阈值的情况
#             cov_max = float(g2['coverage_rate'].max()) if 'coverage_rate' in g2 else np.nan
#             rej_min = float(g2['reject_rate_both'].min()) if 'reject_rate_both' in g2 else np.nan
#             n_at_cov_max = int(g2.loc[g2['coverage_rate'].idxmax(), 'n_samples']) if 'coverage_rate' in g2 and g2['coverage_rate'].notna().any() else np.nan
#             n_at_rej_min = int(g2.loc[g2['reject_rate_both'].idxmin(), 'n_samples']) if 'reject_rate_both' in g2 and g2['reject_rate_both'].notna().any() else np.nan
#
#             row.update(dict(
#                 N_star=np.nan,
#                 bias_at_N=np.nan, rmse_at_N=np.nan, var_at_N=np.nan,
#                 cov_at_N=np.nan, rej_at_N=np.nan, rej_norm_both_at_N=np.nan,
#                 status="no_stable_region",
#                 cov_max=cov_max, n_at_cov_max=n_at_cov_max,
#                 rej_min=rej_min, n_at_rej_min=n_at_rej_min
#             ))
#         else:
#             first = met.iloc[0]
#             row.update(dict(
#                 N_star=int(first['n_samples']),
#                 bias_at_N=float(first.get('bias', np.nan)),
#                 rmse_at_N=float(first.get('rmse', np.nan)),
#                 var_at_N=float(first.get('variance', np.nan)),
#                 cov_at_N=float(first.get('coverage_rate', np.nan)),
#                 rej_at_N=float(first.get('rejection_rate', np.nan)),
#                 rej_norm_both_at_N=float(first.get('reject_rate_both', np.nan)),
#                 status="ok",
#                 cov_max=np.nan, n_at_cov_max=np.nan,
#                 rej_min=np.nan, n_at_rej_min=np.nan
#             ))
#         records.append(row)
#
#     return pd.DataFrame(records)

def compute_stability_threshold(all_results: pd.DataFrame,
                                normality_rates: pd.DataFrame,
                                by_keys: list[str],
                                coverage_thr: float = 0.90,
                                normality_thr: float = 0.20) -> pd.DataFrame:
    if all_results is None or all_results.empty:
        return pd.DataFrame()
    if normality_rates is None or normality_rates.empty:
        return pd.DataFrame()

    # 统一用于合并的键
    merge_keys = list(set(by_keys + ['n_samples']))
    merged = pd.merge(all_results, normality_rates, how='inner', on=merge_keys)

    # 对每个 by_keys 组合，按 n_samples 升序查找 N*
    records = []
    for keys, g in merged.groupby(by_keys, dropna=False, sort=False):
        g2 = g.sort_values('n_samples', kind='mergesort')
        met = g2[(g2['coverage_rate'] >= coverage_thr) &
                 (g2['reject_rate_both'] <= normality_thr)]

        row = {k: v for k, v in zip(by_keys, keys if isinstance(keys, tuple) else (keys,))}
        if met.empty:# 诊断：最接近阈值的情况
            cov_max = float(g2['coverage_rate'].max()) if 'coverage_rate' in g2 else np.nan
            rej_min = float(g2['reject_rate_both'].min()) if 'reject_rate_both' in g2 else np.nan
            n_at_cov_max = int(g2.loc[g2['coverage_rate'].idxmax(), 'n_samples']) if 'coverage_rate' in g2 and g2['coverage_rate'].notna().any() else np.nan
            n_at_rej_min = int(g2.loc[g2['reject_rate_both'].idxmin(), 'n_samples']) if 'reject_rate_both' in g2 and g2['reject_rate_both'].notna().any() else np.nan

            row.update(dict(
                N_star=np.nan,
                bias_at_N=np.nan, rmse_at_N=np.nan, var_at_N=np.nan,
                cov_at_N=np.nan, rej_at_N=np.nan, rej_norm_both_at_N=np.nan,
                status="no_stable_region",
                cov_max=cov_max, n_at_cov_max=n_at_cov_max,
                rej_min=rej_min, n_at_rej_min=n_at_rej_min
            ))
        else:
            first = met.iloc[0]
            row.update(dict(
                N_star=int(first['n_samples']),
                bias_at_N=float(first.get('bias', np.nan)),
                rmse_at_N=float(first.get('rmse', np.nan)),
                var_at_N=float(first.get('variance', np.nan)),
                cov_at_N=float(first.get('coverage_rate', np.nan)),
                rej_at_N=float(first.get('rejection_rate', np.nan)),
                rej_norm_both_at_N=float(first.get('reject_rate_both', np.nan)),
                status="ok",
                cov_max=np.nan, n_at_cov_max=np.nan,
                rej_min=np.nan, n_at_rej_min=np.nan
            ))
        records.append(row)

    return pd.DataFrame(records)


def save_tables(out_dir: str,
                normality_tests: pd.DataFrame,
                normality_rates: pd.DataFrame,
                thresholds: pd.DataFrame):
    os.makedirs(out_dir, exist_ok=True)
    if normality_tests is not None and not normality_tests.empty:
        normality_tests.to_csv(os.path.join(out_dir, "normality_by_group.csv"),
                               index=False, encoding="utf-8-sig")
    if normality_rates is not None and not normality_rates.empty:
        normality_rates.to_csv(os.path.join(out_dir, "normality_reject_rates.csv"),
                               index=False, encoding="utf-8-sig")
    if thresholds is not None and not thresholds.empty:
        thresholds.to_csv(os.path.join(out_dir, "stability_threshold.csv"),
                          index=False, encoding="utf-8-sig")
    print(f"[OK] 表格已写入：{out_dir}")


def main_normality_and_threshold(exp_root: str,
                                 group_keys=None,
                                 agg_keys=None,
                                 by_keys=None,
                                 coverage_thr: float = 0.90,
                                 normality_thr: float = 0.20):

    # 汇总：正态性检验-拒绝率曲线数据-临界点-表格落盘-调用 _4 画曲线

    if group_keys is None:
        group_keys = ['config_name', 'n_samples', 'd_dim', 'dgp_num']
    if agg_keys is None:
        agg_keys = ['config_name', 'n_samples', 'd_dim', 'dgp_num']
    if by_keys is None:
        by_keys = ['config_name', 'd_dim', 'dgp_num']

    exp_root = os.path.abspath(exp_root)
    out_dir = os.path.join(exp_root, "DataAnalysis")
    os.makedirs(out_dir, exist_ok=True)

    # 1) 汇总 all_results.csv（若不存在）
    all_csv = os.path.join(out_dir, "all_results.csv")
    if not os.path.exists(all_csv):
        collect_and_merge_results(exp_root)
    if not os.path.exists(all_csv):
        print("[main_normality_and_threshold] 无 all_results.csv，结束。")
        return
    all_results = pd.read_csv(all_csv)

    # 2) 合并逐 run 估计并做正态性检验
    est_df = load_all_estimates(exp_root)
    if est_df is None or est_df.empty:
        print("[main_normality_and_threshold] 无逐 run 估计，结束。")
        return

    test_df = compute_normality_tests(est_df, group_keys=group_keys)
    rate_df = aggregate_normality_reject_rate(test_df, agg_keys=agg_keys)

    # 3) 计算稳定性临界点
    thr_df = compute_stability_threshold(all_results, rate_df,
                                         by_keys=by_keys,
                                         coverage_thr=coverage_thr,
                                         normality_thr=normality_thr)

    # 4) 落盘
    save_tables(out_dir, test_df, rate_df, thr_df)

    # 5) 画reject_rate_curve曲线（_4）
    try:
        # plot_normality_reject_rate_curve(rate_df, out_dir, facet_by=['config_name'])
        # 分组绘图：每个 (config_name, d_dim, dgp_num) 一张图，输出到专属子目录
        curves_root = os.path.join(out_dir, "normality_curves")
        for (cfg, d, g), sub in rate_df.groupby(['config_name', 'd_dim', 'dgp_num'], dropna=False, sort=False):
            subdir = os.path.join(curves_root, str(cfg), f"d{int(d)}_g{int(g)}")
            os.makedirs(subdir, exist_ok=True)
            # 单图：不再分面，避免混合
            plot_normality_reject_rate_curve(
                sub, subdir,
                facet_by=[],  # 关键：不分面，每组一张
                x='n_samples',
                ys=('reject_rate_shapiro', 'reject_rate_ad', 'reject_rate_both')
            )

    except Exception as e:
        print(f"[Warn] 绘制正态性拒绝率曲线失败：{e}")

    # 表格：每组的小表落到分目录，便于检查
    tables_root = os.path.join(out_dir, "normality_tables")
    os.makedirs(tables_root, exist_ok=True)

    # 强制写入列
    cols_ok = [
        'config_name', 'd_dim', 'dgp_num',
        'N_star', 'bias_at_N', 'rmse_at_N', 'var_at_N', 'cov_at_N', 'rej_at_N', 'rej_norm_both_at_N',
        'status', 'cov_max', 'n_at_cov_max', 'rej_min', 'n_at_rej_min'
    ]

    thr_map = thr_df.set_index(['config_name', 'd_dim', 'dgp_num'])

    # 3) 各组的 threshold.csv
    for (cfg, d, g), _ in rate_df.groupby(['config_name', 'd_dim', 'dgp_num'], dropna=False, sort=False):
        key = (cfg, d, g)
        subdir = os.path.join(tables_root, str(cfg), f"d{int(d)}_g{int(g)}")
        os.makedirs(subdir, exist_ok=True)

        if key not in thr_map.index:
            out = pd.DataFrame([{
                'config_name': cfg, 'd_dim': d, 'dgp_num': g, 'status': 'no_record'
            }], columns=cols_ok)
        else:
            row = thr_map.loc[key]
            out = row.to_frame().T.reset_index()
            out = out.rename(columns={'level_0': 'config_name', 'level_1': 'd_dim', 'level_2': 'dgp_num'})
            for c in cols_ok:
                if c not in out.columns:
                    out[c] = np.nan
            out = out[cols_ok]

        out.to_csv(os.path.join(subdir, "threshold.csv"),
                   index=False, encoding="utf-8-sig")

    # 1) 各组的 normality_reject_rates 子表
    for (cfg, d, g), sub in rate_df.groupby(['config_name', 'd_dim', 'dgp_num'], dropna=False, sort=False):
        subdir = os.path.join(tables_root, str(cfg), f"d{int(d)}_g{int(g)}")
        os.makedirs(subdir, exist_ok=True)
        sub.to_csv(os.path.join(subdir, "reject_rates.csv"), index=False, encoding="utf-8-sig")

    # 2) 各组的 normality_by_group 子表（更细，包含 W、A2 等）
    for (cfg, d, g), sub in test_df.groupby(['config_name', 'd_dim', 'dgp_num'], dropna=False, sort=False):
        subdir = os.path.join(tables_root, str(cfg), f"d{int(d)}_g{int(g)}")
        os.makedirs(subdir, exist_ok=True)
        sub.to_csv(os.path.join(subdir, "by_group.csv"), index=False, encoding="utf-8-sig")


def filter_and_plot(agg_csv_path: str, out_dir: str, factor: str | None = None, **filters):

    # 单因素扰动可视化：
    # - 先用 **filters** 固定住 DGP 结构参数与除 factor 外的 DML 参数；
    # - 指定 factor（例如 'interaction'），在 x 轴按 n_samples 展开，
    #   每个 factor 水平绘制一组（分系列）——便于观察单因素随样本量变化的性能曲线/柱状图。
    # - 若未提供 factor，则保持原先的自动分组（最多两个未指定参数作为标签）。


    df = pd.read_csv(agg_csv_path)
    os.makedirs(out_dir, exist_ok=True)

    # 0) 校验 factor 是否有效
    if factor is not None and factor not in df.columns:
        print(f"[警告] 指定的 factor='{factor}' 不在数据列中，将回退为原自动分组逻辑。")
        factor = None

    # 1) 过滤：用 filters 固定其他参数
    used_keys = []
    for key, val in filters.items():
        if key in df.columns and val is not None:
            if not isinstance(val, (list, tuple, set)):
                val = [val]
            df = df[df[key].isin(list(val))]
            used_keys.append(key)
    if not used_keys and factor is None:
        raise ValueError("必须至少指定一个 DML 参数用于筛选（例如 n_samples=[100,200], dgp_num=2）。")
    if df.empty:
        print("筛选后无数据")
        return

    # ====== 模式A：单因素扰动（推荐） ======
    if factor is not None:
        # 要求 factor 在局部数据中至少有2个水平，n_samples 也应至少1个
        if df[factor].nunique(dropna=False) <= 1:
            print(f"[提示] factor='{factor}' 在筛选后仅有一个水平，改用原自动分组逻辑。")
            factor = None
        elif 'n_samples' not in df.columns:
            print("[提示] 数据中不存在 n_samples 列，改用原自动分组逻辑。")
            factor = None

    if factor is not None:
        # 只保留绘图需要的列
        keep_cols = ['n_samples', factor] + [c for c in METRIC_COLS if c in df.columns]
        df = df[keep_cols].copy()

        # 聚合：相同 (n_samples, factor) 取均值（多次重复/多配置稳健处理）
        grp = df.groupby(['n_samples', factor], as_index=False).mean(numeric_only=True)

        # 生成 x 轴与系列
        x_vals = sorted(grp['n_samples'].unique())
        levels = sorted(grp[factor].unique(), key=lambda v: (isinstance(v, str), v))

        for metric in METRIC_COLS:
            if metric not in grp.columns:
                continue
            plt.figure(figsize=(9, 5))
            # 分系列柱状：每个 factor 水平一组
            width = 0.8 / max(1, len(levels))
            x_idx = np.arange(len(x_vals))

            for i, lev in enumerate(levels):
                gi = grp[grp[factor] == lev].set_index('n_samples').reindex(x_vals)
                y = gi[metric].to_numpy()
                plt.bar(x_idx + i*width, y, width=width, label=f"{factor}={lev}")
                # 数值标注
                for xi, yi in zip(x_idx + i*width, y):
                    if pd.notna(yi):
                        plt.text(xi, yi, f"{yi:.3f}", ha='center', va='bottom' if yi>=0 else 'top', fontsize=8)

            plt.xticks(x_idx + (len(levels)-1)*width/2, [str(x) for x in x_vals])
            plt.xlabel('n_samples')
            plt.ylabel(metric)
            plt.title(f"Single-factor: {factor}")
            plt.legend(frameon=False, fontsize=9)
            plt.tight_layout()

            out_file = os.path.join(out_dir, f"{metric}__by_n__factor_{factor}.png")
            plt.savefig(out_file)
            plt.close()
            print(f"[Plot] {out_file}")
        return

    # ====== 模式B：自动分组（最多取两个未指定参数） ======
    unspecified = [c for c in DML_PARAM_COLS if c in df.columns and c not in used_keys]
    unspecified = [c for c in unspecified if df[c].nunique(dropna=False) > 1]
    unspecified = sorted(unspecified, key=lambda c: PARAM_ORDER.index(c))[:2]

    if len(unspecified) == 0:
        df["__label__"] = "all"
        grouped = df.groupby("__label__", as_index=False).mean(numeric_only=True)
        title_suffix = "all specified"
        label_cols = []
    elif len(unspecified) == 1:
        col = unspecified[0]
        df["__label__"] = df[col].astype(str)
        grouped = df.groupby("__label__", as_index=False).mean(numeric_only=True)
        title_suffix = col
        label_cols = [col]
    else:
        col1, col2 = unspecified[0], unspecified[1]
        df["__label__"] = df[[col1, col2]].astype(str).agg(",".join, axis=1)
        grouped = df.groupby("__label__", as_index=False).mean(numeric_only=True)
        title_suffix = f"{col1} + {col2}"
        label_cols = [col1, col2]

    for metric in METRIC_COLS:
        if metric not in grouped.columns:
            continue
        x_labels = grouped["__label__"].tolist()
        y_vals = grouped[metric].values
        plt.figure(figsize=(8, 5))
        bars = plt.bar(x_labels, y_vals)
        for b, v in zip(bars, y_vals):
            plt.text(b.get_x() + b.get_width() / 2, v,
                     f"{v:.3f}", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=9)
        plt.title(title_suffix)
        plt.xlabel(" , ".join(label_cols) if label_cols else "")
        plt.ylabel(metric)
        plt.xticks(rotation=0)
        if unspecified:
            suffix = "+".join(unspecified)
            out_file = os.path.join(out_dir, f"{metric}__{suffix}.png")
        else:
            out_file = os.path.join(out_dir, f"{metric}.png")
        plt.tight_layout()
        plt.savefig(out_file)
        plt.close()
        print(f"[Plot] {out_file}")


def facet_single_factor(agg_csv_path: str,
                        out_dir: str,
                        factor: str,
                        x_axis: str = 'n_samples',   # or 'd_dim'
                        facet_dim: str | None = None,
                        annotate: bool = True,
                        **filters):

    df = pd.read_csv(agg_csv_path)
    os.makedirs(out_dir, exist_ok=True)

    if factor not in df.columns:
        raise ValueError(f"[facet_single_factor] 指定的 factor='{factor}' 不在列中。")

    tri = {'n_samples', 'd_dim', 'dgp_num'}
    if x_axis not in tri:
        raise ValueError(f"[facet_single_factor] x_axis 必须在 {tri} 中。")

    # 过滤
    used_keys = []
    for key, val in filters.items():
        if key in df.columns and val is not None:
            if not isinstance(val, (list, tuple, set)):
                val = [val]
            df = df[df[key].isin(list(val))]
            used_keys.append(key)

    if df.empty:
        print("[facet_single_factor] 筛选后无数据。")
        return

    remaining = list(tri - {x_axis})
    if facet_dim is None:
        facet_dim = 'd_dim' if x_axis == 'n_samples' else 'n_samples'
    if facet_dim not in remaining:
        raise ValueError(f"[facet_single_factor] facet_dim 应在 {remaining} 中，收到 {facet_dim}。")

    # 块分组维度（优先 dgp_num）
    block_dim = [c for c in remaining if c != facet_dim]
    block_dim = block_dim[0] if block_dim else ('dgp_num' if 'dgp_num' in df.columns else remaining[0])

    keep = list({x_axis, facet_dim, block_dim, 'config_name', factor, *METRIC_COLS})
    keep = [c for c in keep if c in df.columns]
    work = df[keep].copy()

    grp = (
        work.groupby([facet_dim, x_axis, block_dim, 'config_name', factor],
                     dropna=False, as_index=False)
            .mean(numeric_only=True)
    )



    facets = sorted(grp[facet_dim].dropna().unique().tolist(), key=lambda v: (isinstance(v, str), v))
    x_vals = sorted(grp[x_axis].dropna().unique().tolist(), key=lambda v: (isinstance(v, str), v))
    blocks = sorted(grp[block_dim].dropna().unique().tolist(), key=lambda v: (isinstance(v, str), v))
    if set([2, 3]).issubset(set(blocks)):
        blocks = [2, 3] + [b for b in blocks if b not in (2, 3)]

    non_base = grp[grp['config_name'] != '0_base'][factor].dropna().unique().tolist()
    try:
        factor_levels = sorted(non_base, key=lambda v: float(v))
    except Exception:
        factor_levels = sorted(non_base, key=lambda v: (isinstance(v, str), v))

    L = 1 + len(factor_levels)  # 每个 block 的柱数

    # ===== 自适应图形尺寸与文字密度控制 =====
    n_facets = max(1, len(facets))
    n_cols = min(1, n_facets)
    n_rows = int(np.ceil(n_facets / n_cols))
    bars_per_x = max(1, len(blocks)) * L
    bars_total_per_facet = max(1, len(x_vals)) * bars_per_x

    # 动态标注开关
    if bars_total_per_facet >= 120:
        annotate = False

    # 每个子图宽度 + 总画布高度
    per_facet_w = max(10.0, min(0.22 * bars_total_per_facet, 18.0))
    fig_w = per_facet_w
    fig_h = max(3.7 * n_rows, 4.2)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    # 颜色
    cmap = plt.cm.get_cmap('tab10')
    color_base = '#8c8c8c'
    color_map = {('baseline', None): color_base}
    for i, lev in enumerate(factor_levels):
        color_map[('factor', lev)] = cmap(i % 10)

    # 字号
    title_fs = 11
    axis_fs = 9
    tick_fs = 8
    legend_fs = 9
    annot_fs = 7

    for metric in METRIC_COLS:
        if metric not in grp.columns:
            continue

        # 清空画布
        for ar in axes:
            for ax in ar:
                ax.clear()
        # fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

        # ===== 每个子图 =====
        for idx_f, fval in enumerate(facets if len(facets) > 0 else [None]):
            ax = axes[idx_f][0]
            sub = grp if fval is None else grp[grp[facet_dim] == fval]

            base_x = np.arange(len(x_vals))
            total_per_x = max(1, len(blocks)) * L
            width = 0.8 / total_per_x

            xs, ys, cols, labels = [], [], [], []

            # (x -> block -> [0_base + levels])
            for xi, xv in enumerate(x_vals):
                for bi, b in enumerate(blocks):
                    # baseline
                    g0 = sub[(sub[x_axis] == xv) & (sub[block_dim] == b) & (sub['config_name'] == '0_base')]
                    y0 = float(g0[metric].mean()) if not g0.empty else np.nan
                    xs.append(base_x[xi] - 0.4 + bi * L * width + 0 * width)
                    ys.append(y0); cols.append(color_map[('baseline', None)]); labels.append('0_base')

                    # factor levels
                    for j, lev in enumerate(factor_levels, start=1):
                        gi = sub[(sub[x_axis] == xv) & (sub[block_dim] == b) &
                                 (sub['config_name'] != '0_base') & (sub[factor] == lev)]
                        yi = float(gi[metric].mean()) if not gi.empty else np.nan
                        xs.append(base_x[xi] - 0.4 + bi * L * width + j * width)
                        ys.append(yi); cols.append(color_map[('factor', lev)]); labels.append(f"{factor}={lev}")

            bars = ax.bar(xs, ys, width=width, color=cols, edgecolor='white', linewidth=0.6)

            # 数值标注
            if annotate:
                for rect, val in zip(bars, ys):
                    if pd.notna(val):
                        ax.text(rect.get_x() + rect.get_width()/2, val,
                                f"{val:.3f}", ha='center',
                                va='bottom' if val >= 0 else 'top',
                                fontsize=annot_fs)

            # x 轴刻度（稀疏）
            ax.set_xticks(base_x)
            xticklabels = [str(x) for x in x_vals]
            sparsity = 1
            if len(x_vals) * max(1, len(blocks)) >= 12: sparsity = 2
            if len(x_vals) * max(1, len(blocks)) >= 24: sparsity = 3
            for i in range(len(xticklabels)):
                if i % sparsity != 0:
                    xticklabels[i] = ""
            ax.set_xticklabels(xticklabels, rotation=30, ha='right', fontsize=tick_fs)
            ax.tick_params(axis='y', labelsize=tick_fs)

            ax.set_xlabel(x_axis, fontsize=axis_fs)
            ax.set_ylabel(metric, fontsize=axis_fs)
            # ——把标题放高一点（y=1.08），并把“DGP2 | DGP3”放在标题下方（y=1.02），避免重叠
            ax.set_title(f"{facet_dim}={fval}", fontsize=title_fs, y=1.08)
            ax.text(0.5, 1.02, "DGP2    |    DGP3",
                    transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=legend_fs, alpha=0.8)
            ax.grid(alpha=0.2)

            # 分割线：两块之间的竖线
            if len(blocks) >= 2:
                for xi in range(len(x_vals)):
                    boundary_x = base_x[xi] - 0.4 + L * width  # 第一块结束处
                    ax.axvline(boundary_x, color='k', linestyle=':', linewidth=0.8, alpha=0.25)

        # ===== 顶部统一图例（总标题下方） =====
        handles, labels_legend = [], []
        handles.append(plt.Rectangle((0, 0), 1, 1, color=color_base))
        labels_legend.append('0_base')
        for lev in factor_levels:
            handles.append(plt.Rectangle((0, 0), 1, 1, color=color_map[('factor', lev)]))
            labels_legend.append(f"{factor}={lev}")

        # 总标题+图例
        fig.suptitle(f"{metric}  |  x={x_axis}  |  facet={facet_dim}  |  blocks by {block_dim}",
                     y=0.98, fontsize=12)

        # 图例放在总标题下方一点，且在图内
        lg = fig.legend(handles, labels_legend,
                        loc='upper center', bbox_to_anchor=(0.5, 0.955),
                        ncol=min(6, len(labels_legend)),
                        fontsize=legend_fs, frameon=False)

        # 给 suptitle+legend 腾空间，减小子图区域
        fig.subplots_adjust(top=0.90)          # 顶部留白
        fig.tight_layout(rect=[0, 0, 1, 0.90])  # 子图布局在 0.90 以下

        out_file = os.path.join(out_dir, f"{metric}__{facet_dim}__{x_axis}__{factor}.png")
        fig.savefig(out_file, dpi=220)
        plt.close(fig)
        print(f"[Plot] {out_file}")

def facet_by_configs(agg_csv_path: str,
                     out_dir: str,
                     configs: list[str],          # 要对比的一组 config_name（不含 '0_base'）
                     x_axis: str = 'n_samples',   # or 'd_dim'
                     facet_dim: str | None = None,
                     annotate: bool = True,
                     **filters):

    df = pd.read_csv(agg_csv_path)
    os.makedirs(out_dir, exist_ok=True)

    if 'config_name' not in df.columns:
        raise ValueError("[facet_by_configs] 缺少列 'config_name'。")

    tri = {'n_samples', 'd_dim', 'dgp_num'}
    if x_axis not in tri:
        raise ValueError(f"[facet_by_configs] x_axis 必须在 {tri} 中。")

    # 过滤
    used_keys = []
    for key, val in filters.items():
        if key in df.columns and val is not None:
            if not isinstance(val, (list, tuple, set)):
                val = [val]
            df = df[df[key].isin(list(val))]
            used_keys.append(key)
    if df.empty:
        print("[facet_by_configs] 筛选后无数据。")
        return

    remaining = list(tri - {x_axis})
    if facet_dim is None:
        facet_dim = 'd_dim' if x_axis == 'n_samples' else 'n_samples'
    if facet_dim not in remaining:
        raise ValueError(f"[facet_by_configs] facet_dim 应在 {remaining} 中，收到 {facet_dim}。")

    # 块分组维度（优先 dgp_num）
    block_dim = [c for c in remaining if c != facet_dim]
    block_dim = block_dim[0] if block_dim else ('dgp_num' if 'dgp_num' in df.columns else remaining[0])

    # 只保留需要的列
    keep = list({x_axis, facet_dim, block_dim, 'config_name', *METRIC_COLS})
    keep = [c for c in keep if c in df.columns]
    work = df[keep].copy()

    # ---- 强制把六个指标转为数值，避免 groupby.mean(numeric_only=True) 丢列 ----
    for m in METRIC_COLS:
        if m in work.columns:
            work[m] = pd.to_numeric(work[m], errors='coerce')

    grp = (
        work.groupby([facet_dim, x_axis, block_dim, 'config_name'],
                     dropna=False, as_index=False)
            .mean(numeric_only=True)
    )

    # 维度取值
    facets = sorted(grp[facet_dim].dropna().unique().tolist(), key=lambda v: (isinstance(v, str), v))
    x_vals = sorted(grp[x_axis].dropna().unique().tolist(), key=lambda v: (isinstance(v, str), v))
    blocks = sorted(grp[block_dim].dropna().unique().tolist(), key=lambda v: (isinstance(v, str), v))
    if set([2, 3]).issubset(set(blocks)):
        blocks = [2, 3] + [b for b in blocks if b not in (2, 3)]

    # 系列顺序：baseline(若存在) + 用户给定的 configs
    have_base = '0_base' in grp['config_name'].unique()
    series = (['0_base'] if have_base else []) + [c for c in configs if c in grp['config_name'].unique()]

    # 颜色映射
    cmap = plt.cm.get_cmap('tab10')
    color_map = {}
    if have_base:
        color_map['0_base'] = '#8c8c8c'
    for i, name in enumerate([c for c in series if c != '0_base']):
        color_map[name] = cmap(i % 10)

    # 布局（每行 1 张子图）
    n_facets = max(1, len(facets))
    n_cols = 1
    n_rows = n_facets

    L = max(1, len(series))  # 每个 block 的柱数
    bars_per_x = max(1, len(blocks)) * L
    bars_total_per_facet = max(1, len(x_vals)) * bars_per_x

    if bars_total_per_facet >= 120:
        annotate = False

    per_facet_w = max(10.0, min(0.22 * bars_total_per_facet, 18.0))
    fig_w = per_facet_w
    fig_h = max(3.7 * n_rows, 4.2)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    # 字号
    title_fs = 11
    axis_fs = 9
    tick_fs = 8
    legend_fs = 9
    annot_fs = 7

    for metric in METRIC_COLS:
        if metric not in grp.columns:
            continue

        for ar in axes:
            for ax in ar:
                ax.clear()
        # fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

        for idx_f, fval in enumerate(facets if len(facets) > 0 else [None]):
            ax = axes[idx_f][0]
            sub = grp if fval is None else grp[grp[facet_dim] == fval]

            base_x = np.arange(len(x_vals))
            total_per_x = max(1, len(blocks)) * L
            width = 0.8 / total_per_x

            xs, ys, cols, labs = [], [], [], []

            for xi, xv in enumerate(x_vals):
                for bi, b in enumerate(blocks):
                    for j, name in enumerate(series):
                        gi = sub[(sub[x_axis] == xv) & (sub[block_dim] == b) & (sub['config_name'] == name)]
                        yi = float(gi[metric].mean()) if not gi.empty else np.nan
                        xs.append(base_x[xi] - 0.4 + bi * L * width + j * width)
                        ys.append(yi)
                        cols.append(color_map.get(name, '#999999'))
                        labs.append(name)

            bars = ax.bar(xs, ys, width=width, color=cols, edgecolor='white', linewidth=0.6)

            if annotate:
                for rect, val in zip(bars, ys):
                    if np.isfinite(val):
                        ax.text(rect.get_x() + rect.get_width()/2, val,
                                f"{val:.3f}", ha='center',
                                va='bottom' if val >= 0 else 'top',
                                fontsize=annot_fs)

            # x 轴刻度（稀疏）
            ax.set_xticks(base_x)
            xticklabels = [str(x) for x in x_vals]
            sparsity = 1
            if len(x_vals) * max(1, len(blocks)) >= 12: sparsity = 2
            if len(x_vals) * max(1, len(blocks)) >= 24: sparsity = 3
            for i in range(len(xticklabels)):
                if i % sparsity != 0:
                    xticklabels[i] = ""
            ax.set_xticklabels(xticklabels, rotation=30, ha='right', fontsize=tick_fs)
            ax.tick_params(axis='y', labelsize=tick_fs)

            ax.set_xlabel(x_axis, fontsize=axis_fs)
            ax.set_ylabel(metric, fontsize=axis_fs)
            ax.set_title(f"{facet_dim}={fval}", fontsize=title_fs, y=1.08)
            ax.text(0.5, 1.02, "DGP2    |    DGP3",
                    transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=legend_fs, alpha=0.8)
            ax.grid(alpha=0.2)

            # 分割线
            if len(blocks) >= 2:
                for xi in range(len(x_vals)):
                    boundary_x = base_x[xi] - 0.4 + L * width
                    ax.axvline(boundary_x, color='k', linestyle=':', linewidth=0.8, alpha=0.25)

        # 顶部统一图例（总标题下方）
        handles, labels_legend = [], []
        for name in series:
            handles.append(plt.Rectangle((0, 0), 1, 1, color=color_map.get(name, '#999999')))
            labels_legend.append(name)

        fig.suptitle(f"{metric}  |  x={x_axis}  |  facet={facet_dim}  |  blocks by {block_dim}",
                     y=0.98, fontsize=12)
        fig.legend(handles, labels_legend,
                   loc='upper center', bbox_to_anchor=(0.5, 0.955),
                   ncol=min(6, len(labels_legend)),
                   fontsize=legend_fs, frameon=False)

        fig.subplots_adjust(top=0.90)
        fig.tight_layout(rect=[0, 0, 1, 0.90])

        out_file = os.path.join(out_dir, f"{metric}__facet_{facet_dim}__x_{x_axis}__configs.png")
        print(f"[DBG] metric={metric}, fig_num={fig.number}, axes_count={len(fig.axes)}")
        fig.savefig(out_file, dpi=220)
        plt.close(fig)
        print(f"[Plot] {out_file}")




if __name__ == "__main__":
    # root_dir = "./exp_6"
    root_dir = "./result"
    choose_num = 4
    if choose_num ==1:
        # 数据汇总
        collect_and_merge_results(root_dir)
    elif choose_num == 2:
        # 数据分析可视化草案
        agg_file = os.path.join(root_dir, "DataAnalysis", "all_results.csv")
        if os.path.exists(agg_file):
            # preferred_cols = [
            #     "config_name",
            #     "n_samples", "d_dim", "dgp_num", "n_runs",
            #     "nonlinearity", "interaction", "sparse_k", "skewness","heterogeneous", "noise_std",
            #     'rho' , 's1' , 'a0' , 'a1' , 'b0' , 'b1' ,
            # ]
            out_dir = os.path.join(root_dir, "DataAnalysis", "g2d30_noise_std")
            filter_and_plot(
                agg_file,
                out_dir=out_dir,
                factor='noise_std',
                n_samples=[50,100,200,400,800],
                # n_samples=[20, 50],
                dgp_num=2,
                d_dim=30,
            )
    elif choose_num == 3:
        # 单参数分析可视化
        facet_parameter = 'noise_std'
        agg_file = os.path.join(root_dir, "DataAnalysis", "all_results.csv")
        out_dir = os.path.join(root_dir, "DataAnalysis", "d_dim",facet_parameter)
        if os.path.exists(agg_file):
            # 横轴：样本量
            facet_single_factor(
                agg_csv_path=agg_file,
                out_dir=out_dir,
                factor=facet_parameter,
                x_axis='d_dim',
                facet_dim='n_samples',
                n_samples=[50, 100, 200, 400, 800],
                d_dim=[3, 5, 10, 20, 30],
                dgp_num=[2, 3]
            )
          # # 横轴：维度数
            # facet_single_factor(
            #     agg_csv_path=agg_file,
            #     out_dir=out_dir,
            #     factor='heterogeneous',
            #     x_axis='d_dim',
            #     facet_dim='n_samples',
            #     d_dim=[3, 5, 10, 20, 30],
            #     n_samples=[50, 200, 800],
            #     dgp_num=[2, 3]
            # )
        else:
            print("path not exist")

    elif choose_num == 4:
        # 多参数分析可视化
        facet_parameter = 'skewness'
        agg_file = os.path.join(root_dir, "DataAnalysis", "all_results_nodirect.csv")
        out_dir = os.path.join(root_dir, "DataAnalysis", "d_dim",facet_parameter)
        if os.path.exists(agg_file):
            # 横轴：样本量
            # facet_by_configs(
            #     agg_csv_path=agg_file,
            #     out_dir=out_dir,
            #     configs=['ind_nl_mD', 'ind_nl_mY', 'ind_nl_both'],  # 多参数打包配置
            #     x_axis='d_dim',  # 横轴：样本量
            #     facet_dim='n_samples',  # 子图：不同 d_dim n_samples
            #     n_samples=[50, 100, 200, 400, 800],
            #     d_dim=[3, 5, 10, 20, 30, 50, 100],
            #     dgp_num=[2, 3]  # 同图内分块：DGP2 | DGP3
            # )

            facet_by_configs(
                agg_csv_path=agg_file,
                out_dir=out_dir,
                configs=['ind_skew_Dpos', 'ind_skew_Dneg', 'ind_skew_Y'],  # 多参数打包配置
                x_axis='d_dim',  # 横轴：样本量/维度数
                facet_dim='n_samples',  # 子图：不同 d_dim / n_samples
                n_samples=[50, 100, 200, 400, 800],
                d_dim=[3, 5, 10, 20, 30, 50, 100],
                dgp_num=[2, 3]  # 同图内分块：DGP2 | DGP3
            )

        else:
            print("path not exist")

    elif choose_num == 5:
        # 正态性检验 + 临界点 + 曲线与表格
        main_normality_and_threshold(exp_root=root_dir,
                                     group_keys=['config_name', 'n_samples', 'd_dim', 'dgp_num'],
                                     agg_keys=['config_name', 'n_samples', 'd_dim', 'dgp_num'],
                                     by_keys=['config_name', 'd_dim', 'dgp_num'],
                                     coverage_thr=0.90,
                                     normality_thr=0.20)

    else:
        print("choose_num无效")