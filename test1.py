import numpy as np
from scipy.stats import spearmanr
import pandas as pd
import os

# === Load data ===
csv_path = "./result/DataAnalysis/all_results.csv"
OUT_DIR = "./result/DataAnalysis/Step1"             # 本步骤输出目录
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(csv_path)

# 只保留核心列（配置名 + 维度/样本量/DGP编号 + 六个指标）
metrics = ['bias','rmse','variance','coverage_rate','rejection_rate','mean_estimate']
core_cols = ['config_name','n_samples','d_dim','dgp_num'] + metrics
missing = [c for c in core_cols if c not in df.columns]
assert not missing, f"Missing columns in CSV: {missing}"  # 确认文件包含所有必需列

df = df[core_cols].copy()

# 基线配置 0_base
base = df[df['config_name'] == '0_base'].copy()
assert not base.empty, "未在 all_results.csv 中找到基线配置 '0_base' 的记录。"

def agg_by_axis(g, axis='n_samples'):
    # 对某一维度 (n_samples 或 d_dim) 分组，计算各指标的平均值。
    # 得到随样本量或随维度的平均趋势（另一维平均掉）。
    tab = g.groupby(axis, as_index=False).agg({m: 'mean' for m in metrics})
    tab['config_name'] = g['config_name'].iloc[0]  # 统一填回 "0_base"
    tab = tab.sort_values(axis)  # 按数值排序，确保绘图/趋势检验时有序
    # 调整列顺序：把 config_name 放到第一列
    cols = ['config_name', axis] + [c for c in tab.columns if c not in ['config_name', axis]]
    tab = tab[cols]
    return tab

def spearman_trend(tab, axis):
    # 计算指标与某一维度的 Spearman 等级相关系数 (ρ, p)，量化“指标随 n 或 d 的单调趋势”。
    # tab： agg_by_axis 的结果；axis：'n_samples' 或 'd_dim'。
    res = {}
    x = tab[axis].to_numpy()
    for m in metrics:
        try:
            rho, p = spearmanr(x, tab[m].to_numpy())  # 相关系数与显著性
            res[m] = {'rho': float(rho), 'p': float(p)}
        except Exception:
            res[m] = {'rho': np.nan, 'p': np.nan}  # 异常时返回 NaN
    return res

# 针对 DGP2 和 DGP3 计算趋势，并构建“n/d 的基线均值表”
results = {}  # 保存中间结果（便于后续内存复用）
for dgp in [2, 3]:
    g = base[base['dgp_num'] == dgp].copy()
    if g.empty:
        print(f"[WARN] 基线在 DGP{dgp} 下无记录，跳过。")
        continue
    # 1) 沿样本量聚合（另一维度 d_dim 做均值）
    tab_n = agg_by_axis(g, 'n_samples')
    tab_n['dgp_num'] = dgp
    tab_n = tab_n[['config_name', 'dgp_num', 'n_samples'] + metrics]
    tab_n.to_csv(os.path.join(OUT_DIR, f"Step1_DGP{dgp}_baseline_by_n.csv"),
                 index=False, encoding="utf-8-sig")

    # 2) 沿维度聚合（另一维度 n_samples 做均值）
    tab_d = agg_by_axis(g, 'd_dim')
    tab_d['dgp_num'] = dgp
    tab_d = tab_d[['config_name', 'dgp_num', 'd_dim'] + metrics]
    tab_d.to_csv(os.path.join(OUT_DIR, f"Step1_DGP{dgp}_baseline_by_d.csv"),
                 index=False, encoding="utf-8-sig")

    # 3) 计算 Spearman 趋势（两个轴各一份）
    trend_n = spearman_trend(tab_n, 'n_samples')
    trend_d = spearman_trend(tab_d, 'd_dim')

    # 保存到内存字典
    results[dgp] = {'tab_n': tab_n, 'tab_d': tab_d,
                    'trend_n': trend_n, 'trend_d': trend_d}

# 汇总 DGP×轴 的趋势系数与显著性到一张总表
rows = []
for dgp in [2, 3]:
    if dgp not in results:
        continue
    for axis in ['n_samples', 'd_dim']:
        tab = results[dgp]['tab_n'] if axis == 'n_samples' else results[dgp]['tab_d']
        cfg = tab['config_name'].iloc[0] if ('config_name' in tab.columns and not tab.empty) else '0_base'
        tr = results[dgp]['trend_n' if axis == 'n_samples' else 'trend_d']
        for m in metrics:
            rows.append({
                'config_name' : cfg,
                'dgp_num'     : dgp,
                'axis'        : axis,
                'metric'      : m,
                'spearman_rho': tr[m]['rho'],
                'p_value'     : tr[m]['p'],
            })

trend_df = pd.DataFrame(rows)
# config_name 放到第一列
trend_df = trend_df[['config_name','dgp_num','axis','metric','spearman_rho','p_value']] \
           .sort_values(['dgp_num','axis','metric'])



# 基线趋势的汇总表
trend_path = os.path.join(OUT_DIR, "Step1_baseline_spearman_trends.csv")
trend_df.to_csv(trend_path, index=False, encoding="utf-8-sig")

print(f"Step1 完成：输出目录 = {OUT_DIR}")
print(f" - DGP2/3 基线表：Step1_DGP*_baseline_by_n.csv, Step1_DGP*_baseline_by_d.csv")
print(f" - 趋势总表    ：{trend_path}")

