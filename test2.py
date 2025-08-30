import numpy as np
from scipy.stats import spearmanr
import pandas as pd
import os
# === Load data ===
csv_path = "./result/DataAnalysis/all_results_nodirect.csv"
df_full = pd.read_csv(csv_path)


# 只保留核心列（配置名 + 维度/样本量/DGP编号 + 六个指标）
metrics = ['bias','rmse','variance','coverage_rate','rejection_rate','mean_estimate']
core_cols = ['config_name','n_samples','d_dim','dgp_num'] + metrics

# missing = [c for c in core_cols if c not in df_full.columns]
# assert not missing, f"Missing columns in CSV: {missing}"  # 确认文件包含所有必需列
# df = df_full[core_cols].copy()

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

# 为了排序友好，尝试把 factor 水平转为浮点；失败则按字符串排序
def _sort_levels(vals):
    try:
        return sorted(vals, key=lambda v: float(v))
    except Exception:
        return sorted(vals, key=lambda v: (isinstance(v, str), v))

def _mask_by_conds(frame, conds: dict):
    m = np.ones(len(frame), dtype=bool)
    for k, v in conds.items():
        if k not in frame.columns:
            raise KeyError(f"[Step2] 缺少列 {k}（复合特征需要）")
        m &= (frame[k] == v)
    return m

# Step2 入口配置（两种模式）

# 【模式A：单列因子】/interaction /heterogeneous /noise_std /rho /s1
# FACTOR_NAME = 'noise_std'         # 输出用的文件名字（目录/文件/列）
# # FACTOR_COL  = 'noise_std'         #  CSV 里的列名（单列因子）
# FACTOR_COL  = FACTOR_NAME
# LEVELS_DEF  = None                # 单列因子时设为 None

# 【模式B：复合特征因子】多列参数合并，定义“一个特征”的不同水平
# 间接-非线性（D侧 / Y侧 / 双侧）
# FACTOR_NAME = 'nonlinearity'
# FACTOR_COL  = None
# LEVELS_DEF  = [
#     {'name': '0_base', 'conds': {'config_name': '0_base'}},  # 基准
#     {'name': 'ind_nl_mD',   'conds': {'a0': 0.0, 'a1': 2.0}},                      # D 更非线性
#     {'name': 'ind_nl_mY',   'conds': {'b0': 2.0, 'b1': 0.0}},                      # Y 更非线性
#     {'name': 'ind_nl_both', 'conds': {'a0': 0.0, 'a1': 2.0, 'b0': 2.0, 'b1': 0.0}} # 双侧
# ]

# 间接-偏态
FACTOR_NAME = 'skewness'
FACTOR_COL  = None
LEVELS_DEF  = [
    {'name': '0_base', 'conds': {'config_name': '0_base'}},  # 基准
    {'name': 'ind_skew_Dpos', 'conds': {'a0': 0.0, 'a1': 4.0, 's1': 0.2}},
    {'name': 'ind_skew_Dneg', 'conds': {'a0': 0.0, 'a1': -4.0, 's1': 0.2}},
    {'name': 'ind_skew_Y', 'conds': {'b0': 4.0, 'b1': 0.0, 'noise_std': 0.5}},
]

OUT_DIR2 = os.path.join("./result/DataAnalysis", "Step2", FACTOR_NAME)
os.makedirs(OUT_DIR2, exist_ok=True)

rows2 = []

for dgp in [2, 3]:
    base = df_full[df_full['dgp_num'] == dgp].copy()
    if base.empty:
        print(f"[Step2][WARN] DGP{dgp} 无数据，跳过。")
        continue

    if FACTOR_COL is not None:
        # 模式A：单列因子
        need_cols = set(core_cols + [FACTOR_COL])
        missing2 = [c for c in need_cols if c not in base.columns]
        assert not missing2, f"[Step2] CSV缺少必要列：{missing2}"

        g_dgp = base[list(need_cols)].copy()
        levels = sorted(g_dgp[FACTOR_COL].dropna().unique().tolist(), key=lambda x: (isinstance(x,str), x))
        level_gen = [(str(lv), g_dgp[g_dgp[FACTOR_COL] == lv].copy()) for lv in levels]

    else:
        # 模式B：复合特征因子
        assert LEVELS_DEF and isinstance(LEVELS_DEF, list), "[Step2] 复合特征请提供 LEVELS_DEF"
        # 需要的列 = core + 所有 conds 涉及的参数列
        need_cols = set(core_cols)
        for lev in LEVELS_DEF:
            need_cols.update(lev['conds'].keys())
        missing2 = [c for c in need_cols if c not in base.columns]
        assert not missing2, f"[Step2] CSV缺少必要列：{missing2}"

        g_dgp = base[list(need_cols)].copy()
        level_gen = []
        for lev in LEVELS_DEF:
            name  = lev['name']
            conds = lev['conds']
            mask  = _mask_by_conds(g_dgp, conds)
            sub   = g_dgp[mask].copy()
            if sub.empty:
                print(f"[Step2][WARN] DGP{dgp} 复合特征 {name} 无匹配行，跳过。")
                continue
            level_gen.append((name, sub))

    # 对于本 DGP 下的各“水平”逐一生成 by_n / by_d 聚合与趋势
    for lev_name, g_lev in level_gen:
        # 1) by_n
        tab_n = agg_by_axis(g_lev, axis='n_samples')
        tab_n.insert(1, 'factor', FACTOR_NAME)
        tab_n.insert(2, 'level', lev_name)
        tab_n.insert(3, 'dgp_num', dgp)
        tab_n.to_csv(os.path.join(OUT_DIR2, f"Step2_DGP{dgp}_{FACTOR_NAME}={lev_name}_by_n.csv"),
                     index=False, encoding="utf-8-sig")

        # 2) by_d
        tab_d = agg_by_axis(g_lev, axis='d_dim')
        tab_d.insert(1, 'factor', FACTOR_NAME)
        tab_d.insert(2, 'level', lev_name)
        tab_d.insert(3, 'dgp_num', dgp)
        tab_d.to_csv(os.path.join(OUT_DIR2, f"Step2_DGP{dgp}_{FACTOR_NAME}={lev_name}_by_d.csv"),
                     index=False, encoding="utf-8-sig")

        # 3) Spearman 趋势
        tr_n = spearman_trend(tab_n.rename(columns={'n_samples':'__n'}), axis='__n')
        tr_d = spearman_trend(tab_d.rename(columns={'d_dim':'__d'}),      axis='__d')
        for m in metrics:
            rows2.append({
                'config_name' : tab_n.get('config_name', pd.Series(['NA'])).iloc[0],
                'dgp_num'     : dgp,
                'factor'      : FACTOR_NAME,
                'level'       : lev_name,
                'axis'        : 'n_samples',
                'metric'      : m,
                'spearman_rho': tr_n[m]['rho'],
                'p_value'     : tr_n[m]['p'],
            })
            rows2.append({
                'config_name' : tab_d.get('config_name', pd.Series(['NA'])).iloc[0],
                'dgp_num'     : dgp,
                'factor'      : FACTOR_NAME,
                'level'       : lev_name,
                'axis'        : 'd_dim',
                'metric'      : m,
                'spearman_rho': tr_d[m]['rho'],
                'p_value'     : tr_d[m]['p'],
            })

# 趋势总表
trend2_df = pd.DataFrame(rows2)
if not trend2_df.empty:
    cols = ['config_name','dgp_num','factor','level','axis','metric','spearman_rho','p_value']
    cols = [c for c in cols if c in trend2_df.columns] + [c for c in trend2_df.columns if c not in cols]
    trend2_df = trend2_df[cols].sort_values(['dgp_num','factor','level','axis','metric'])
    trend2_df.to_csv(os.path.join(OUT_DIR2, f"Step2_{FACTOR_NAME}_spearman_trends.csv"),
                     index=False, encoding="utf-8-sig")

print(f"[Step2] 完成单分子（单特征）分析：factor={FACTOR_NAME}；输出目录 = {OUT_DIR2}")
