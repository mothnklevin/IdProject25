import os, glob
import pandas as pd

# === 参数区 ===
FACTOR_NAME = "nonlinearity"  # 改成要处理的因子名
ROOT_DIR = os.path.join("result/DataAnalysis", "Step2", FACTOR_NAME)
# ROOT_DIR = os.path.join("result/DataAnalysis", "Step1")

OUT_N = os.path.join(ROOT_DIR, f"Step2_{FACTOR_NAME}_ALL_by_n.csv")
OUT_D = os.path.join(ROOT_DIR, f"Step2_{FACTOR_NAME}_ALL_by_d.csv")
TREND_IN  = os.path.join(ROOT_DIR, f"Step2_{FACTOR_NAME}_spearman_trends.csv")
# 可选：合并后是否删除原始分散的小文件（谨慎开启）
PRUNE_RAW = False

assert os.path.isdir(ROOT_DIR), f"目录不存在：{ROOT_DIR}"
assert os.path.isfile(TREND_IN), f"未找到趋势表：{TREND_IN}"

# === 收集并合并 by_n ===
pat_n = os.path.join(ROOT_DIR, f"Step2_DGP*_{FACTOR_NAME}=*_by_n.csv")
files_n = sorted(glob.glob(pat_n))
assert files_n, f"未找到 *_by_n.csv 文件：{pat_n}"

dfs_n = []
for fp in files_n:
    df = pd.read_csv(fp)
    # 统一列顺序（把识别信息放前面）
    cols_front = [c for c in ["config_name","factor","level","dgp_num","n_samples"] if c in df.columns]
    metrics = [c for c in ["bias","rmse","variance","coverage_rate","rejection_rate","mean_estimate"] if c in df.columns]
    cols = cols_front + [c for c in df.columns if c not in cols_front+metrics] + metrics
    df = df[cols]
    # 排序辅助：把 level 转成字符串避免不同类型冲突
    if "level" in df.columns:
        df["level"] = df["level"].astype(str)
    dfs_n.append(df)

all_n = pd.concat(dfs_n, ignore_index=True)
# 统一排序：dgp_num -> level -> n_samples
sort_cols = [c for c in ["dgp_num","level","n_samples"] if c in all_n.columns]
all_n = all_n.sort_values(sort_cols, kind="mergesort")
all_n.to_csv(OUT_N, index=False, encoding="utf-8-sig")

# === 收集并合并 by_d ===
pat_d = os.path.join(ROOT_DIR, f"Step2_DGP*_{FACTOR_NAME}=*_by_d.csv")
files_d = sorted(glob.glob(pat_d))
assert files_d, f"未找到 *_by_d.csv 文件：{pat_d}"

dfs_d = []
for fp in files_d:
    df = pd.read_csv(fp)
    cols_front = [c for c in ["config_name","factor","level","dgp_num","d_dim"] if c in df.columns]
    metrics = [c for c in ["bias","rmse","variance","coverage_rate","rejection_rate","mean_estimate"] if c in df.columns]
    cols = cols_front + [c for c in df.columns if c not in cols_front+metrics] + metrics
    df = df[cols]
    if "level" in df.columns:
        df["level"] = df["level"].astype(str)
    dfs_d.append(df)

all_d = pd.concat(dfs_d, ignore_index=True)
sort_cols = [c for c in ["dgp_num","level","d_dim"] if c in all_d.columns]
all_d = all_d.sort_values(sort_cols, kind="mergesort")
all_d.to_csv(OUT_D, index=False, encoding="utf-8-sig")

print(f"[OK] 合并完成：\n - {OUT_N}\n - {OUT_D}\n - 趋势表保留：{TREND_IN}")

# # === 可选：精简目录，仅保留 2 张性能表 + 1 张趋势表 ===
# if PRUNE_RAW:
#     keep = {OUT_N, OUT_D, TREND_IN}
#     for fp in glob.glob(os.path.join(ROOT_DIR, "*.csv")):
#         if os.path.abspath(fp) not in {os.path.abspath(p) for p in keep}:
#             try:
#                 os.remove(fp)
#             except Exception as e:
#                 print(f"[WARN] 删除失败：{fp} -> {e}")
#     print("[OK] 已清理原始分散文件，仅保留 2+1 输出。")
