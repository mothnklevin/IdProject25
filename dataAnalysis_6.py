import os
import re
import pandas as pd

import matplotlib.pyplot as plt

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
    # if "config_name" in big.columns:
    #     idx_list, fallback = [], 0
    #     for name in big["config_name"].astype(str).tolist():
    #         m = RX_CFGIDX.match(name)
    #         if m:
    #             idx_list.append(int(m.group("idx")))
    #         else:
    #             idx_list.append(fallback)
    #             fallback += 1
    #     big["config_index"] = idx_list
    # else:
    #     big["config_name"]  = "unknown"
    #     big["config_index"] = range(len(big))
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
        "heterogeneous", "true_effect", "noise_std",
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


def filter_and_plot(agg_csv_path: str, out_dir: str, **filters):
    df = pd.read_csv(agg_csv_path)
    os.makedirs(out_dir, exist_ok=True)

    # 过滤
    used_keys = []
    for key, val in filters.items():
        if key in df.columns and val is not None:
            if not isinstance(val, (list, tuple, set)):
                val = [val]
            df = df[df[key].isin(list(val))]
            used_keys.append(key)
    if not used_keys:
        raise ValueError("必须至少指定一个 DML 参数用于筛选（例如 n_samples=[100,200], dgp_num=1）。")
    if df.empty:
        print("筛选后无数据")
        return

    # 未指定参数 = 用作分组
    unspecified = [c for c in DML_PARAM_COLS if c in df.columns and c not in used_keys]
    # 仅保留在当前数据中实际“有差异”的列
    unspecified = [c for c in unspecified if df[c].nunique(dropna=False) > 1]
    # 只取前 2 个（严格顺序）
    unspecified = sorted(unspecified, key=lambda c: PARAM_ORDER.index(c))[:2]

    # 3) 生成 x 轴标签（并对重复组合做均值聚合，稳健处理）
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
        # 严格 col1 在前，col2 在后
        df["__label__"] = df[[col1, col2]].astype(str).agg(",".join, axis=1)
        grouped = df.groupby("__label__", as_index=False).mean(numeric_only=True)
        title_suffix = f"{col1} + {col2}"
        label_cols = [col1, col2]

    # 4) 逐指标绘制柱状图（纵轴=数值；柱顶标注）
    for metric in METRIC_COLS:
        if metric not in grouped.columns:
            continue

        x_labels = grouped["__label__"].tolist()
        y_vals = grouped[metric].values

        plt.figure(figsize=(8, 5))
        bars = plt.bar(x_labels, y_vals)

        # 柱顶标注具体数值
        for b, v in zip(bars, y_vals):
            plt.text(b.get_x() + b.get_width() / 2, v,
                     f"{v:.3f}", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=9)

        # 组装标题：config_name（若有） + ' — ' + 差异参数标题

        full_title = title_suffix

        plt.title(full_title)
        plt.xlabel(" , ".join(label_cols) if label_cols else "")
        plt.ylabel(metric)
        plt.xticks(rotation=0)

        # 输出文件名：metric + 标题安全化
        safe_suffix = re.sub(r"[^\w\-_.\u4e00-\u9fa5]", "_", title_suffix) if title_suffix else "all"
        base = os.path.splitext(os.path.basename(agg_csv_path))[0]

        # out_file = os.path.join(out_dir, f"{base}__{metric}__{safe_suffix}.png")
        if unspecified:
            suffix = "+".join(unspecified)
            out_file = os.path.join(out_dir, f"{metric}__{suffix}.png")
        else:
            out_file = os.path.join(out_dir, f"{metric}.png")

        plt.tight_layout()
        plt.savefig(out_file)
        plt.close()
        print(f"[Plot] {out_file}")

if __name__ == "__main__":
    choose_num = 2
    if choose_num ==1:
        # 指定要处理的实验文件夹（示例："./exp_2"）
        collect_and_merge_results("./exp_3")
    elif choose_num == 2:
        agg_file = "./exp_2/DataAnalysis/all_results.csv"
        if os.path.exists(agg_file):
            filter_and_plot(
                agg_file,
                out_dir="./exp_2/DataAnalysis/n100_200_g2",
                n_samples=[100, 200],
                dgp_num=2,
            )
        else:
            print("找不到示例文件，请先运行汇总步骤。")
    else:
        print("choose_num无效")

    # agg_file = "./exp_1/DataAnalysis/agg_config0_0_基准.csv"
    # if os.path.exists(agg_file):
    #     filter_and_plot(
    #         agg_file,
    #         out_dir="./exp_1/DataAnalysis/config0_",
    #         n_samples=[200],
    #         dgp_num=2,
    #         # d_dim=10,
    #     )
    # else:
    #     print("找不到示例文件，请先运行汇总步骤。")