import pandas as pd
import openpyxl

# # CSV → XLSX
# df = pd.read_csv("data.csv")
# df.to_excel("data.xlsx", index=False)

# XLSX → CSV
df = pd.read_excel("./result/config1/all_results.xlsx")
df.to_csv("./result/config1/all_results.csv", index=False)

# # ===== 自适应图形尺寸与文字密度控制 =====
# n_facets = max(1, len(facets))
# n_cols = min(1, n_facets)
# n_rows = int(np.ceil(n_facets / n_cols))
# bars_per_x = max(1, len(blocks)) * L
# bars_total_per_facet = max(1, len(x_vals)) * bars_per_x