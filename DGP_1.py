
import numpy as np
import pandas as pd

from scipy.special import expit
from sklearn.preprocessing import StandardScaler

# 数据加载函数（支持子样本提取）
def load_twins_X(file_path, n_samples=500, seed=42):
    df = pd.read_csv(file_path)
    df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col])
    # 数据清洗： 只保留数值列 + 缺失值填充 （需进一步测试）
    df = df.select_dtypes(include=[np.number])
    df = df.fillna(df.mean())  # 或使用 fillna(0) 等其他策略

    np.random.seed(seed)
    idx = np.random.choice(df.shape[0], n_samples, replace=False)

    X_sub = df.iloc[idx].to_numpy()

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)

    return X_scaled


def generate_controlled_dgp(X_real: np.ndarray,
                             true_effect: float = 1.0,
                             noise_std: float = 1.0,
                             nonlinearity: bool = False,
                             interaction: bool = False,
                             skewness_level: float = 0.3,
                             sparse_beta: bool = False,
                             heterogeneous: bool = False,
                             random_seed: int = 42):

    # 参数：   X_real : 原始特征矩阵 (n, d)
    #         其他：特征控制变量
    # 返回：   X, D, Y : 三个 ndarray，可用于 DoubleML

    np.random.seed(random_seed)
    if np.isnan(X_real).any():
        raise ValueError("X contains NaN. 请在数据加载阶段先进行清洗。")

    n, d = X_real.shape
    X = X_real.copy()



    # 倾向评分控制：生成 D
    gamma = np.random.normal(loc=0.0, scale=skewness_level, size=d)
    logits = X @ gamma
    p_score = expit(logits) # 使用稳定 sigmoid
    p_score = np.clip(p_score, 1e-6, 1 - 1e-6) # 避免出现 0/1 导致 binomial 报错
    # 强制调试输出
    # print("倾向得分分布: min =", p_score.min(), ", max =", p_score.max())
    # print("X 均值范围:", X.mean(axis=0).min(), "-", X.mean(axis=0).max())
    # print("X 方差范围:", X.std(axis=0).min(), "-", X.std(axis=0).max())
    # print("logits 范围:", logits.min(), "-", logits.max())

    D = np.random.binomial(1, p_score)
    # 如果 D 只有一个值，则重新采样 gamma 或报错
    if len(np.unique(D)) < 2:
        raise ValueError("生成的 D 全为同一类别，请调低 skewness_level 或更换 gamma 初始化。")

    # 稀疏 β 控制：build β
    if sparse_beta:
        beta = np.zeros(d)
        beta[:5] = np.random.uniform(-1, 1, size=5)
    else:
        beta = np.random.uniform(-1, 1, size=d)

    # 非线性结构控制：build f(X)
    if nonlinearity:
        f_X = X[:, 0]**2 + np.sin(X[:, 1])
    else:
        f_X = X @ beta

    #  异质效应控制：build θ
    if heterogeneous:
        theta = true_effect + 0.3 * X[:, 0]
    else:
        theta = true_effect

    #  构造结果变量 Y ：build Y
    eps = np.random.normal(0, noise_std, size=n) # fixed noise_std
    Y = theta * D + f_X + eps

    #  交互项控制：加入 D·X₀ 项 ： add D·X₀
    if interaction:
        Y += 0.5 * D * X[:, 0]

    return X, D, Y
