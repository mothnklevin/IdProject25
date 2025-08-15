from __future__ import annotations

import numpy as np
from scipy.special import expit
from scipy.linalg import toeplitz
from typing import Dict, Any

from doubleml.datasets import make_plr_CCDDHNR2018


# -------------------- 默认参数（统一且数值化） --------------------
DEFAULTS: Dict[str, Any] = {
    # 公共参数
    'true_effect': 1.0,       # θ 的基准截距；异质性关闭时即为因果效应常数
    'noise_std': 1.0,         # 观测噪声 ε 的标准差 -- CDDDHNR2018：s2
    'interaction': 0.0,       # 交互项强度：乘在 (D * X[:, interaction_idx]) 前
    'heterogeneous': 0.0,     # 异质性强度：θ(X) = true_effect + heterogeneous * X[:, hetero_idx]
    'interaction_idx': 0,     # D·X 交互使用的列索引
    'hetero_idx': 0,          # θ(X) 里使用的列索引

    # v0/1 公用
    'nonlinearity': 0.0,  # g(X) 中非线性项的强度（0=无，>0=放大系数）
    # v0专用（通用二元处理）
    'skewness': 0.3,          # 倾向评分 γ 的尺度；越大 D 越偏态/极端
    'sparse_k': 0,            # β 的非零个数；0 或 >=d 表示用全量随机 β（不稀疏）

    # v1专用（兴趣-容忍度）
    'bg_start': 2,            # 背景变量起始列（含）
    'bg_end': 5,              # 背景变量结束列（不含）

    # v2专用（CDDDHNR2018 连续处理）
    'rho': 0.7,               # Toeplitz 协方差里的相关系数 ρ,该值在官方DGP中固定为0.7
    's1': 1.0,                # 连续处理的噪声尺度（D = m0 + s1*v）
    'a0': 1.0, 'a1': 0.25,    # m0(X) 的系数（a0·x0 + a1·sigmoid(x2)）
    'b0': 1.0, 'b1': 0.25,    # g(X) 的系数（b0·sigmoid(x0) + b1·x2）
}

# 原布尔值，更新为 True=1 / False=0
BOOL_KEYS = ['nonlinearity', 'interaction', 'heterogeneous']

# 合并 config 到 DEFAULTS
def _merge_cfg(user: Dict[str, Any] | None) -> Dict[str, Any]:
    cfg = dict(DEFAULTS)
    if user:
        cfg.update(user)
    # 兼容旧布尔值
    for k in BOOL_KEYS:
        if isinstance(cfg.get(k), bool):
            cfg[k] = 1.0 if cfg[k] else 0.0
    # 稀疏参数清洗
    sk = cfg.get('sparse_k', 0)
    cfg['sparse_k'] = int(sk) if isinstance(sk, (int, float)) else 0
    return cfg

# -------------------- 工具函数 --------------------
# 生成 θ：若 heterogeneous<=0 则返回常数，否则按指定列添加异质性项
def _theta(X, cfg):
    if cfg['heterogeneous'] <= 0:        # 无异质性时
        return cfg['true_effect']       # 直接返回标量 θ
    idx = int(cfg['hetero_idx']) % X.shape[1]  # 防越界：索引取模到列数
    return cfg['true_effect'] + cfg['heterogeneous'] * X[:, idx]  # θ(x)

# 在 Y 上加入 D·X[:,idx] 的交互项（按强度 interaction）
def _add_interaction(Y, D, X, cfg):
    s = cfg['interaction']                 # 交互强度
    if s > 0:                              # 仅在开启时添加
        idx = int(cfg['interaction_idx']) % X.shape[1]  # 防越界
        Y = Y + s * D * X[:, idx]          # 加交互项
    return Y

# -------------------- 一个函数：按 dgp_num 切结构 --------------------

def generate_dgp(n: int, d: int, dgp_num: int = 0, cfg: Dict[str, Any] | None = None, seed: int = 42):
    # 返回 (X, D, Y)。
    # dgp_num:
    #   0 = 简单原型，通用二元处理（p-score 二项法）：可调 skewness / 稀疏线性 / 非线性等
    #   1 = 广告模型，兴趣-容忍度（binary）：D = Bernoulli(sigmoid(1.5*x0 + 1.0*x1)); g 在背景变量上
    #   2 = Chernozhukov, CDDDHNR2018（continuous）：D = m0(X)+s1*v; g = g0(X); X ~ N(0, Toeplitz(rho))
    # cfg 只负责强度与系数；0 表示关闭。

    rng = np.random.default_rng(seed)  # 随机数（确保可复现）
    C = _merge_cfg(cfg)                # 合并 cfg

    # --- 生成 X ---
    if dgp_num in (0, 1):                             # 结构0/1：独立标准正态，暂时直接简单生成
        X = rng.standard_normal(size=(n, d))          # X ~ N(0, I)

    elif dgp_num == 2:                                 # 结构2：Toeplitz 相关正态，doubleML给出了明确结构
        first_col = (C['rho'] ** np.arange(d))        # Toeplitz 第一列：rho^{|j|}
        Sigma = toeplitz(first_col)                   # 构造协方差矩阵 Σ
        L = np.linalg.cholesky(Sigma)                 # Cholesky 分解：Σ = L L^T
        X = rng.standard_normal(size=(n, d)) @ L.T    # 生成相关正态：Z L^T
    elif dgp_num == 3:
        np.random.seed(seed)

        # alpha 用 true_effect 对齐既有含义；返回数组以与现有管线一致
        X, Y, D = make_plr_CCDDHNR2018(
            n_obs=n,
            dim_x=d,
            alpha=C['true_effect'],
            a0=C['a0'], a1=C['a1'],
            b0=C['b0'], b1=C['b1'],
            s1=C['s1'], s2=C['noise_std'],
            return_type='array'
        )
        # 与现有接口保持一致：返回 (X, D, Y)
        return X, D, Y
    else:
        raise ValueError(f"未知 dgp_num={dgp_num}")   # 不支持的结构号

    # --- 生成 D（按结构差异） ---
    # 1 简单原型
    if dgp_num == 0:                                   # 通用二元处理：p-score→Bernoulli
        gamma = rng.normal(0.0, C['skewness'], size=d) # γ ~ N(0, skewness^2)
        p = expit(X @ gamma)                           # p = sigmoid(Xγ)
        p = np.clip(p, 1e-6, 1-1e-6)                   # 避免 0/1 边界
        D = rng.binomial(1, p)                         # D ~ Bernoulli(p)
        if len(np.unique(D)) < 2:                      # 保障有变异
            raise RuntimeError("D 全为同一类；调参 skewness 或换 seed")
    # 2 广告点击模型
    elif dgp_num == 1:                                 # 兴趣-容忍度：固定系数生成 p-score
        logits = 1.5 * X[:, 0] + 1.0 * X[:, 1]        # logit p = 1.5*x0 + 1.0*x1
        p = expit(logits)                              # p = sigmoid(logits)
        p = np.clip(p, 1e-6, 1-1e-6)                   # 数值裁剪
        D = rng.binomial(1, p)                         # D 二元
    # 3 Chernozhukov ： PLR DGP -- doubleml.datasets.make_plr_CCDDHNR2018
    elif dgp_num == 2:                                 # CDDDHNR2018：连续 D
        m0 = C['a0'] * X[:, 0] + C['a1'] * expit(X[:, 2])  # m0(X)
        v = rng.standard_normal(n)                          # 高斯噪声 v
        D = m0 + C['s1'] * v                                # 连续处理 D

    # --- 构造 g(X)（按结构差异） ---
    # 1 简单原型
    if dgp_num == 0:                                   # 通用：线性(可稀疏) + 非线性
        if C['sparse_k'] <= 0 or C['sparse_k'] >= d:   # 不稀疏时：全维随机 β
            beta = rng.uniform(-1, 1, size=d)          # β ~ U(-1,1)
        else:                                          # 稀疏：前 k 维非零
            beta = np.zeros(d)                         # 先置零
            beta[:C['sparse_k']] = rng.uniform(-1, 1, size=C['sparse_k'])  # 前 k 赋值
        g = X @ beta                                   # 线性部分
        if C['nonlinearity'] > 0:                      # 可选非线性
            x0 = X[:, 0]                               # 第1列
            x1 = X[:, 1] if d > 1 else X[:, 0]        # 第2列（不足则复用第1列）
            g = g + C['nonlinearity'] * (x0**2 + np.sin(x1))  # 加非线性项
    # 2 广告点击模型
    elif dgp_num == 1:                                 # 兴趣-容忍度：非线性加在背景列
        # 计算背景列范围并裁剪防越界
        bg_start = int(np.clip(C['bg_start'], 0, d))   # 起始列
        bg_end = int(np.clip(C['bg_end'], bg_start, d))# 终止列（不含）
        s = slice(bg_start, bg_end)                    # 切片对象
        B = X[:, s]                                    # 取背景子矩阵 (n, q)
        lin = B @ rng.uniform(-1, 1, size=B.shape[1]) if B.shape[1] > 0 else 0.0  # 线性项
        g = lin                                        # 初始化 g
        if C['nonlinearity'] > 0 and B.shape[1] > 0:   # 若开启非线性且有背景列
            term = np.sin(B[:, 0])                     # 第一背景列的 sin
            if B.shape[1] >= 2:                        # 至少两列时
                term = term + B[:, 1]**2               # 加上第二列平方
            g = g + C['nonlinearity'] * term           # 加权叠加
    # 3 Chernozhukov ： PLR DGP -- doubleml.datasets.make_plr_CCDDHNR2018
    elif dgp_num == 2:                                 # CDDDHNR2018：固定 g0 形式
        # 注意：若 d<3 将退化为使用可用列（这里假设 d>=3 更符合标准设置）
        g = C['b0'] * expit(X[:, 0]) + C['b1'] * X[:, 2]  # g0(X)

    # --- θ 与 Y ---
    theta = _theta(X, C)                               # 可能为标量或长度 n 的向量
    eps = rng.normal(0, C['noise_std'], size=n)        # 结果噪声 ε
    Y = theta * D + g + eps                            # 结构方程：Y = θ·D + g + ε
    Y = _add_interaction(Y, D, X, C)                   # 可选：加入 D·X 交互

    return X, D, Y                                     # 返回三元组

# -------------------- 兼容层--------------------
# 旧代码：generate_controlled_dgp(X_real, **kwargs) -> (X,D,Y)

def generate_controlled_dgp(X_real: np.ndarray, **kwargs):
    n, d = X_real.shape                               # 从输入推断维度
    seed = kwargs.pop('random_seed', 42)              # 读取并移除旧参数 random_seed
    # 将旧参数名映射为新名
    cfg_map = {
        'skewness_level': 'skewness',
        'sparse_beta': 'sparse_k',
    }
    cfg = {cfg_map.get(k, k): v for k, v in kwargs.items()}  # 映射键名
    return generate_dgp(n, d, dgp_num=0, cfg=cfg, seed=seed) # 统一转调, 结构0


def generate_controlled_dgp_1(X_real: np.ndarray, **kwargs):
    n, d = X_real.shape
    seed = kwargs.pop('random_seed', 42)
    cfg_map = {'skewness_level': 'skewness', 'sparse_beta': 'sparse_k'}  # 键名映射
    cfg = {cfg_map.get(k, k): v for k, v in kwargs.items()}  # 应用映射
    return generate_dgp(n, d, dgp_num=1, cfg=cfg, seed=seed) # 结构1


def generate_controlled_dgp_2(X_real: np.ndarray, **kwargs):
    n, d = X_real.shape
    seed = kwargs.pop('random_seed', 42)
    cfg_map = {'skewness_level': 'skewness', 'sparse_beta': 'sparse_k'}  # 键名映射
    cfg = {cfg_map.get(k, k): v for k, v in kwargs.items()}  # 应用映射
    return generate_dgp(n, d, dgp_num=2, cfg=cfg, seed=seed) # 结构2
