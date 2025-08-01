import numpy as np

def evaluate_dml_results(estimates: list, standard_errors: list, true_theta=1.0):
    # 将输入列表转换为NumPy数组
    estimates = np.array(estimates)
    se = np.array(standard_errors)
    z = 1.96 # 置信区间的z值，对应95%置信水平

    # 1偏差：估计值平均与真实因果效应之差，衡量系统误差
    # 用于评估非线性结构、交互结构、异质性效应
    bias = np.mean(estimates) - true_theta

    # 2均方根误差（RMSE）：反映误差的整体幅度（包含偏差与方差）
    # 用于评估非线性结构、交互结构、稀疏性
    rmse = np.sqrt(np.mean((estimates - true_theta)**2))

    # 3方差：衡量估计值在重复实验中的不稳定性
    # 用于评估稀疏性、交互结构
    variance = np.var(estimates)

    # 4置信区间覆盖率：估计的置信区间覆盖真实效应的比例
    # 用于评估倾向分数偏态、异质性效应
    coverage = np.mean((estimates - z * se <= true_theta) & (estimates + z * se >= true_theta))

    # 5拒绝率：估计是否具有显著性（|t| > z）
    # 用于评估倾向分数偏态
    rejection_rate = np.mean(np.abs(estimates / se) > z)

    # 6平均估计值：辅助解释偏差与显著性表现
    mean_estimate = np.mean(estimates)

    # 返回评估指标字典
    return {
        'bias': bias,
        'rmse': rmse,
        'variance': variance,
        'coverage_rate': coverage,
        'rejection_rate': rejection_rate,
        'mean_estimate': mean_estimate
    }
