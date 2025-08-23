import numpy as np

def evaluate_dml_results(estimates: list, standard_errors: list, true_theta=1.0):
    # 将输入列表转换为NumPy数组
    estimates_array = np.asarray(estimates, dtype=float)
    standard_errors_array = np.asarray(standard_errors, dtype=float)
    true_theta_input = np.asarray(true_theta, dtype=float)

    # 统一 true_theta 形状：标量或与 estimates 等长的一维数组
    if true_theta_input.shape == ():
        true_theta_array = np.full_like(estimates_array, float(true_theta_input))
    else:
        if true_theta_input.shape != estimates_array.shape:
            # 可广播则广播；否则报错
            try:
                true_theta_array = np.broadcast_to(true_theta_input, estimates_array.shape).astype(float)
            except ValueError:
                raise ValueError(
                    f"true_theta 的形状 {true_theta_input.shape} 与 estimates {estimates_array.shape} 不兼容。"
                )
        else:
            true_theta_array = true_theta_input
    # 避免除零
    standard_errors = np.where(standard_errors_array > 0, standard_errors_array, np.inf)
    differences = estimates_array - true_theta_array


    z = 1.96 # 置信区间的z值，对应95%置信水平

    # 1偏差：估计值平均与真实因果效应之差，衡量系统误差
    # 用于评估非线性结构、交互结构、异质性效应
    bias = float(np.mean(differences))


    # 2均方根误差（RMSE）：反映误差的整体幅度（包含偏差与方差）
    # 用于评估非线性结构、交互结构、稀疏性
    rmse = float(np.sqrt(np.mean(differences ** 2)))

    # 3方差：衡量估计值在重复实验中的不稳定性
    # 用于评估稀疏性、交互结构
    variance = float(np.var(estimates_array, ddof=0))

    # 4置信区间覆盖率：估计的置信区间覆盖真实效应的比例
    # 用于评估倾向分数偏态、异质性效应

    coverage_rate = float(np.mean(
        (estimates_array - z * standard_errors <= true_theta_array) &
        (estimates_array + z * standard_errors >= true_theta_array)
    ))

    # 5拒绝率：估计是否具有显著性（|t| > z）
    # 用于评估倾向分数偏态
    # rejection_rate = np.mean(np.abs(estimates_array - true_theta_array / standard_errors) > z)
    rejection_rate = float(np.mean(np.abs(differences / standard_errors) > z))

    # 6平均估计值：辅助解释偏差与显著性表现
    mean_estimate = float(np.mean(estimates_array))

    # 返回评估指标字典
    return {
        'bias': bias,
        'rmse': rmse,
        'variance': variance,
        'coverage_rate': coverage_rate,
        'rejection_rate': rejection_rate,
        'mean_estimate': mean_estimate
    }
