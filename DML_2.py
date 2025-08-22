from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold

def run_doubleml_plr_rf(X, D, Y, n_folds=5, seed=60):
    dml_data = DoubleMLData.from_arrays(X, Y, D)

    #  PLR 允许使用回归器拟合二元D，因此将ml_m也直接使用RF回归拟合，作为基本模型
    model = DoubleMLPLR(
        dml_data,
        ml_l=RandomForestRegressor(
            n_jobs=-1,
            random_state=seed),
        ml_m=RandomForestRegressor(
            n_jobs=-1,
            random_state=seed),
        n_folds=n_folds,
        draw_sample_splitting=False,  # 关闭自动随机抽样
    )

    # 使用固定的可复现 K 折划分
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed) # 显式生成 K 折划分
    smpls = [(train_idx, test_idx) for train_idx, test_idx in kf.split(X)] # 构建列表
    model.set_sample_splitting(smpls)  # 显式传入划分


    model.fit()
    return {
        'theta_hat': model.coef[0],
        'se': model.se[0],
        'ci_lower': model.confint().iloc[0, 0],
        'ci_upper': model.confint().iloc[0, 1],
        'significant': abs(model.t_stat[0]) > 1.96
    }

