from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def run_doubleml_plr_rf(X, D, Y, n_folds=5, seed=60):
    dml_data = DoubleMLData.from_arrays(X, Y, D)
    model = DoubleMLPLR(
        dml_data,
        ml_l=RandomForestRegressor(random_state=seed),
        ml_m=RandomForestClassifier(random_state=seed),
        n_folds=n_folds,
    )
    model.fit()
    return {
        'theta_hat': model.coef[0],
        'se': model.se[0],
        'ci_lower': model.confint().iloc[0, 0],
        'ci_upper': model.confint().iloc[0, 1],
        'significant': abs(model.t_stat[0]) > 1.96
    }

