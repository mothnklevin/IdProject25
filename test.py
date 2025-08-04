import numpy as np
from DML_2 import run_doubleml_plr_rf
from EVAL_3 import evaluate_dml_results
from main_4 import plot_qq_distribution
import os

def run_synthetic_qq_test(n=500, d=5, n_runs=50, true_theta=1.0, save_dir="test_qq_output"):
    os.makedirs(save_dir, exist_ok=True)
    all_estimates = []
    np.random.seed(0)

    for i in range(n_runs):
        X = np.random.normal(0, 1, size=(n, d))
        beta = np.random.uniform(-1, 1, size=d)
        f_X = X @ beta

        gamma = np.random.normal(0, 0.3, size=d)
        p = 1 / (1 + np.exp(-X @ gamma))
        p = np.clip(p, 1e-3, 1 - 1e-3)
        D = np.random.binomial(1, p)

        Y = true_theta * D + f_X + np.random.normal(0, 1, size=n)

        result = run_doubleml_plr_rf(X, D, Y, seed=100 + i)
        all_estimates.append(("synthetic", result['theta_hat'], result['se'], true_theta))

    plot_qq_distribution(all_estimates, save_dir)
    print(f"QQ图已保存至 {save_dir}")

run_synthetic_qq_test()
