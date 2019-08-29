import numpy as np
from outcome_adaptive_lasso import calc_outcome_adaptive_lasso, generate_synthetic_dataset
# TODO: define the structural parameters only once per simulation

dgp_params = {'n': 1000, 'd': 20, 'rho': 0, 'eta': 2, 'scenario_num': 1}
oal_params = {'gamma_convergence_factor': 2,
              'log_lambdas': np.array([-10, -5, -2, -1, -0.75, -0.5, -0.25, 0.25, 0.49])}
df = generate_synthetic_dataset(dgp_params)
ate = calc_outcome_adaptive_lasso(df, oal_params)
print(ate)
