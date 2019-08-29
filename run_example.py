import numpy as np
from outcome_adaptive_lasso import calc_outcome_adaptive_lasso, generate_synthetic_dataset
# TODO: define the structural parameters only once per simulation

n = 1000
gamma_convergence_factor = 2
log_lambdas = np.array([-10, -5, -2, -1, -0.75, -0.5, -0.25, 0.25, 0.49])
lambdas = n ** log_lambdas
df = generate_synthetic_dataset(n, d=20, rho=0, eta=2, num_scenario=1)
amd_vec = np.zeros(log_lambdas.shape[0])
ate_vec = np.zeros(log_lambdas.shape[0])
for il, Lambda in enumerate(log_lambdas):
    amd_vec[il] = 1

    ate_vec[il] = calc_outcome_adaptive_lasso(df, Lambda, gamma_convergence_factor)


