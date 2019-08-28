import numpy as np
from generate_synthetic_dataset import generate_synthetic_dataset, calc_outcome_adaptive_lasso
# TODO: define the structural parameters only once per simulation

n = 1000
gamma_convergence_factor = 2
log_lambdas = np.array([-10, -5, -2, -1, -0.75, -0.5, -0.25, 0.25, 0.49])
lambdas = n ** log_lambdas
df = generate_synthetic_dataset(n, d=20, rho=0, eta=2, num_scenario=1)
Lambda = 1
effect = calc_outcome_adaptive_lasso(df, Lambda, gamma_convergence_factor)
print(effect)
