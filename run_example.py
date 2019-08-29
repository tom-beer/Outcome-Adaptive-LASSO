import numpy as np
from outcome_adaptive_lasso import calc_outcome_adaptive_lasso, generate_synthetic_dataset
# TODO: define the structural parameters only once per simulation

n = 1000
d = 20
gamma_convergence_factor = 2
log_lambdas = np.array([-10, -5, -2, -1, -0.75, -0.5, -0.25, 0.25, 0.49])
lambdas = n ** log_lambdas
df = generate_synthetic_dataset(n, d, rho=0, eta=2, num_scenario=1)
ate = calc_outcome_adaptive_lasso(df, lambdas, gamma_convergence_factor)




