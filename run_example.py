from tqdm import tqdm

from outcome_adaptive_lasso import calc_outcome_adaptive_lasso, generate_synthetic_dataset

oal_params = {'gamma_convergence_factor': 2,
              'log_lambdas': [-10, -5, -2, -1, -0.75, -0.5, -0.25, 0.25, 0.49]}

ate_vec = list()
for i in tqdm(range(int(100))):
    df = generate_synthetic_dataset(n=200, d=100, rho=0.5, eta=0, scenario_num=4)
    ate = calc_outcome_adaptive_lasso(df)
    ate_vec.append(ate)
