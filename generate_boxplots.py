from outcome_adaptive_lasso import calc_outcome_adaptive_lasso, generate_synthetic_dataset, calc_ate_vanilla_ipw
from collections import defaultdict
import pandas as pd
res_dict = defaultdict(list)  # {'sample_id': [], 'number_samples': [], 'method': [], 'error': []}
for rep in range(1000):
    print(rep)
    df = generate_synthetic_dataset(n=200, d=100, rho=0, eta=0, scenario_num=4)
    ate_oal = calc_outcome_adaptive_lasso(df['A'], df['Y'], df[[col for col in df if col.startswith('X')]])
    ate_conf = calc_ate_vanilla_ipw(df['A'], df['Y'], df[[col for col in df if col.startswith('Xc')]])
    ate_targ = calc_ate_vanilla_ipw(df['A'], df['Y'], df[[col for col in df if col.startswith('Xc')] +
                                                             [col for col in df if col.startswith('Xp')]])
    ate_pot_conf = calc_ate_vanilla_ipw(df['A'], df['Y'], df[[col for col in df if col.startswith('Xc')] +
                                                             [col for col in df if col.startswith('Xp')] +
                                                             [col for col in df if col.startswith('Xi')]])
    res_dict['ate'].extend([ate_oal, ate_conf, ate_targ, ate_pot_conf])
    res_dict['method'].extend(['OAL', 'Conf', 'Targ', 'PotConf'])
    res_dict['rep'].extend(4*[rep])

df_res = pd.DataFrame(res_dict)
df_res.to_csv("OAL vs PS.csv")
