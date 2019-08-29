import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from causallib.estimation import IPW
from math import log


def generate_col_names(d):
    assert(d >= 6)
    pC = 2  # number of confounders
    pP = 2  # number of outcome predictors
    pI = 2  # number of exposure predictors
    pS = d - (pC + pI + pP)  # number of spurious covariates
    col_names = ['A', 'Y'] + [f'Xc{i}' for i in range(1, pC + 1)] + [f'Xp{i}' for i in range(1, pP + 1)] + \
                [f'Xi{i}' for i in range(1, pI + 1)] + [f'Xs{i}' for i in range(1, pS + 1)]
    return col_names


def load_dgp_scenario(scenario, d):
    confounder_indexes = [1, 2]
    predictor_indexes = [3, 4]
    exposure_indexes = [5, 6]
    nu = np.zeros(d)
    beta = np.zeros(d)
    if scenario == 1:
        beta[confounder_indexes] = 0.6
        beta[predictor_indexes] = 0.6
        nu[confounder_indexes] = 1
        nu[exposure_indexes] = 1
    elif scenario == 2:
        beta[confounder_indexes] = 0.6
        beta[predictor_indexes] = 0.6
        nu[confounder_indexes] = 0.4
        nu[exposure_indexes] = 1
    elif scenario == 3:
        beta[confounder_indexes] = 0.6
        beta[predictor_indexes] = 0.6
        nu[confounder_indexes] = 1
        nu[exposure_indexes] = 1.8
    else:
        assert(scenario == 4)
        beta[confounder_indexes] = 0.2
        beta[predictor_indexes] = 0.6
        nu[confounder_indexes] = 0.4
        nu[exposure_indexes] = 1
    return beta, nu


def generate_synthetic_dataset(n=1000, d=20, rho=0, eta=2, num_scenario=1):
    mean_x = 0
    var_x = 1
    cov_x = var_x * (np.eye(d) + ~np.eye(d, dtype=bool) * rho)  # covariance matrix of the Gaussian covariates.
    # Variance of each covariate is 1, correlation coefficient of every pair is rho
    X = np.random.multivariate_normal(mean=mean_x * np.ones(d), cov=cov_x, size=n)  # shape (n,d)
    # Normalize coviarates to have mean 0 and standard deviation 1
    scaler = StandardScaler(copy=False)
    scaler.fit_transform(X)

    beta, nu = load_dgp_scenario(num_scenario, d)
    A = np.random.binomial(np.ones(n, dtype=int), expit(np.dot(X, nu)))
    Y = np.random.randn(n) + eta * A + np.dot(X, beta)
    col_names = generate_col_names(d)
    df = pd.DataFrame(np.hstack([A.reshape(-1, 1), Y.reshape(-1, 1), X]), columns=col_names)
    return df


def calc_outcome_adaptive_lasso_single_lambda(df, Lambda, gamma_convergence_factor):
    n = df.shape[0]  # number of samples
    gamma = 2*(1 + gamma_convergence_factor - log(Lambda, n))
    XA = df.drop(columns=['Y'])
    X = XA.drop(columns=['A'])
    lr = LinearRegression(fit_intercept=True).fit(XA, df['Y'])
    xy_coefs = lr.coef_[1:]
    weights = (np.abs(xy_coefs)) ** (-1 * gamma)
    X_w = X / weights
    learner = LogisticRegression(solver='liblinear', penalty='l1', C=Lambda)
    ipw = IPW(learner, use_stabilized=True)
    ipw.fit(X_w, df['A'])
    weights = ipw.compute_weights(X_w, df['A'])
    outcomes = ipw.estimate_population_outcome(X_w, df['A'], df['Y'], w=weights)
    effect = ipw.estimate_effect(outcomes[1], outcomes[0])
    return effect, xy_coefs, weights


def calc_outcome_adaptive_lasso(df, lambdas, gamma_convergence_factor):
    x = df.drop(columns=['A', 'Y'])
    d = x.shape[1]
    if lambdas.shape[0] == 1:
        ate, _, _ = calc_outcome_adaptive_lasso_single_lambda(df, lambdas, gamma_convergence_factor)
    else:
        amd_vec = np.zeros(lambdas.shape[0])
        ate_vec = np.zeros(lambdas.shape[0])
        for il, Lambda in enumerate(lambdas):
            ate_vec[il], xy_coefs, ipw = calc_outcome_adaptive_lasso_single_lambda(df, lambdas[il], gamma_convergence_factor)
            amd = 0
            for j in range(d):
                xj = x.iloc[:, j]
                diff_d = np.average(xj[df['A'] == 1], weights=ipw[df['A'] == 1]) - \
                         np.average(xj[df['A'] == 0], weights=ipw[df['A'] == 0])
                amd += abs(xy_coefs[j]) * abs(diff_d)
            amd_vec[il] = amd

    return ate_vec[np.argmin(amd_vec)]
