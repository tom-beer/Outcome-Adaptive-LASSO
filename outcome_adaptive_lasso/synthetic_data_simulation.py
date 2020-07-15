import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.preprocessing import StandardScaler


def generate_col_names(d):
    """Utility function to generate column names for the synthetic dataset """
    assert (d >= 6)
    pC = 2  # number of confounders
    pP = 2  # number of outcome predictors
    pI = 2  # number of exposure predictors
    pS = d - (pC + pI + pP)  # number of spurious covariates
    col_names = ['A', 'Y'] + [f'Xc{i}' for i in range(1, pC + 1)] + [f'Xp{i}' for i in range(1, pP + 1)] + \
                [f'Xi{i}' for i in range(1, pI + 1)] + [f'Xs{i}' for i in range(1, pS + 1)]
    return col_names


def load_dgp_scenario(scenario, d):
    """Utility function to load predefined scenarios"""
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
        beta[confounder_indexes] = 0.2
        beta[predictor_indexes] = 0.6
        nu[confounder_indexes] = 0.4
        nu[exposure_indexes] = 1
    else:
        assert (scenario == 4)
        beta[confounder_indexes] = 0.6
        beta[predictor_indexes] = 0.6
        nu[confounder_indexes] = 1
        nu[exposure_indexes] = 1.8
    return beta, nu


def generate_synthetic_dataset(n=1000, d=100, rho=0, eta=0, scenario_num=1):
    """Generate a simulated dataset according to the settings described in section 4.1 of the paper
    Covariates X are zero mean unit variance Gaussians with correlation rho
    Exposure A is logistic in X: logit(P(A=1)) = nu.T*X (nu is set according to scenario_num)
    Outcome Y is linear in A and X: Y =  eta*A + beta.T*X + N(0,1)
    Parameters
    ----------
    n : number of samples in the dataset
    d : total number of covariates. Of the d covariates, d-6 are spurious,
        i.e. they do not influence the exposure or the outcome
    rho : correlation between pairwise Gaussian covariates
    eta : True treatment effect
    scenario_num : one of {1-4}. Each scenario differs in the vectors nu and beta.
        According to the supplementary material of the paper, the four scenarios are:
        1) beta = [0.6, 0.6, 0.6, 0.6, 0, ..., 0] and nu = [1, 1, 0, 0, 1, 1, 0, ..., 0]
        2) beta = [0.6, 0.6, 0.6, 0.6, 0, ..., 0] and nu = [0.4, 0.4, 0, 0, 1, 1, 0, ..., 0]
        3) beta = [0.2, 0.2, 0.6, 0.6, 0, ..., 0] and nu = [0.4, 0.4, 0, 0, 1, 1, 0, ..., 0]
        4) beta = [0.6, 0.6, 0.6, 0.6, 0, ..., 0] and nu = [1, 1, 0, 0, 1.8, 1.8, 0, ..., 0]
    Returns
    -------
    df : DataFrame of n rows and d+2 columns: A, Y and d covariates.
         Covariates are named Xc if they are confounders, Xi if they are instrumental variables,
         Xp if they are predictors of outcome and Xs if they are spurious
    TODO:
     * Enable manual selection of nu and beta
    """
    cov_x = np.eye(d) + ~np.eye(d, dtype=bool) * rho  # covariance matrix of the Gaussian covariates.
    # Variance of each covariate is 1, correlation coefficient of every pair is rho
    X = np.random.multivariate_normal(mean=0 * np.ones(d), cov=cov_x, size=n)  # shape (n,d)
    # Normalize covariates to have 0 mean unit std
    scaler = StandardScaler(copy=False)
    scaler.fit_transform(X)

    # Load beta and nu from the predefined scenarios
    beta, nu = load_dgp_scenario(scenario_num, d)
    A = np.random.binomial(np.ones(n, dtype=int), expit(np.dot(X, nu)))
    Y = np.random.randn(n) + eta * A + np.dot(X, beta)
    col_names = generate_col_names(d)
    df = pd.DataFrame(np.hstack([A.reshape(-1, 1), Y.reshape(-1, 1), X]), columns=col_names)
    return df
