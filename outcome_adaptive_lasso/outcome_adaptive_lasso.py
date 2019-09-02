import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from causallib.estimation import IPW
from math import log


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


def calc_outcome_adaptive_lasso_single_lambda(df, Lambda, gamma_convergence_factor):
    """Calculate ATE with the outcome adaptive lasso"""
    n = df.shape[0]  # number of samples
    # extract gamma according to Lambda and gamma_convergence_factor
    gamma = 2 * (1 + gamma_convergence_factor - log(Lambda, n))
    XA = df.drop(columns=['Y'])
    X = XA.drop(columns=['A'])
    # fit regression from covariates X and exposure A to outcome Y
    lr = LinearRegression(fit_intercept=True).fit(XA, df['Y'])
    # extract the coefficients of the covariates
    xy_coefs = lr.coef_[1:]
    # calculate outcome adaptive penalization weights
    weights = (np.abs(xy_coefs)) ** (-1 * gamma)
    # apply the penalization to the covariates themselves
    X_w = X / weights
    # fit logistic propensity score model from penalized covariates to the exposure
    ipw = IPW(LogisticRegression(solver='liblinear', penalty='l1', C=Lambda), use_stabilized=False).fit(X_w, df['A'])
    # compute inverse propensity weighting and calculate ATE
    weights = ipw.compute_weights(X_w, df['A'])
    outcomes = ipw.estimate_population_outcome(X_w, df['A'], df['Y'], w=weights)
    effect = ipw.estimate_effect(outcomes[1], outcomes[0])
    return effect, xy_coefs, weights


def calc_group_diff(x_df, idx_trt, ipw):
    """Utility function to calculate the difference in covariates between treatment and control groups"""
    return (np.abs(np.average(x_df[idx_trt], weights=ipw[idx_trt], axis=0)) -
            np.average(x_df[~idx_trt], weights=ipw[~idx_trt], axis=0))


def calc_wamd(df, ipw, xy_coefs):
    """Utility function to calculate the weighted absolute mean difference"""
    x_df = df.drop(columns=['A', 'Y'])
    idx_trt = df['A'] == 1
    return calc_group_diff(x_df.values, idx_trt.values, ipw.values).dot(np.abs(xy_coefs))


def calc_outcome_adaptive_lasso(df, gamma_convergence_factor=2,
                                log_lambdas=None):
    """Calculate estimate of average treatment effect using the outcome adaptive LASSO (Shortreed and Ertefaie, 2017)

    Parameters
    ----------
    df : Dataset for which ATE will be calculated
         The dataframe must have one column named A, one column named Y and the rest are covariates (arbitrarily named)

    log_lambdas : log of lambda - strength of adaptive LASSO regularization.
        If log_lambdas has multiple values, lambda will be selected according to the minimal absolute mean difference,
        as suggested in the paper
        If None, it will be set to the suggested search list in the paper:
        [-10, -5, -2, -1, -0.75, -0.5, -0.25, 0.25, 0.49]


    gamma_convergence_factor : a constant to couple between lambda and gamma, the single-feature penalization strength
        The equation relating gamma and lambda is lambda * n^(gamma/2 -1) = n^gamma_convergence_factor
        Default value is 2, as suggested in the paper for the synthetic dataset experiments


    Returns
    -------
    ate : estimate of the average treatment effect
    """
    if log_lambdas is None:
        log_lambdas = [-10, -5, -2, -1, -0.75, -0.5, -0.25, 0.25, 0.49]
    n = df.shape[0]
    lambdas = n ** np.array(log_lambdas)
    amd_vec = np.zeros(lambdas.shape[0])
    ate_vec = np.zeros(lambdas.shape[0])

    # Calculate ATE for each lambda, select the one minimizing the weighted absolute mean difference
    for il in range(len(lambdas)):
        ate_vec[il], xy_coefs, ipw = calc_outcome_adaptive_lasso_single_lambda(df, lambdas[il], gamma_convergence_factor)
        amd_vec[il] = calc_wamd(df, ipw, xy_coefs)

    ate = ate_vec[np.argmin(amd_vec)]

    return ate
