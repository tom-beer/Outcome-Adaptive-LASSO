import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from causallib.estimation import IPW
from math import log


def calc_ate_vanilla_ipw(A, Y, X):
    ipw = IPW(LogisticRegression(solver='liblinear', penalty='l1', C=1e2, max_iter=500), use_stabilized=True).fit(X, A)
    weights = ipw.compute_weights(X, A)
    outcomes = ipw.estimate_population_outcome(X, A, Y, w=weights)
    effect = ipw.estimate_effect(outcomes[1], outcomes[0])
    return effect[0]


def calc_group_diff(X, idx_trt, ipw, l_norm):
    """Utility function to calculate the difference in covariates between treatment and control groups"""
    return (np.abs(np.average(X[idx_trt], weights=ipw[idx_trt], axis=0) -
                   np.average(X[~idx_trt], weights=ipw[~idx_trt], axis=0)))**l_norm


def calc_wamd(A, X, ipw, x_coefs, l_norm=1):
    """Utility function to calculate the weighted absolute mean difference"""
    idx_trt = A == 1
    return calc_group_diff(X.values, idx_trt.values, ipw.values, l_norm).dot(np.abs(x_coefs))


def calc_outcome_adaptive_lasso_single_lambda(A, Y, X, Lambda, gamma_convergence_factor):
    """Calculate ATE with the outcome adaptive lasso"""
    n = A.shape[0]  # number of samples
    # extract gamma according to Lambda and gamma_convergence_factor
    gamma = 2 * (1 + gamma_convergence_factor - log(Lambda, n))
    # fit regression from covariates X and exposure A to outcome Y
    lr = LinearRegression(fit_intercept=True).fit(np.hstack([A.values.reshape(-1, 1), X]), Y)
    # extract the coefficients of the covariates
    x_coefs = lr.coef_[1:]
    # calculate outcome adaptive penalization weights
    weights = (np.abs(x_coefs)) ** (-1 * gamma)
    # apply the penalization to the covariates themselves
    X_w = X / weights
    # fit logistic propensity score model from penalized covariates to the exposure
    ipw = IPW(LogisticRegression(solver='liblinear', penalty='l1', C=1/Lambda), use_stabilized=False).fit(X_w, A)
    # compute inverse propensity weighting and calculate ATE
    weights = ipw.compute_weights(X_w, A)
    outcomes = ipw.estimate_population_outcome(X_w, A, Y, w=weights)
    effect = ipw.estimate_effect(outcomes[1], outcomes[0])
    return effect, x_coefs, weights


def calc_outcome_adaptive_lasso(A, Y, X, gamma_convergence_factor=2, log_lambdas=None):
    """Calculate estimate of average treatment effect using the outcome adaptive LASSO (Shortreed and Ertefaie, 2017)
    Parameters
    ----------
    A : Dataset for which ATE will be calculated
         The dataframe must have one column named A, one column named Y and the rest are covariates (arbitrarily named)
    Y :
    X :
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
    n = A.shape[0]
    lambdas = n ** np.array(log_lambdas)
    amd_vec = np.zeros(lambdas.shape[0])
    ate_vec = np.zeros(lambdas.shape[0])

    # Calculate ATE for each lambda, select the one minimizing the weighted absolute mean difference
    for il in range(len(lambdas)):
        ate_vec[il], x_coefs, ipw = \
            calc_outcome_adaptive_lasso_single_lambda(A, Y, X, lambdas[il], gamma_convergence_factor)
        amd_vec[il] = calc_wamd(A, X, ipw, x_coefs)

    ate = ate_vec[np.argmin(amd_vec)]
    return ate
