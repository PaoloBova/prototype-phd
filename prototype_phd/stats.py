import pandas as pd
import statsmodels.api as sm
import numpy as np
import logging
import prototype_phd.data_utils as data_utils
import statsmodels.genmod.families.links as sm_links
import statsmodels.genmod.generalized_linear_model

statsmodels.genmod.generalized_linear_model.SET_USE_BIC_LLF(True)

def robust_logistic_fit(x, y, freq_weights=None, regularize=False, alpha=1.0,
                        L1_wt=0.0, link=sm_links.Logit()):
    """
    Fit a logistic regression model using the GLM framework (Binomial family).
    
    This function supports both binary data (each row is a Bernoulli trial) and
    proportion data (observations are averages over several trials). In the latter
    case, frequency weights should be provided.
    
    Parameters
    ----------
    x : array-like or pd.Series/DataFrame
        Predictor variable(s). For one predictor, use a 1D array/Series or a DataFrame
        with one column.
    y : array-like or pd.Series
        Outcome variable. For binary data, y should be 0 or 1. For proportion data,
        y is expected to lie in (0,1) unless frequency weights are provided. If no
        frequency weights are given and y is not binary, a warning is logged.
    freq_weights : array-like, optional
        Weights indicating the number of trials behind each observation. For example:
          - If each y is an average over a constant n, pass a scalar or an array of n.
          - If trial counts vary, pass the specific counts.
    regularize : bool, default False
        If True, uses fit_regularized() to apply a penalty (L1, L2, or elastic net) to
        the coefficients. Standard errors are not available for regularized fits.
    alpha : float, default 1.0
        Regularization strength; larger values impose more shrinkage.
    L1_wt : float, default 0.0
        Weight on the L1 penalty (1.0 for pure L1, 0.0 for pure L2, intermediate for
        elastic net).
    link : statsmodels link function, default Logit()
        The link function for the model. Alternatives include:
          - Probit() for models assuming a normally distributed latent
            variable.
          - CLogLog() for extreme value modeling.
    
    Data and Model Considerations
    ------------------------------
    - Binary Data: Each observation should be 0 or 1.
    - Proportion Data: y should lie in (0,1) unless freq_weights are provided.
      Without weights, proportions of 0 or 1 may lead to issues due to the logit.
    - The GLM assumes independent observations and the correct binomial mean-variance
      relationship.
    
    Post-Fit Diagnostics (see run_diagnostics function)
    -----------------------------------------------------
      * Examine model summary, deviance, and Pearson chi-square.
      * Plot residuals versus fitted values and check for patterns.
      * Evaluate leverage and influential observations.
    
    Returns
    -------
    result : GLMResults (or similar)
        The fitted model. Unregularized fits include standard errors and diagnostic stats.
    """
    # Input validation
    x = np.asarray(x) if not isinstance(x, (pd.DataFrame, pd.Series)) else x
    y = np.asarray(y) if not isinstance(y, (pd.Series, pd.DataFrame)) else y
    
    if len(x) != len(y):
        logging.error("Length mismatch: x and y must have the same number of observations.")
        raise ValueError("x and y must have the same number of observations.")
    
    # Warn if y is non-binary and no frequency weights are provided.
    unique_y = np.unique(y)
    if freq_weights is None and not np.all(np.isin(unique_y, [0, 1])):
        logging.warning("y appears to be proportion data but no frequency weights "
                        "were provided. Each observation will be treated as a single trial.")
    
    # Convert x to DataFrame if not already.
    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x, columns=[x.name] if hasattr(x, 'name') else ['x'])
    
    # Add constant term.
    X_with_const = sm.add_constant(x)
    
    # Define the GLM model.
    glm_model = sm.GLM(y, X_with_const,
                       family=sm.families.Binomial(link=link),
                       freq_weights=freq_weights)
    
    # Fit the model.
    if regularize:
        result = glm_model.fit_regularized(alpha=alpha, L1_wt=L1_wt)
    else:
        result = glm_model.fit()
    
    logging.info("Model fitting complete using link: %s", link.__class__.__name__)
    return result

def run_diagnostics(result):
    """
    Run a suite of diagnostic tests on a fitted GLM logistic regression model.
    
    Logs key statistics and measures:
      - AIC, BIC, deviance, and degrees of freedom.
      - Ratios of deviance and Pearson chi-square to residual degrees of freedom.
      - Maximum leverage and Cook's distance to identify influential points.
    
    Parameters
    ----------
    result : GLMResults or similar
        The fitted logistic regression model.
    
    Returns
    -------
    diagnostics : dict
        A dictionary containing diagnostic metrics.
    """
    diagnostics = {}
    diagnostics['AIC'] = result.aic
    diagnostics['BIC'] = result.bic
    diagnostics['Deviance'] = result.deviance
    diagnostics['DF_Resid'] = result.df_resid
    
    dev_ratio = result.deviance / result.df_resid if result.df_resid else np.nan
    diagnostics['Deviance/DF'] = dev_ratio
    
    if hasattr(result, 'pearson_chi2'):
        pearson_ratio = result.pearson_chi2 / result.df_resid if result.df_resid else np.nan
        diagnostics['Pearson_Chi2'] = result.pearson_chi2
        diagnostics['Pearson/DF'] = pearson_ratio
    else:
        diagnostics['Pearson_Chi2'] = np.nan
        diagnostics['Pearson/DF'] = np.nan

    # Log diagnostic metrics.
    logging.info("AIC: %.3f, BIC: %.3f", diagnostics['AIC'], diagnostics['BIC'])
    logging.info("Deviance: %.3f with DF: %d (Ratio: %.3f)",
                 diagnostics['Deviance'], int(diagnostics['DF_Resid']), dev_ratio)
    if not np.isnan(diagnostics['Pearson/DF']):
        logging.info("Pearson Chi-Square/DF: %.3f", diagnostics['Pearson/DF'])
    
    # Influence measures.
    try:
        influence = result.get_influence()
        # Hat values (leverage)
        leverage = influence.hat_matrix_diag
        diagnostics['Max_Leverage'] = np.max(leverage)
        # Cook's distances.
        cooks_d = influence.cooks_distance[0]
        diagnostics['Max_Cooks_D'] = np.max(cooks_d)
        logging.info("Max Leverage: %.3f, Max Cook's D: %.3f",
                     diagnostics['Max_Leverage'], diagnostics['Max_Cooks_D'])
    except Exception as e:
        logging.warning("Could not compute influence measures: %s", e)
        diagnostics['Max_Leverage'] = np.nan
        diagnostics['Max_Cooks_D'] = np.nan
    
    # Recommendations based on diagnostics.
    if dev_ratio > 1.5:
        logging.warning("Deviance/DF ratio (%.2f) is high; check for overdispersion.",
                        dev_ratio)
    if diagnostics.get('Pearson/DF', 1) > 1.5:
        logging.warning("Pearson Chi-Square/DF ratio is high; overdispersion may be present.")
    
    logging.info("Diagnostics complete.")
    return diagnostics

def compare_link_functions(x, y, freq_weights=None, regularize=False, alpha=1.0,
                           L1_wt=0.0):
    """
    Compare different link functions (logit, probit, cloglog) by fitting models and
    reporting AIC and BIC. Lower values indicate a better fit.
    
    In the probit model, it is assumed that there is an underlying latent variable that
    follows a standard normal distribution. The observed binary outcome is 1 if this
    latent variable exceeds a threshold (typically 0) and 0 otherwise.
    
    Parameters
    ----------
    x : array-like or pd.Series/DataFrame
        Predictor variable(s).
    y : array-like or pd.Series
        Outcome variable.
    freq_weights : array-like, optional
        Frequency weights for each observation.
    regularize : bool, default False
        Whether to use regularization.
    alpha : float, default 1.0
        Regularization strength.
    L1_wt : float, default 0.0
        Weight for the L1 penalty.
    
    Returns
    -------
    metrics : dict
        Dictionary with keys for each link ('logit', 'probit', 'cloglog') and values
        containing a tuple (AIC, BIC).
    best_link : str
        The link function with the lowest AIC.
    """
    links = {
        'logit': sm_links.Logit(),
        'probit': sm_links.Probit(),
        'cloglog': sm_links.CLogLog()
    }
    
    metrics = {}
    for name, link in links.items():
        logging.info("Fitting model using %s link.", name)
        result = robust_logistic_fit(x, y, freq_weights=freq_weights,
                                     regularize=regularize, alpha=alpha,
                                     L1_wt=L1_wt, link=link)
        metrics[name] = (result.aic, result.bic)
        logging.info("%s link: AIC = %.3f, BIC = %.3f",
                     name, result.aic, result.bic)
    
    # Choose the best link based on AIC.
    best_link = min(metrics, key=lambda k: metrics[k][0])
    logging.info("Best link based on AIC: %s", best_link)
    return metrics, best_link

def compute_threshold(result):
    """
    Compute the 50% probability threshold for a logistic model with one predictor.
    
    For a model of the form:
        log(p/(1-p)) = beta0 + beta1 * x,
    the 50% threshold is given by:
        x = -beta0 / beta1
    
    Parameters
    ----------
    result : GLMResults or similar
        The fitted model with parameter estimates.
    
    Returns
    -------
    threshold : float or None
        The 50% threshold if one predictor exists; otherwise, None.
    """
    if len(result.params) == 2:
        intercept, slope = result.params.iloc[0], result.params.iloc[1]
        return -intercept / slope if slope != 0 else np.nan
    else:
        return None
