import logging
import numpy as np
import pandas as pd
import prototype_phd.utils as utils
from pydantic import BaseModel, Field
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import statsmodels.api as sm
import statsmodels.genmod.families.links as sm_links
import statsmodels.genmod.generalized_linear_model
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
from typing import Tuple, Optional, List
import warnings

statsmodels.genmod.generalized_linear_model.SET_USE_BIC_LLF(True)

def bin_proportion_data(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    n_bins: int
) -> pd.DataFrame:
    """
    Aggregate raw (x, y) trials into n_bins bins and compute per‐bin proportions.

    Parameters
    ----------
    df : DataFrame
        Must contain columns x_col (numeric predictor) and y_col (binary 0/1 outcome).
    x_col : str
        Name of predictor column.
    y_col : str
        Name of binary outcome column.
    n_bins : int
        Number of equal‐width bins over x.

    Returns
    -------
    binned_df : DataFrame
        Columns:
          - x: bin center (mean x in bin)
          - y: proportion of successes (sum(y)/count)
          - freq_weights: count of trials in bin
    """
    edges = np.linspace(df[x_col].min(), df[x_col].max(), n_bins + 1)
    # assign each row to a bin
    df2 = df[[x_col, y_col]].copy()
    df2['bin'] = pd.cut(df2[x_col], edges, include_lowest=True)
    
    records = []
    for _interval, grp in df2.groupby('bin'):
        if grp.empty:
            continue
        x_mean = grp[x_col].mean()
        total = len(grp)
        successes = grp[y_col].sum()
        records.append({
            'x': x_mean,
            'y': successes / total,
            'freq_weights': total
        })
    
    df_avg =  pd.DataFrame.from_records(records)
    x_binned = df_avg['x'].values
    y_binned = df_avg['y'].values
    
    idx = np.digitize(x_binned, edges) - 1

    x_rows, y_rows, w = [], [], []
    for i in range(n_bins):
        mask = idx == i
        if not mask.any():
            continue
        x_c = x_binned[mask].mean()
        succ = y_binned[mask].sum()
        total = mask.sum()
        # one row for successes, one for failures
        x_rows += [x_c, x_c]
        y_rows += [1, 0]
        w += [succ, total - succ]

    X = np.array(x_rows).reshape(-1, 1)
    y = np.array(y_rows)
    sw = np.array(w)
    # Convert to DataFrame
    
    return pd.DataFrame({
        'x': X.flatten(),
        'y': y,
        'freq_weights': sw
    })

def bin_continuous_data(
    df: pd.DataFrame, x_col: str, y_col: str, n_bins: int
) -> pd.DataFrame:
    """
    Bin raw (x, y={0,1}) into n_bins equal-width groups.
    Returns one row per bin with columns:
      - x       : mean x in bin
      - y       : proportion successes
      - freq_weights: number of trials in bin
    """
    edges = np.linspace(df[x_col].min(), df[x_col].max(), n_bins + 1)
    df2 = df[[x_col, y_col]].copy()
    df2['bin'] = pd.cut(df2[x_col], edges, include_lowest=True)
    records = []
    for _, grp in df2.groupby('bin'):
        if grp.empty: continue
        x_mean = grp[x_col].mean()
        total = len(grp)
        succ = grp[y_col].sum()
        records.append({'x': x_mean, 'y': succ/total, 'freq_weights': total})
    return pd.DataFrame.from_records(records)

def expand_binned_proportions(
    df_binned: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y',
    weight_col: str = 'freq_weights'
) -> pd.DataFrame:
    """
    Expand each proportion row into two binary rows (y=1 and y=0)
    with sample_weight = [#successes, #failures].
    """
    x_arr = df_binned[x_col].values
    y_arr = df_binned[y_col].values
    freq = df_binned[weight_col].astype(int).values
    succ = np.round(y_arr * freq).astype(int)
    fail = freq - succ
    X_list, y_list, sw_list = [], [], []
    for xi, s, f in zip(x_arr, succ, fail):
        X_list += [xi, xi]
        y_list += [1, 0]
        sw_list += [s, f]
    return pd.DataFrame({'x': X_list, 'y': y_list, 'sample_weight': sw_list})

def prepare_binned_data(
    df: pd.DataFrame, x_col: str, y_col: str, n_bins: int
) -> pd.DataFrame:
    """
    Combine bin_continuous_data + expand_binned_proportions into one step.
    """
    df_b = bin_continuous_data(df, x_col, y_col, n_bins)
    return expand_binned_proportions(df_b, x_col='x', y_col='y', weight_col='freq_weights')

class LogRegConfig(BaseModel):
    engine: str = Field("statsmodels", description="Engine: 'statsmodels' or 'scikit-learn'.")
    C: float = Field(1.0, description="Inverse of regularization strength (for scikit-learn).")
    solver: str = Field("lbfgs", description="Solver to use (scikit-learn).")
    max_iter: int = Field(1000, description="Maximum iterations for convergence (scikit-learn).")
    random_state: int = Field(42, description="Random seed (scikit-learn).")
    sample_weight: Optional[np.ndarray] = Field(None, description="Sample weights (scikit-learn).")
    standardize_data: bool = Field(
        False,
        description="Whether to standardize the data before fitting. \
            Standardization is done using StandardScaler from scikit-learn. \
            If True, the data is standardized before fitting the model. \
            If False, the data is used as is. \
            Note: Standardization is not applied when using statsmodels.")
    freq_weights: Optional[np.ndarray] = Field(
        None,
        description="Frequency weights indicating the number of trials behind each observation. For example:"
        "- If each y is an average over a constant n, pass a scalar or an array of n."
        "- If trial counts vary, pass the specific counts."
        " (statsmodels)"
        )
    regularize: bool = Field(
        False,
        description="Whether to use regularization (statsmodels). If True, uses \
            fit_regularized() to apply a penalty (L1, L2, or elastic net) to the \
            coefficients. Standard errors are not available for regularized fits.")
    alpha: float=Field(1.0, description="Regularization strength, larger values impose more shrinkage (statsmodels).")
    L1_wt: float=Field(0.0, description="Weight on the L1 penalty (1.0 for pure L1, 0.0 for pure L2, intermediate for elastic net). (statsmodels).")
    link: sm_links.Link=Field(
        sm_links.Logit(),
        description="The link function for the model. Alternatives include: \
          - Probit() for models assuming a normally distributed latent variable. \
          - CLogLog() for extreme value modeling. (statsmodels)."
        )
    
    class Config:
        arbitrary_types_allowed = True
    


class LogRegResult(BaseModel):
    coeffs: np.ndarray = Field(..., description="Estimated coefficients of the model.")
    convergence: bool = Field(True, description="Whether the model fitting converged.")
    warning: Optional[str] = Field(None, description="Warning message if fitting did not converge.")
    glm_result: Optional[GLMResultsWrapper] = Field(None, description="Fitted GLM result (statsmodels).")
    scaler: Optional[StandardScaler] = Field(None, description="Scaler used for standardization (scikit-learn).")
    sk_result: Optional[LogisticRegression] = Field(None, description="Fitted logistic regression model (scikit-learn).")

    class Config:
        arbitrary_types_allowed = True
    

@utils.multi
def fit_logistic(_x:np.ndarray, _y:np.ndarray, config:LogRegConfig) -> str:
    return config.engine

@utils.method(fit_logistic, "statsmodels")
def fit_logistic(x, y, config:LogRegConfig=LogRegConfig()) ->LogRegResult:
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
    config : LogRegConfig
        Configuration parameters for the logistic regression model.
    
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
    LogRegResult
        The fitted model. Unregularized fits include standard errors and diagnostic stats.
    """
    regularize = config.regularize
    alpha = config.alpha
    L1_wt = config.L1_wt
    link = config.link
    freq_weights = config.freq_weights
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

    # Convert y to Series if not already.
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name=y.name if hasattr(y, 'name') else 'y')

    # Handle cases where y is constant (all 0s or all 1s)
    if len(unique_y) == 1:
        outcome = y.iloc[0]
        metadata = {"converged": True, "warning": "all_success" if outcome == 1 else "all_failure"}
        # Set intercept to -1
        params = np.zeros(x.shape[1] + 1)
        params[0] = -1
        for j, col in enumerate(x.columns, start=1):
            default_threshold_j = x[col].max() if outcome == 1 else x[col].min()
            # Set coefficients such that -1 * intercept / coefficient = default_threshold_j
            params[j] = 1 / default_threshold_j
        return LogRegResult(coeffs=params, **metadata)

    # Typically, other issues (e.g. (quasi-)perfect separation, 
    # insufficient variation) are handled using regularization.
    
    # Add constant term.
    X_with_const = sm.add_constant(x)
    
    # Define the GLM model.
    glm_model = sm.GLM(y, X_with_const,
                       family=sm.families.Binomial(link=link),
                       freq_weights=freq_weights)
    
    # Fit the model.
    try:
        if regularize:
            raise ValueError("""Regularization with statsmodels GLM is disabled because
                             it fails to behave as expected. Use scikit-learn instead.""")
        else:
            result = glm_model.fit()
        logging.info("Model fitting complete using link: %s", link.__class__.__name__)
        metadata = {"converged": True, "warning": ""}
    except Exception as e:
        result = None
        logging.error("Model fitting failed: %s", e)
        metadata = {"converged": False, "warning": str(e)}
    
    coeffs = result.params if result else np.full(x.shape[1] + 1, np.nan)
    coeffs = np.array(coeffs)
    return LogRegResult(
        coeffs=coeffs,
        **metadata,
        glm_result=result
    )

@utils.method(fit_logistic, "scikit-learn")
def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    config: LogRegConfig
) -> LogRegResult:
    """
    Fit a logistic regression model on a single predictor with standardization.
    
    Parameters
    ----------
    X : np.ndarray
        Predictor array with shape (n_samples,) or (n_samples, 1).
    y : np.ndarray
        Binary outcome array.
    config : LogRegConfig
        Configuration parameters for logistic regression.
    
    Returns
    -------
    LogRegResult
        Includes the fitted logistic regression model and the scaler used.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X.shape[0] != len(y):
        logging.error("Length mismatch: x and y must have the same number of observations.")
        raise ValueError("x and y must have the same number of observations.")
    
    # Warn if y is non-binary
    unique_y = np.unique(y)
    if not np.all(np.isin(unique_y, [0, 1])):
        logging.warning("y appears to be proportion data. Scikit-learn's LogisticRegression"
                        "does not support proportion data. Expand y into binary rows with sample weights.")

    # Handle cases where y is constant (all 0s or all 1s)
    if len(unique_y) == 1:
        outcome = y[0]
        metadata = {"converged": True, "warning": "all_success" if outcome == 1 else "all_failure"}
        # Set intercept to -1
        params = np.zeros(X.shape[1] + 1)
        params[0] = -1
        for j, x in enumerate(X.T, start=1):
            default_threshold_j = np.max(x) if outcome == 1 else np.min(x)
            # Set coefficients such that -1 * intercept / coefficient = default_threshold_j
            params[j] = 1 / default_threshold_j
        return LogRegResult(coeffs=params, **metadata)

    # Typically, other issues (e.g. (quasi-)perfect separation, 
    # insufficient variation) are handled using regularization.
    sample_weight = config.sample_weight
    if config.standardize_data:
        logging.warning("Standardizing data before fitting.")
        logging.warning("Some algorithms appear to perform poorly with standardized data."
                        "It is not known if this is due to theoretical reasons or due to our implementation."
                        "I advise against standardizing data unless you have a good reason to do so."
                        "The Saga algorithm appears to work poorly with standardized data."
                        "LBFGS appears to work fine with standardized data.")
        scaler = StandardScaler()
        # Fit scaler using sample_weight if available to avoid bias from uneven bin counts
        scaler.fit(X, sample_weight=sample_weight)
        X_scaled = scaler.transform(X)
    else:
        scaler = None
        X_scaled = X
    
    clf = LogisticRegression(
        C=config.C, 
        solver=config.solver, 
        max_iter=config.max_iter, 
        random_state=config.random_state,
    )
    conv_flag = True
    warning = ""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(X_scaled, y, sample_weight=sample_weight)
        coeffs = np.concatenate(([clf.intercept_[0]], *clf.coef_))
        for warn in w:
            if issubclass(warn.category, ConvergenceWarning):
                conv_flag = False
                logging.warning("Convergence warning in logistic regression fit.")
                warning += f"Convergence warning: {warn.message}"
    return LogRegResult(coeffs=coeffs,
                        scaler=scaler,
                        sk_result=clf,
                        convergence=conv_flag,
                        warning=warning)

@utils.method(fit_logistic)
def fit_logistic(_x: np.ndarray, _y: np.ndarray, config: LogRegConfig) -> None:
    raise ValueError(f"Unsupported engine: {config.engine}")

def run_diagnostics(result:GLMResultsWrapper) -> dict:
    """
    Run a suite of diagnostic tests on a fitted GLM logistic regression model.
    
    Logs key statistics and measures:
      - AIC, BIC, deviance, and degrees of freedom.
      - Ratios of deviance and Pearson chi-square to residual degrees of freedom.
      - Maximum leverage and Cook's distance to identify influential points.
    
    Parameters
    ----------
    result : GLMResultsWrapper or similar
        The fitted logistic regression model.
    
    Returns
    -------
    diagnostics : dict
        A dictionary containing diagnostic metrics.
        
    Notes
    -----
    - Only suppports statsmodels GLMResultsWrapper objects.
    """
    result = result.glm_result
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

def compare_link_functions(x, y, config:LogRegConfig=LogRegConfig()) -> Tuple[dict, str]:
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
    config : LogRegConfig
        Configuration parameters for the logistic regression model.
    
    Returns
    -------
    metrics : dict
        Dictionary with keys for each link ('logit', 'probit', 'cloglog') and values
        containing a tuple (AIC, BIC).
    best_link : str
        The link function with the lowest AIC.
    """
    assert config.engine.lower() == "statsmodels", "Only statsmodels engine is supported for link comparison."
    links = {
        'logit': sm_links.Logit(),
        'probit': sm_links.Probit(),
        'cloglog': sm_links.CLogLog()
    }
    
    metrics = {}
    for name, link in links.items():
        logging.info("Fitting model using %s link.", name)
        config_args = {**config.model_dump(), 'link': link}
        result = fit_logistic(x, y, config=LogRegConfig(**config_args))
        glm_result = result.glm_result
        metrics[name] = (glm_result.aic, glm_result.bic)
        logging.info("%s link: AIC = %.3f, BIC = %.3f",
                     name, glm_result.aic, glm_result.bic)
    
    # Choose the best link based on AIC.
    best_link = min(metrics, key=lambda k: metrics[k][0])
    logging.info("Best link based on AIC: %s", best_link)
    return metrics, best_link

def compute_threshold_from_result(result:LogRegResult, p:float=0.5) -> Optional[float]:
    """
    Extract the 100*p% probability threshold from a fitted logistic regression model.
    
    For a model of the form:
        log(p/(1-p)) = β₀ + β₁ x,
    the threshold (where p = 0.5) is given by x* = -β₀ / β₁,
    or in the general case x* = (log(p/(1-p)) - β₀) / β₁
    
    If a scaler is provied, transform threshold back to the original scale.
    
    Parameters
    ----------
    result : LogRegResult
        The fitted logistic regression model.
    p : float, default 0.5
        The probability threshold to compute the threshold for.

    Returns
    -------
    Optional[float]
        The threshold on the original scale, or None if beta1 is near zero.
    """
    beta0, beta1 = result.coeffs
    scaler = result.scaler
    if np.abs(beta1) < 1e-8:
        logging.error("Coefficient too close to zero. Cannot compute threshold.")
        return np.nan
    log_odds = np.log(p / (1 - p))
    threshold_std = (log_odds - beta0) / beta1
    if scaler:
        # Retrieve the scaler parameters
        mu = scaler.mean_[0]    # mean of x
        sigma = scaler.scale_[0]  # standard deviation of x
        intercept_unscaled = beta0 - (beta1/ sigma) * mu
        coef_unscaled = beta1 / sigma
        original_threshold = -intercept_unscaled / coef_unscaled
        # Or more simply:
        # original_threshold = threshold_std * scaler.scale_[0] + scaler.mean_[0]
    else:
        original_threshold = threshold_std
    return original_threshold

def compute_threshold(
    X: np.ndarray,
    y: np.ndarray,
    config: LogRegConfig,
    p: float = 0.5
) -> Optional[float]:
    """
    Convenience function: fit logistic regression on X and return the threshold.
    Returns
    -------
    Optional[float]
        The computed threshold value, or None if computation fails.
    """
    model_fit = fit_logistic(X, y, config)
    return compute_threshold_from_result(model_fit, p=p)
