"""
Stage-two estimation of the causal parameter from cross-fitted residuals.

Given out-of-sample residuals :math:`\\hat\\chi_t = y_t - \\hat g^r(X_t)` and
:math:`\\hat\\xi_t = d_t - \\hat m(X_t)`, the partially linear model reduces to
the residual-on-residual regression :math:`\\hat\\chi_t = \\theta \\hat\\xi_t +
\\epsilon_t`.  Two estimators are provided:

``'block'`` (paper default, eq. 2.5)
    Fit :math:`\\hat\\theta_k` on each main block :math:`B_k` and average:
    :math:`\\hat\\theta = K^{-1}\\sum_k \\hat\\theta_k`.  The standard error is
    **not** an average of fold standard errors -- it is built from the stacked,
    time-ordered score sequence

    .. math::

        s_t = \\hat\\xi_t e_t, \\qquad
        e_t = \\hat\\epsilon_{t,k} + (\\hat\\theta_k - \\hat\\theta)\\hat\\xi_t,

    so that dependence across adjacent main-block boundaries is picked up:
    :math:`\\widehat{\\mathrm{Var}}(\\hat\\theta) = \\hat\\Sigma / (J^2 T)` with
    :math:`J = \\mathbb E[\\hat\\xi_t^2]`.

``'full'``
    Pool every residual and run one HAC regression.

Both paths reproduce the paper's replication code exactly.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import scipy.stats as stats
import statsmodels.api as sm

from .hac import (
    calculate_optimal_lag,
    compute_hac_bandwidth,
    get_critical_value,
    hac_lrv,
    normalize_kernel,
)

__all__ = ["fit_stage_two", "fit_ols_with_hac", "block_statistics", "full_statistics"]


# --------------------------------------------------------------------------- #
# OLS with HAC
# --------------------------------------------------------------------------- #

def fit_ols_with_hac(model, kernel: str = "bartlett", bandwidth: Optional[int] = None,
                     horizon: int = 0):
    """
    Fit a ``statsmodels`` OLS model with a HAC covariance matrix.

    Bartlett uses the fast ``statsmodels`` path; QS and Parzen are applied as a
    manual sandwich built from :func:`tsdml.hac.hac_lrv`.
    """
    k = normalize_kernel(kernel)
    if bandwidth is None:
        bandwidth = calculate_optimal_lag(horizon)
    bw = max(1, int(bandwidth))

    if k == "bartlett":
        return model.fit(cov_type="HAC", cov_kwds={"maxlags": bw})

    results = model.fit()
    X = np.asarray(model.exog, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    score = X * np.asarray(results.resid, dtype=float).reshape(-1, 1)
    Sigma = hac_lrv(score, K=0, kernel=k, bandwidth=bw)
    if np.ndim(Sigma) == 0:
        Sigma = np.array([[float(Sigma)]])
    T = X.shape[0]
    XtX_inv = np.linalg.inv(X.T @ X)
    results.cov_params_default = XtX_inv @ (T * Sigma) @ XtX_inv
    return results


# --------------------------------------------------------------------------- #
# Stage two
# --------------------------------------------------------------------------- #

def fit_stage_two(
    outcome_residuals: np.ndarray,
    treatment_residuals: np.ndarray,
    estimation_method: str = "block",
    n_blocks: int = 5,
    main_blocks: Optional[List[List[int]]] = None,
    include_constant: bool = True,
    use_hac: bool = True,
    horizon: int = 0,
    confidence_level: float = 0.95,
    use_hac_final_se: bool = True,
    hac_lag: Optional[int] = None,
    hac_kernel: str = "bartlett",
    hac_bandwidth_rule: str = "small",
    hac_bandwidth_value: Optional[int] = None,
    use_fixed_b_critical: bool = False,
) -> Dict[str, object]:
    """
    Estimate :math:`\\theta` from cross-fitted residuals.

    Parameters
    ----------
    outcome_residuals, treatment_residuals : ndarray, shape (T,)
        Out-of-sample residuals in time order.
    estimation_method : {'block', 'full'}
        Fold-average (paper default) or pooled regression.
    n_blocks : int
        Number of folds; used only when ``main_blocks`` is ``None``.
    main_blocks : list of list of int, optional
        Main-block indices from the fold construction.
    include_constant : bool
        Include an intercept in the residual-on-residual regressions.
    use_hac : bool
        Use HAC inference.
    horizon : int
        Local-projection horizon, feeding the ``'small'`` bandwidth rule.
    confidence_level : float
        Confidence level for the reported interval.
    use_hac_final_se : bool
        If ``True`` (default) the block-average standard error comes from the
        stacked score HAC formula; if ``False`` fold standard errors are simply
        averaged, which understates dependence across block boundaries.
    hac_lag : int, optional
        Explicit truncation lag for the final HAC variance, bypassing the rule.
    hac_kernel : {'bartlett', 'qs', 'parzen', 'ewc'}
    hac_bandwidth_rule : str
        See :func:`tsdml.hac.compute_hac_bandwidth`.
    hac_bandwidth_value : int, optional
        Bandwidth when ``hac_bandwidth_rule='fixed'``.
    use_fixed_b_critical : bool
        Use Kiefer-Vogelsang (2005) / LLSW (2018) fixed-``b`` critical values.

    Returns
    -------
    dict
        ``coef``, ``std_error``, ``t_stat``, ``p_value``, ``ci_lower``,
        ``ci_upper``, ``block_ses``, ``block_coefs``.
    """
    outcome_residuals = np.asarray(outcome_residuals, dtype=float)
    treatment_residuals = np.asarray(treatment_residuals, dtype=float)
    coef_idx = 1 if include_constant else 0
    T_total = len(outcome_residuals)

    optimal_lag = None
    if use_hac:
        try:
            optimal_lag = compute_hac_bandwidth(
                rule=hac_bandwidth_rule,
                horizon=horizon,
                T=T_total,
                value=hac_bandwidth_value,
            )
        except Exception:
            optimal_lag = calculate_optimal_lag(horizon)

    # ---------------------------------------------------------------- full -- #
    if estimation_method == "full":
        X = sm.add_constant(treatment_residuals) if include_constant else treatment_residuals
        model = sm.OLS(outcome_residuals, X)
        if use_hac and optimal_lag is not None:
            results = fit_ols_with_hac(model, kernel=hac_kernel,
                                       bandwidth=optimal_lag, horizon=horizon)
        else:
            results = model.fit()

        coef = float(results.params[coef_idx])
        se = float(results.bse[coef_idx])
        t_stat = abs(coef / se) if se > 1e-10 else 0.0
        df = T_total - (2 if include_constant else 1)
        crit = get_critical_value(confidence_level, T_total, use_fixed_critical=True)
        return {
            "coef": coef,
            "std_error": se,
            "t_stat": t_stat,
            "ci_lower": coef - crit * se,
            "ci_upper": coef + crit * se,
            "p_value": float(2 * (1 - stats.t.cdf(abs(t_stat), df))),
            "block_ses": None,
            "block_coefs": None,
        }

    if estimation_method != "block":
        raise ValueError(f"Unknown estimation_method: {estimation_method}")

    # --------------------------------------------------------------- block -- #
    if main_blocks is not None:
        blocks_to_use = main_blocks
    else:
        block_size = T_total // n_blocks
        blocks_to_use = [
            list(range(i * block_size,
                       T_total if i == n_blocks - 1 else (i + 1) * block_size))
            for i in range(n_blocks)
        ]

    block_coefs: List[float] = []
    block_ses: List[float] = []
    block_resid: Dict[int, np.ndarray] = {}
    used_blocks: List[List[int]] = []

    for k, main_idx in enumerate(blocks_to_use):
        if len(main_idx) <= 1:
            continue
        y_b = outcome_residuals[main_idx]
        t_b = treatment_residuals[main_idx]
        X_b = sm.add_constant(t_b) if include_constant else t_b
        res = sm.OLS(y_b, X_b).fit()
        block_coefs.append(float(res.params[coef_idx]))
        block_ses.append(float(res.bse[coef_idx]))
        block_resid[len(used_blocks)] = np.asarray(res.resid, dtype=float)
        used_blocks.append(list(main_idx))

    K = len(block_coefs)
    if K == 0:
        return {
            "coef": 0.0, "std_error": np.nan, "t_stat": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan, "p_value": np.nan,
            "block_ses": block_ses, "block_coefs": block_coefs,
        }

    coef = float(np.mean(block_coefs))

    if use_hac_final_se:
        # e_{t,k} = eps_{t,k} + (theta_k - theta) * xi_t
        all_idx: List[int] = []
        pieces = []
        for k, main_idx in enumerate(used_blocks):
            idx = np.asarray(main_idx)
            all_idx.extend(idx.tolist())
            pieces.append(block_resid[k] + (block_coefs[k] - coef) * treatment_residuals[idx])

        all_idx = sorted(all_idx)
        e_t = np.concatenate(pieces)
        xi_main = treatment_residuals[all_idx]
        score_t = xi_main * e_t
        jacobian = float((xi_main ** 2).mean())

        if hac_lag is not None:
            sigma_hat = hac_lrv(score_t, K=K, kernel=hac_kernel, bandwidth=int(hac_lag))
            bw_used: Optional[int] = int(hac_lag)
        elif (hac_bandwidth_rule or "small").lower() == "small" and hac_bandwidth_value is None:
            sigma_hat = hac_lrv(score_t, K=K, kernel=hac_kernel)
            bw_used = None
        else:
            bw_final = compute_hac_bandwidth(
                rule=hac_bandwidth_rule, horizon=horizon, T=len(score_t),
                scores=score_t, value=hac_bandwidth_value,
            )
            sigma_hat = hac_lrv(score_t, K=K, kernel=hac_kernel, bandwidth=bw_final)
            bw_used = int(bw_final)

        var_theta = float(sigma_hat) / (jacobian ** 2)
        se = float(np.sqrt(var_theta / len(all_idx)))
        n_total = len(all_idx)
    else:
        se = float(np.sqrt(np.sum((1.0 / K) ** 2 * np.asarray(block_ses) ** 2)))
        bw_used = None
        score_t = np.empty(0)
        n_total = sum(len(b) for b in used_blocks)

    t_stat = abs(coef / se) if se > 1e-10 else 0.0
    df = n_total - (2 if include_constant else 1)

    fb_kernel = fb_b = fb_nu = None
    if use_fixed_b_critical and use_hac_final_se:
        kn = normalize_kernel(hac_kernel)
        if kn == "ewc":
            fb_kernel = "ewc"
            fb_nu = bw_used if bw_used is not None else compute_hac_bandwidth(
                rule="lls_ewc", T=len(score_t)
            )
        elif kn == "bartlett" and bw_used is not None and len(score_t) > 0:
            fb_kernel = "bartlett"
            fb_b = float(bw_used) / float(len(score_t))

    crit = get_critical_value(
        confidence_level, n_total, use_fixed_critical=True,
        fixed_b_kernel=fb_kernel, fixed_b_b=fb_b, fixed_b_nu=fb_nu, m=1,
    )

    return {
        "coef": coef,
        "std_error": se,
        "t_stat": t_stat,
        "ci_lower": coef - crit * se,
        "ci_upper": coef + crit * se,
        "p_value": float(2 * (1 - stats.t.cdf(abs(t_stat), df))),
        "block_ses": block_ses,
        "block_coefs": block_coefs,
    }


# --------------------------------------------------------------------------- #
# Descriptive per-block / pooled statistics
# --------------------------------------------------------------------------- #

def block_statistics(
    outcome_residuals: np.ndarray,
    treatment_residuals: np.ndarray,
    main_blocks: List[List[int]],
    include_constant: bool = True,
    use_hac: bool = False,
    hac_lag: Optional[int] = None,
    confidence_level: float = 0.95,
    hac_kernel: str = "bartlett",
) -> List[Dict[str, object]]:
    """
    Per-block OLS diagnostics (coefficient, SE, ``t``, ``p``, CI, ``n``).

    These are *reporting* quantities.  Inference on the fold-average estimate
    comes from :func:`fit_stage_two`, which accounts for cross-block dependence.
    """
    coef_idx = 1 if include_constant else 0
    alpha = 1.0 - confidence_level
    out: List[Dict[str, object]] = []

    for k, main_idx in enumerate(main_blocks):
        null = {
            "block_id": k, "coefficient": 0.0, "std_error": float("nan"),
            "t_statistic": float("nan"), "p_value": float("nan"),
            "ci_lower": float("nan"), "ci_upper": float("nan"),
            "n_obs": len(main_idx) if main_idx else 0, "df": 0,
        }
        if not main_idx or len(main_idx) <= 1:
            out.append(null)
            continue
        t_b = treatment_residuals[main_idx]
        if np.var(t_b) <= 1e-10:
            out.append(null)
            continue
        try:
            y_b = outcome_residuals[main_idx]
            X_b = sm.add_constant(t_b) if include_constant else t_b
            model = sm.OLS(y_b, X_b)
            res = (fit_ols_with_hac(model, kernel=hac_kernel, bandwidth=hac_lag)
                   if (use_hac and hac_lag is not None) else model.fit())
            coef = float(res.params[coef_idx])
            se = float(res.bse[coef_idx])
            df = float(res.df_resid)
            crit = stats.t.ppf(1 - alpha / 2, df)
            out.append({
                "block_id": k,
                "coefficient": coef,
                "std_error": se,
                "t_statistic": float(abs(res.tvalues[coef_idx])),
                "p_value": float(res.pvalues[coef_idx]),
                "ci_lower": coef - crit * se,
                "ci_upper": coef + crit * se,
                "n_obs": len(y_b),
                "df": df,
            })
        except Exception:
            out.append(null)
    return out


def full_statistics(
    outcome_residuals: np.ndarray,
    treatment_residuals: np.ndarray,
    include_constant: bool = True,
    use_hac: bool = False,
    hac_lag: Optional[int] = None,
    confidence_level: float = 0.95,
    hac_kernel: str = "bartlett",
) -> Dict[str, object]:
    """Pooled residual-on-residual OLS diagnostics for the whole sample."""
    stats_ = block_statistics(
        outcome_residuals, treatment_residuals,
        [list(range(len(outcome_residuals)))],
        include_constant=include_constant, use_hac=use_hac, hac_lag=hac_lag,
        confidence_level=confidence_level, hac_kernel=hac_kernel,
    )[0]
    stats_.pop("block_id", None)
    return stats_
