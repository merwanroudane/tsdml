"""
Diagnostics for the conditional stability condition.

RCF drops the buffer blocks of neighbor-deletion designs, so its validity rests
entirely on **conditional stability** (Assumption 2.4, Remark 2.1):

.. math::

    \\mathbb E[\\xi_t \\mid X_t, \\mathcal F_{\\mathrm{aux},k}] = 0,
    \\qquad
    \\mathbb E[\\epsilon_t \\mid X_t, \\mathcal F_{\\mathrm{aux},k}] = 0 .

In words: after conditioning on :math:`X_t`, the adjacent auxiliary blocks must
not predict the main-block residual innovations.  Section 2.3 of the paper
notes that this can fail through residual serial dependence after conditioning,
omitted persistent states, asymmetric volatility, or contemporaneously
endogenous regimes -- and suggests, in practice, checking whether leads and lags
of adjacent auxiliary residuals predict main-block residuals.  When they do, the
remedy is either buffer deletion (:class:`tsdml.nlo.NLOCrossFitting`) or a
richer conditioning set.

The two tests here implement that suggestion.  They are operational versions of
a diagnostic the paper describes in words; the paper's own supplementary
appendix reports the formal size and power properties, and these functions are
not a line-by-line port of it.

Read them as *screening* tools: a rejection is a reason to enrich :math:`X_t`
or to switch to a buffered design, not a formal test of Theorem 2.1.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .folds import reverse_cf_folds

__all__ = ["boundary_leakage_test", "residual_autocorrelation_test", "diagnose"]


def boundary_leakage_test(
    residuals: np.ndarray,
    n_blocks: int,
    max_lag: int = 2,
    controls: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Test whether neighbouring observations predict main-block residuals.

    For each fold ``k`` the main-block residuals are regressed on their own
    ``max_lag`` leads and lags taken from the *adjacent* blocks, i.e. only the
    observations that RCF would have deleted under a buffered design.  A joint
    Wald test on those coefficients is reported.

    Parameters
    ----------
    residuals : ndarray, shape (T,)
        Out-of-sample residuals in time order -- typically the policy residuals
        :math:`\\hat\\xi_t` from
        :meth:`tsdml.rcf.ReverseCrossFitting.get_residuals`.
    n_blocks : int
        The ``K`` used in estimation.
    max_lag : int, default 2
        Number of leads and lags from each neighbouring block.
    controls : ndarray, shape (T, q), optional
        Extra regressors to partial out first, if you want the test conditional
        on more than a constant.

    Returns
    -------
    DataFrame
        One row per fold plus a pooled row, with ``F_stat``, ``p_value``,
        ``n_obs`` and ``df``.  Small p-values point to leakage across block
        boundaries.

    Examples
    --------
    White noise residuals should not reject:

    >>> rng = np.random.default_rng(0)
    >>> out = boundary_leakage_test(rng.standard_normal(240), n_blocks=6)
    >>> bool(out.loc[out['fold'] == 'pooled', 'p_value'].iloc[0] > 0.05)
    True
    """
    r = np.asarray(residuals, dtype=float).ravel()
    T = len(r)
    if max_lag < 1:
        raise ValueError("max_lag must be at least 1")
    blocks = reverse_cf_folds(T, n_blocks)
    rows: List[Dict[str, object]] = []

    pooled_t: List[int] = []

    def design(times: List[int]) -> np.ndarray:
        cols = [r[np.asarray(times) + s]
                for s in list(range(-max_lag, 0)) + list(range(1, max_lag + 1))]
        Z = np.column_stack(cols)
        if controls is not None:
            Z = np.column_stack([Z, np.asarray(controls, dtype=float)[np.asarray(times)]])
        return Z

    for k, main in enumerate(blocks.main_blocks):
        block = set(main)
        # boundary observations: those whose lead/lag window reaches outside the
        # fold's own main block -- exactly the observations a buffered design
        # would have deleted.
        times = [
            t for t in main
            if max_lag <= t < T - max_lag
            and any((t + s) not in block
                    for s in list(range(-max_lag, 0)) + list(range(1, max_lag + 1)))
        ]
        pooled_t.extend(times)
        if len(times) < 2 * max_lag + 3:
            # too few boundary observations in this fold for its own regression;
            # they still contribute to the pooled test below
            rows.append({"fold": k, "F_stat": np.nan, "p_value": np.nan,
                         "n_obs": len(times), "df": 0})
            continue
        rows.append({"fold": k, **_joint_wald(r[np.asarray(times)], design(times))})

    if len(pooled_t) >= 2 * max_lag + 3:
        pooled_t = sorted(pooled_t)
        rows.append({"fold": "pooled",
                     **_joint_wald(r[np.asarray(pooled_t)], design(pooled_t))})

    return pd.DataFrame(rows)


def _joint_wald(y: np.ndarray, Z: np.ndarray) -> Dict[str, object]:
    """OLS of ``y`` on ``[1, Z]`` and an F-test that all Z coefficients are 0."""
    X = sm.add_constant(Z, has_constant="add")
    if X.shape[0] <= X.shape[1] + 1:
        return {"F_stat": np.nan, "p_value": np.nan, "n_obs": int(X.shape[0]), "df": 0}
    res = sm.OLS(y, X).fit()
    R = np.zeros((Z.shape[1], X.shape[1]))
    for i in range(Z.shape[1]):
        R[i, i + 1] = 1.0
    try:
        test = res.f_test(R)
        return {
            "F_stat": float(np.squeeze(test.fvalue)),
            "p_value": float(np.squeeze(test.pvalue)),
            "n_obs": int(X.shape[0]),
            "df": int(Z.shape[1]),
        }
    except Exception:  # pragma: no cover - singular designs
        return {"F_stat": np.nan, "p_value": np.nan, "n_obs": int(X.shape[0]),
                "df": int(Z.shape[1])}


def residual_autocorrelation_test(
    residuals: np.ndarray,
    lags: int = 8,
) -> pd.DataFrame:
    """
    Ljung-Box test on cross-fitted residuals.

    Conditional stability implies the residual innovations carry no predictable
    component once :math:`X_t` is conditioned on.  Strong residual
    autocorrelation is the simplest signal that the conditioning set is too
    thin -- typically too few lags, or a missing persistent state.

    Parameters
    ----------
    residuals : ndarray, shape (T,)
    lags : int, default 8

    Returns
    -------
    DataFrame
        Columns ``lag``, ``lb_stat``, ``lb_pvalue``.

    Examples
    --------
    >>> rng = np.random.default_rng(1)
    >>> out = residual_autocorrelation_test(rng.standard_normal(300), lags=4)
    >>> list(out.columns)
    ['lag', 'lb_stat', 'lb_pvalue']
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    r = np.asarray(residuals, dtype=float).ravel()
    out = acorr_ljungbox(r, lags=range(1, lags + 1), return_df=True)
    out = out.reset_index().rename(columns={"index": "lag"})
    return out[["lag", "lb_stat", "lb_pvalue"]]


def diagnose(estimator, max_lag: int = 2, lb_lags: int = 8) -> Dict[str, pd.DataFrame]:
    """
    Run the full conditional-stability screen on a fitted estimator.

    Parameters
    ----------
    estimator : ReverseCrossFitting or NLOCrossFitting
        A fitted estimator exposing ``get_residuals()`` and ``n_blocks``.
    max_lag : int, default 2
        Leads/lags for :func:`boundary_leakage_test`.
    lb_lags : int, default 8
        Lags for :func:`residual_autocorrelation_test`.

    Returns
    -------
    dict of DataFrame
        ``leakage_policy``, ``leakage_outcome``, ``ljungbox_policy``,
        ``ljungbox_outcome``.
    """
    resid = estimator.get_residuals()
    K = estimator.n_blocks
    return {
        "leakage_policy": boundary_leakage_test(resid["treatment"], K, max_lag),
        "leakage_outcome": boundary_leakage_test(resid["outcome"], K, max_lag),
        "ljungbox_policy": residual_autocorrelation_test(resid["treatment"], lb_lags),
        "ljungbox_outcome": residual_autocorrelation_test(resid["outcome"], lb_lags),
    }
