"""
DML local projections: horizon-specific dynamic causal effects.

Section 4.2 of the paper takes the partially linear model to the local
projection (LP) setting:

.. math::

    y_{t+h} = \\theta_h d_t + g_h(X_t) + \\epsilon_{t+h}, \\qquad
    d_t = m_0(X_t) + \\xi_t ,

and residualises at every horizon,
:math:`\\hat\\chi_{t+h} = \\theta_h \\hat\\xi_t + \\tilde\\epsilon_{t+h}`
(eq. 4.16).  Causal reading of :math:`\\theta_h` requires the horizon-specific
exogeneity condition :math:`\\mathbb E[\\epsilon_{t+h} \\mid X_t, d_t] = 0`
(eq. 4.17) -- the dynamic analogue of conditional unconfoundedness.  DML does
not deliver that condition; it makes high-dimensional conditioning feasible so
that the condition is more plausible.

Because ``X`` and ``d`` are fixed across horizons and only the outcome moves,
the policy residuals :math:`\\hat\\xi_t` are computed once and reused -- the
caching path in :class:`tsdml.rcf.ReverseCrossFitting`.

For outcomes entering in differences the class reports **cumulative** responses
by summing residualised outcomes up to ``h`` and estimating a single direct LP
(eq. 5.18), which yields HAC standard errors horizon by horizon without needing
the joint covariance of :math:`\\{\\hat\\theta_h\\}`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .rcf import ReverseCrossFitting, _stars
from .stage2 import block_statistics, fit_stage_two

__all__ = ["DMLLocalProjections", "cumulate_residuals", "scale_irfs"]


# --------------------------------------------------------------------------- #
# Residual cumulation
# --------------------------------------------------------------------------- #

#: Transformation codes whose residuals are cumulated across horizons without
#: further rescaling.
CUMULATE_CODES = (2, 3, 5, 7, 9)
#: Codes cumulated then divided by 12 (twelfth log-differences, monthly).
CUMULATE_DIV12_CODES = (8, 10)
#: Codes cumulated then divided by 4 (fourth log-differences, quarterly).
CUMULATE_DIV4_CODES = (81, 101)


def cumulate_residuals(
    residuals_by_horizon: Dict[int, Dict[str, np.ndarray]],
    horizon: int,
    outcome_code: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Build the horizon-``h`` regressand for a cumulative response.

    Parameters
    ----------
    residuals_by_horizon : dict
        ``{h: {'outcome': ndarray, 'treatment': ndarray}}`` for every horizon up
        to and including ``horizon``.
    horizon : int
        Current horizon ``h``.
    outcome_code : int, optional
        Transformation code of the outcome (see :mod:`tsdml.prep`):

        ``None`` or ``1``
            level outcome -- return the horizon-``h`` residuals untouched;
        ``2, 3, 5, 7, 9``
            differenced outcome -- cumulate :math:`\\sum_{j=0}^{h}\\hat\\chi_{t+j}`;
        ``8, 10``
            cumulate and divide by 12 (year-on-year monthly);
        ``81, 101``
            cumulate and divide by 4 (year-on-year quarterly);
        ``12``
            cumulate, add the current horizon once more, divide by 12;
        anything else
            no cumulation.

    Returns
    -------
    dict
        ``{'outcome': ndarray, 'treatment': ndarray}``.  Policy residuals are
        always those of the current horizon -- they do not cumulate.

    Examples
    --------
    >>> r = {0: {'outcome': np.array([1.0, 2.0]), 'treatment': np.array([.1, .2])},
    ...      1: {'outcome': np.array([3.0, 4.0]), 'treatment': np.array([.1, .2])}}
    >>> cumulate_residuals(r, 1, outcome_code=5)['outcome']
    array([4., 6.])
    """
    if horizon not in residuals_by_horizon:
        raise ValueError(f"horizon {horizon} missing from residuals dict")

    current = residuals_by_horizon[horizon]
    if outcome_code is None or outcome_code == 1:
        return {k: np.array(v, copy=True) for k, v in current.items()}

    outcome = np.array(current["outcome"], copy=True)
    treatment = np.array(current["treatment"], copy=True)

    if horizon > 0:
        for prev in range(horizon):
            if prev in residuals_by_horizon:
                outcome = outcome + residuals_by_horizon[prev]["outcome"]

    if outcome_code in CUMULATE_CODES:
        pass
    elif outcome_code in CUMULATE_DIV12_CODES:
        outcome = outcome / 12.0
    elif outcome_code in CUMULATE_DIV4_CODES:
        outcome = outcome / 4.0
    elif outcome_code == 12:
        outcome = (outcome + current["outcome"]) / 12.0
    else:
        return {k: np.array(v, copy=True) for k, v in current.items()}

    return {"outcome": outcome, "treatment": treatment}


# --------------------------------------------------------------------------- #
# Estimator
# --------------------------------------------------------------------------- #

class DMLLocalProjections:
    """
    Impulse responses by RCF-DML local projections.

    Parameters
    ----------
    outcome_learner, treatment_learner : sklearn-compatible regressors, optional
        Used when no per-fold learners are supplied.
    block_specific_learners : dict, optional
        Per-fold learners shared by every horizon -- normally
        :attr:`tsdml.calibration.Calibrator.block_specific_learners_`.
    horizon_specific_learners : dict, optional
        ``{h: {'outcome_learners': [...], 'treatment_learners': [...]}}`` to
        re-tune at each horizon.  Slower, and it forfeits the policy-residual
        cache.
    n_blocks : int, default 6
        Folds ``K``.  The paper's application uses 6.
    estimation_method : {'block', 'full'}, default 'block'
    outcome_code : int, optional
        Transformation code of the outcome, controlling cumulation (see
        :func:`cumulate_residuals`).  Pass ``5`` for a log-differenced series to
        obtain a cumulative percentage response, ``2`` for a first-differenced
        rate or spread.
    include_constant : bool, default True
    use_hac : bool, default True
    confidence_level : float, default 0.95
        Level of the primary interval.  A 90% interval is always reported too.
    hac_kernel, hac_bandwidth_rule, hac_bandwidth_value, use_fixed_b_critical
        Passed through to :func:`tsdml.stage2.fit_stage_two`.  With the default
        ``'small'`` rule the truncation lag is :math:`m = \\min(h+1, 24)`.
    outcome_name : str, optional
        Label used in printed output and figures.
    random_state : int, optional

    Attributes
    ----------
    horizons_ : ndarray
    irf_ : ndarray
        Point estimates :math:`\\hat\\theta_h`.
    results_ : dict
        ``{h: {...}}`` full statistics per horizon.
    residuals_ : dict
        Raw per-horizon residuals from stage one.
    estimators_ : dict
        The fitted :class:`~tsdml.rcf.ReverseCrossFitting` object per horizon.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Lasso
    >>> rng = np.random.default_rng(7)
    >>> T, H = 180, 4
    >>> X = rng.standard_normal((T, 6))
    >>> d = X[:, 0] + rng.standard_normal(T)
    >>> y = 1.0 * d + rng.standard_normal(T)
    >>> leads = np.column_stack([np.roll(y, -h) for h in range(H + 1)])
    >>> lp = DMLLocalProjections(Lasso(alpha=0.01), Lasso(alpha=0.01), n_blocks=5)
    >>> _ = lp.fit(X, y, d, leads)
    >>> lp.irf_.shape
    (5,)
    """

    def __init__(
        self,
        outcome_learner: Any = None,
        treatment_learner: Any = None,
        block_specific_learners: Optional[Dict[str, List[Any]]] = None,
        horizon_specific_learners: Optional[Dict[int, Dict[str, List[Any]]]] = None,
        n_blocks: int = 6,
        estimation_method: str = "block",
        outcome_code: Optional[int] = None,
        include_constant: bool = True,
        use_hac: bool = True,
        confidence_level: float = 0.95,
        hac_kernel: str = "bartlett",
        hac_bandwidth_rule: str = "small",
        hac_bandwidth_value: Optional[int] = None,
        use_fixed_b_critical: bool = False,
        outcome_name: Optional[str] = None,
        random_state: Optional[int] = None,
    ):
        self.outcome_learner = outcome_learner
        self.treatment_learner = treatment_learner
        self.block_specific_learners = block_specific_learners
        self.horizon_specific_learners = horizon_specific_learners
        self.n_blocks = int(n_blocks)
        self.estimation_method = estimation_method
        self.outcome_code = outcome_code
        self.include_constant = bool(include_constant)
        self.use_hac = bool(use_hac)
        self.confidence_level = float(confidence_level)
        self.hac_kernel = hac_kernel
        self.hac_bandwidth_rule = hac_bandwidth_rule
        self.hac_bandwidth_value = hac_bandwidth_value
        self.use_fixed_b_critical = bool(use_fixed_b_critical)
        self.outcome_name = outcome_name
        self.random_state = random_state

        self.horizons_: Optional[np.ndarray] = None
        self.irf_: Optional[np.ndarray] = None
        self.results_: Dict[int, Dict[str, Any]] = {}
        self.residuals_: Dict[int, Dict[str, np.ndarray]] = {}
        self.residuals_cumulated_: Dict[int, Dict[str, np.ndarray]] = {}
        self.estimators_: Dict[int, ReverseCrossFitting] = {}
        self.block_statistics_: Dict[int, List[Dict[str, Any]]] = {}
        self.index_ = None
        self.is_fitted_ = False

    # ------------------------------------------------------------------ fit -- #

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        treatment: np.ndarray,
        leads: np.ndarray,
        verbose: bool = False,
        index=None,
    ) -> "DMLLocalProjections":
        """
        Estimate :math:`\\theta_h` for every horizon in ``leads``.

        Parameters
        ----------
        X : array-like, shape (T, p)
            Time-ordered controls.
        y : array-like, shape (T,)
            Horizon-0 outcome.
        treatment : array-like, shape (T,)
            Policy variable.
        leads : array-like, shape (T, H+1)
            Column ``h`` holds :math:`y_{t+h}`; column 0 must equal ``y``.
            :meth:`tsdml.prep.DataProcessor.data_prep` builds this for you.
        verbose : bool, default False
        index : array-like, optional
            Date index carried into the output frames.

        Returns
        -------
        self
        """
        X = np.asarray(getattr(X, "values", X), dtype=float)
        y = np.asarray(getattr(y, "values", y), dtype=float).ravel()
        treatment = np.asarray(getattr(treatment, "values", treatment), dtype=float).ravel()
        leads = np.asarray(getattr(leads, "values", leads), dtype=float)
        if leads.ndim != 2:
            raise ValueError(f"leads must be 2-D (T, H+1), got shape {leads.shape}")

        self.horizons_ = np.arange(leads.shape[1])
        self.index_ = index

        use_horizon_specific = self.horizon_specific_learners is not None
        base = None
        if not use_horizon_specific:
            base = ReverseCrossFitting(
                outcome_learner=self.outcome_learner,
                treatment_learner=self.treatment_learner,
                block_specific_learners=self.block_specific_learners,
                n_blocks=self.n_blocks,
                estimation_method=self.estimation_method,
                include_constant=self.include_constant,
                use_hac=self.use_hac,
                confidence_level=self.confidence_level,
                hac_kernel=self.hac_kernel,
                hac_bandwidth_rule=self.hac_bandwidth_rule,
                hac_bandwidth_value=self.hac_bandwidth_value,
                use_fixed_b_critical=self.use_fixed_b_critical,
                cache_residuals=True,
                random_state=self.random_state,
            )

        if verbose:
            label = self.outcome_name or "outcome"
            print(f"\nRCF-DML local projections for {label}: "
                  f"{len(self.horizons_)} horizons, K={self.n_blocks}")

        for h in self.horizons_:
            y_h = y if h == 0 else leads[:, h]

            if use_horizon_specific:
                learners = self.horizon_specific_learners.get(int(h))
                if learners is None:
                    raise ValueError(f"no learners supplied for horizon {h}")
                est = ReverseCrossFitting(
                    block_specific_learners=learners,
                    n_blocks=self.n_blocks,
                    estimation_method=self.estimation_method,
                    include_constant=self.include_constant,
                    use_hac=self.use_hac,
                    confidence_level=self.confidence_level,
                    hac_kernel=self.hac_kernel,
                    hac_bandwidth_rule=self.hac_bandwidth_rule,
                    hac_bandwidth_value=self.hac_bandwidth_value,
                    use_fixed_b_critical=self.use_fixed_b_critical,
                    cache_residuals=False,
                    random_state=self.random_state,
                )
                est.fit(X, y_h, treatment, verbose=False)
            else:
                base.fit(X, y_h, treatment, verbose=False)
                est = base

            self.estimators_[int(h)] = est
            self.residuals_[int(h)] = {k: v.copy() for k, v in est.get_residuals().items()}
            self.residuals_cumulated_[int(h)] = cumulate_residuals(
                self.residuals_, int(h), self.outcome_code)

            self._estimate_horizon(int(h), est)

            if verbose:
                r = self.results_[int(h)]
                print(f"  h={h:>2}  theta={r['coefficient']:>10.5f}  "
                      f"se={r['std_error']:>9.5f}  "
                      f"p={r['p_value']:.4f} {_stars(r['p_value'])}")

        self.irf_ = np.array([self.results_[int(h)]["coefficient"] for h in self.horizons_])
        self.is_fitted_ = True
        return self

    def _estimate_horizon(self, h: int, est: ReverseCrossFitting) -> None:
        resid = self.residuals_cumulated_[h]
        chi, xi = resid["outcome"], resid["treatment"]
        main_blocks = est.blocks_.main_blocks

        # Historical default of the paper's code: the cross-fitted final HAC
        # variance uses lag h+1 under the 'small' rule; other rules pick their
        # own bandwidth inside fit_stage_two.
        hac_lag = (h + 1) if (self.hac_bandwidth_rule or "small") == "small" else None

        stats_block = fit_stage_two(
            outcome_residuals=chi,
            treatment_residuals=xi,
            estimation_method=self.estimation_method,
            n_blocks=self.n_blocks,
            main_blocks=main_blocks,
            include_constant=self.include_constant,
            use_hac=self.use_hac,
            horizon=h,
            confidence_level=self.confidence_level,
            hac_lag=hac_lag,
            hac_kernel=self.hac_kernel,
            hac_bandwidth_rule=self.hac_bandwidth_rule,
            hac_bandwidth_value=self.hac_bandwidth_value,
            use_fixed_b_critical=self.use_fixed_b_critical,
        )

        coef, se = stats_block["coef"], stats_block["std_error"]
        self.results_[h] = {
            "horizon": h,
            "coefficient": coef,
            "std_error": se,
            "t_stat": stats_block["t_stat"],
            "p_value": stats_block["p_value"],
            "ci_lower_95": stats_block["ci_lower"],
            "ci_upper_95": stats_block["ci_upper"],
            "ci_lower_90": coef - 1.645 * se,
            "ci_upper_90": coef + 1.645 * se,
            "block_coefs": stats_block["block_coefs"],
        }
        self.block_statistics_[h] = block_statistics(
            chi, xi, main_blocks,
            include_constant=self.include_constant,
            use_hac=self.use_hac,
            hac_lag=(h + 1) if self.use_hac else None,
            confidence_level=self.confidence_level,
            hac_kernel=self.hac_kernel,
        )

    # --------------------------------------------------------------- output -- #

    def to_frame(self) -> pd.DataFrame:
        """
        Impulse response as a tidy frame.

        Columns: ``Horizon``, ``Coefficient``, ``Std_Error``, ``t_stat``,
        ``p_value``, ``CI_Lower_95``, ``CI_Upper_95``, ``CI_Lower_90``,
        ``CI_Upper_90`` -- the schema consumed by
        :func:`tsdml.plots.plot_irf_panel` and :func:`tsdml.tables.irf_table`.
        """
        self._check_fitted()
        rows = []
        for h in self.horizons_:
            r = self.results_[int(h)]
            rows.append({
                "Horizon": int(h),
                "Coefficient": r["coefficient"],
                "Std_Error": r["std_error"],
                "t_stat": r["t_stat"],
                "p_value": r["p_value"],
                "CI_Lower_95": r["ci_lower_95"],
                "CI_Upper_95": r["ci_upper_95"],
                "CI_Lower_90": r["ci_lower_90"],
                "CI_Upper_90": r["ci_upper_90"],
            })
        return pd.DataFrame(rows)

    def residuals_frame(self) -> pd.DataFrame:
        """
        Residuals for inspection: policy residuals plus outcome residuals per
        horizon, indexed by date when an index was supplied to :meth:`fit`.
        """
        self._check_fitted()
        data = {"treatment_resid": self.residuals_[0]["treatment"]}
        for h in self.horizons_:
            data[f"outcome_resid_h{int(h)}"] = self.residuals_[int(h)]["outcome"]
        frame = pd.DataFrame(data)
        if self.index_ is not None and len(self.index_) == len(frame):
            frame.index = pd.Index(self.index_, name="date")
        return frame

    def summary(self) -> None:
        """Print the horizon-by-horizon impulse response table."""
        self._check_fitted()
        label = self.outcome_name or "outcome"
        pct = int(round(self.confidence_level * 100))
        print("\n" + "=" * 84)
        print(f"RCF-DML local projections  --  {label}")
        print("=" * 84)
        print(f"folds K = {self.n_blocks}   stage-2 = {self.estimation_method}   "
              f"HAC = {self.hac_kernel}   outcome code = {self.outcome_code}")
        print("-" * 84)
        print(f"{'h':>3}{'coef':>13}{'std err':>12}{'t':>9}{'P>|t|':>9}"
              f"{f'  [{pct}% CI]':>26}")
        print("-" * 84)
        for h in self.horizons_:
            r = self.results_[int(h)]
            print(f"{int(h):>3}{r['coefficient']:>13.5f}{r['std_error']:>12.5f}"
                  f"{r['t_stat']:>9.3f}{r['p_value']:>9.4f}"
                  f"   [{r['ci_lower_95']:>9.5f}, {r['ci_upper_95']:>9.5f}]"
                  f" {_stars(r['p_value'])}")
        print("=" * 84)
        print("signif. codes:  *** 0.01   ** 0.05   * 0.10\n")

    def plot(self, **kwargs):
        """
        Plot this impulse response.  See :func:`tsdml.plots.plot_irf`.
        """
        from .plots import plot_irf
        return plot_irf(self.to_frame(), title=self.outcome_name, **kwargs)

    def save(self, path: str) -> pd.DataFrame:
        """Write :meth:`to_frame` to ``path`` as CSV and return the frame."""
        frame = self.to_frame()
        frame.to_csv(path, index=False)
        return frame

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise ValueError("model is not fitted; call fit() first")


def scale_irfs(
    irfs: Dict[str, pd.DataFrame],
    numerator_var: str = "Tier 1 capital",
    denominator_var: str = "Risk-weighted assets",
    basis_point_vars=("PNFC_Spread",),
    target_shock: float = 0.5,
) -> Dict[str, pd.DataFrame]:
    """
    Normalise a set of impulse responses to a common policy shock.

    The paper reports responses to a 50 basis-point rise in the Tier 1
    capital ratio.  Since the ratio is capital over risk-weighted assets, the
    impact response of the ratio is the impact response of capital minus that
    of risk-weighted assets, and the scaling factor is

    .. math::

        c = \\frac{\\text{target}}
                  {\\hat\\theta_0^{\\text{capital}} - \\hat\\theta_0^{\\text{RWA}}}

    with ``target = 0.5`` percentage points, i.e. ``c = 1 / (2(\\cdot))``.
    Variables listed in ``basis_point_vars`` are additionally multiplied by 100.

    Parameters
    ----------
    irfs : dict of {str: DataFrame}
        Output of :meth:`DMLLocalProjections.to_frame`, keyed by variable name.
    numerator_var, denominator_var : str
        Keys of the two capital-ratio components.
    basis_point_vars : sequence of str
        Variables to express in basis points.
    target_shock : float, default 0.5
        Size of the normalised shock in the same units as the ratio.

    Returns
    -------
    dict of {str: DataFrame}
        Scaled copies; the inputs are not modified.
    """
    for key in (numerator_var, denominator_var):
        if key not in irfs:
            raise KeyError(f"irfs is missing '{key}', needed to build the scaling factor")

    impact_num = float(irfs[numerator_var]["Coefficient"].iloc[0])
    impact_den = float(irfs[denominator_var]["Coefficient"].iloc[0])
    denom = impact_num - impact_den
    if abs(denom) < 1e-12:
        raise ValueError(
            "impact responses of capital and risk-weighted assets are equal; "
            "the capital ratio does not move on impact and cannot be normalised"
        )
    factor = target_shock / denom

    cols = ["Coefficient", "CI_Lower_95", "CI_Upper_95", "CI_Lower_90", "CI_Upper_90"]
    out: Dict[str, pd.DataFrame] = {}
    for name, frame in irfs.items():
        scaled = frame.copy()
        mult = factor * 100.0 if name in basis_point_vars else factor
        for col in cols:
            if col in scaled.columns:
                scaled[col] = scaled[col] * mult
        if "Std_Error" in scaled.columns:
            scaled["Std_Error"] = scaled["Std_Error"] * abs(mult)
        out[name] = scaled
    return out
