"""
The Reverse Cross-Fitting DML estimator for the partially linear model.

Model (paper, eqs. 1.1-1.2):

.. math::

    y_t = \\theta_0 d_t + g_0(X_t) + \\epsilon_t, \\qquad
    d_t = m_0(X_t) + \\xi_t .

Stage 1
    For every fold ``k`` the nuisances :math:`\\hat g^{r,(k)}` and
    :math:`\\hat m^{(k)}` are estimated on the auxiliary sample -- the *right*
    blocks read backwards for early folds, the *left* blocks read forwards for
    late folds, both sides averaged for the central fold when ``K`` is odd --
    and out-of-sample residuals are formed on the main block :math:`B_k`.

Stage 2
    :math:`\\hat\\theta_k` is the residual-on-residual OLS slope on
    :math:`B_k`, and :math:`\\hat\\theta = K^{-1}\\sum_k \\hat\\theta_k`
    (eq. 2.5), with HAC inference from the stacked score sequence.

There are **no buffer blocks**: validity comes from the conditional stability
condition (Assumption 2.4 / Remark 2.1), not from fold independence.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import clone

from .folds import BlockStructure, fold_direction, reverse_cf_folds, sample_use_rcf
from .stage2 import block_statistics, fit_stage_two

__all__ = ["ReverseCrossFitting"]


class ReverseCrossFitting:
    """
    RCF-DML estimator of a scalar causal parameter in a time series.

    Parameters
    ----------
    outcome_learner : sklearn-compatible regressor, optional
        Learner for :math:`g_0^r(X) = \\mathbb E[y \\mid X]`.  Ignored when
        ``block_specific_learners`` is given.
    treatment_learner : sklearn-compatible regressor, optional
        Learner for :math:`m_0(X) = \\mathbb E[d \\mid X]`.
    n_blocks : int, default 5
        Number of folds ``K``.  The paper's application uses ``K = 6``.
    estimation_method : {'block', 'full'}, default 'block'
        Fold-average (paper default, eq. 2.5) or pooled regression.
    block_specific_learners : dict, optional
        ``{'outcome_learners': [...], 'treatment_learners': [...]}`` with one
        entry per fold -- typically
        :attr:`tsdml.calibration.Calibrator.block_specific_learners_`.
    include_constant : bool, default True
        Intercept in the stage-two regressions.
    use_hac : bool, default True
        HAC inference.
    confidence_level : float, default 0.95
    hac_kernel : {'bartlett', 'qs', 'parzen', 'ewc'}, default 'bartlett'
    hac_bandwidth_rule : str, default 'small'
        See :func:`tsdml.hac.compute_hac_bandwidth`.
    hac_bandwidth_value : int, optional
        Bandwidth when the rule is ``'fixed'``.
    use_fixed_b_critical : bool, default False
        Fixed-``b`` critical values (Kiefer-Vogelsang 2005 / LLSW 2018).
    cache_residuals : bool, default True
        Reuse the policy residuals when only the outcome changes between calls
        -- the case in local projections, where ``X`` and ``d`` are fixed and
        only :math:`y_{t+h}` moves.  Cuts stage-one work roughly in half.
    random_state : int, optional
        Forwarded to nothing internally (blocks are deterministic); kept so the
        estimator can carry a seed alongside stochastic learners.

    Attributes
    ----------
    theta_ : float
        The point estimate :math:`\\hat\\theta`.
    results_ : dict
        ``coef``, ``std_error``, ``t_stat``, ``p_value``, ``ci_lower``,
        ``ci_upper``, ``block_coefs``, ``block_ses``.
    residuals_ : dict
        ``{'outcome': ndarray, 'treatment': ndarray}`` out-of-sample residuals
        in original time order.
    blocks_ : BlockStructure
        The fold construction actually used.
    is_fitted_ : bool

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LassoCV
    >>> rng = np.random.default_rng(1)
    >>> T, p = 200, 10
    >>> X = rng.standard_normal((T, p))
    >>> d = X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(T)
    >>> y = 1.5 * d + 0.8 * X[:, 0] + rng.standard_normal(T)
    >>> est = ReverseCrossFitting(LassoCV(), LassoCV(), n_blocks=5).fit(X, y, d)
    >>> bool(abs(est.theta_ - 1.5) < 0.25)
    True
    """

    def __init__(
        self,
        outcome_learner: Any = None,
        treatment_learner: Any = None,
        n_blocks: int = 5,
        estimation_method: str = "block",
        block_specific_learners: Optional[Dict[str, List[Any]]] = None,
        include_constant: bool = True,
        use_hac: bool = True,
        confidence_level: float = 0.95,
        hac_kernel: str = "bartlett",
        hac_bandwidth_rule: str = "small",
        hac_bandwidth_value: Optional[int] = None,
        use_fixed_b_critical: bool = False,
        cache_residuals: bool = True,
        random_state: Optional[int] = None,
    ):
        if estimation_method not in ("block", "full"):
            raise ValueError("estimation_method must be 'block' or 'full'")
        if n_blocks < 2:
            raise ValueError("n_blocks must be at least 2")

        self.n_blocks = int(n_blocks)
        self.estimation_method = estimation_method
        self.include_constant = bool(include_constant)
        self.use_hac = bool(use_hac)
        self.confidence_level = float(confidence_level)
        self.hac_kernel = hac_kernel
        self.hac_bandwidth_rule = hac_bandwidth_rule
        self.hac_bandwidth_value = hac_bandwidth_value
        self.use_fixed_b_critical = bool(use_fixed_b_critical)
        self.cache_residuals = bool(cache_residuals)
        self.random_state = random_state

        if block_specific_learners is not None:
            missing = {"outcome_learners", "treatment_learners"} - set(block_specific_learners)
            if missing:
                raise ValueError(f"block_specific_learners is missing keys: {sorted(missing)}")
            outs = block_specific_learners["outcome_learners"]
            trts = block_specific_learners["treatment_learners"]
            if len(outs) != n_blocks or len(trts) != n_blocks:
                raise ValueError(
                    f"block_specific_learners must supply {n_blocks} learners per "
                    f"equation (got {len(outs)} outcome, {len(trts)} treatment)"
                )
            self.outcome_learners = list(outs)
            self.treatment_learners = list(trts)
            self.outcome_learner = outs[0]
            self.treatment_learner = trts[0]
            self.block_specific = True
        else:
            if outcome_learner is None or treatment_learner is None:
                raise ValueError(
                    "provide outcome_learner and treatment_learner, or "
                    "block_specific_learners"
                )
            self.outcome_learner = outcome_learner
            self.treatment_learner = treatment_learner
            self.outcome_learners = [outcome_learner] * n_blocks
            self.treatment_learners = [treatment_learner] * n_blocks
            self.block_specific = False

        self.theta_: Optional[float] = None
        self.results_: Optional[Dict[str, Any]] = None
        self.residuals_: Optional[Dict[str, np.ndarray]] = None
        self.blocks_: Optional[BlockStructure] = None
        self.block_statistics_: Optional[List[Dict[str, Any]]] = None
        self.is_fitted_ = False

        self._cached_treatment_residuals: Optional[np.ndarray] = None
        self._cached_outcome_hash: Optional[str] = None
        self._cached_treatment_hash: Optional[str] = None
        self._cached_n: Optional[int] = None

    # ------------------------------------------------------------------ fit -- #

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        treatment: Union[np.ndarray, pd.Series],
        verbose: bool = False,
    ) -> "ReverseCrossFitting":
        """
        Run stages one and two.

        Parameters
        ----------
        X : array-like, shape (T, p)
            Control matrix, **time-ordered**.  Rows must be consecutive periods:
            the block structure is positional, so any reshuffling invalidates it.
        y : array-like, shape (T,)
            Outcome.
        treatment : array-like, shape (T,)
            Scalar policy variable.
        verbose : bool, default False
            Print fold-by-fold progress.

        Returns
        -------
        self
        """
        X = self._as_array(X)
        y = self._as_array(y).ravel()
        treatment = self._as_array(treatment).ravel()

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        T = X.shape[0]
        if not (len(y) == len(treatment) == T):
            raise ValueError(
                f"length mismatch: X has {T} rows, y has {len(y)}, "
                f"treatment has {len(treatment)}"
            )
        if T % self.n_blocks:
            import warnings
            warnings.warn(
                f"T={T} is not divisible by K={self.n_blocks}; the last "
                f"{T % self.n_blocks} observation(s) fall outside every main "
                f"block and are used only for nuisance training.",
                stacklevel=2,
            )

        self.blocks_ = reverse_cf_folds(T, self.n_blocks)
        strategy = self._caching_strategy(y, treatment)

        if verbose:
            print(f"RCF-DML: T={T}, K={self.n_blocks}, "
                  f"block size={self.blocks_.block_size}, "
                  f"sample use={sample_use_rcf(self.n_blocks):.3f}")
            print(f"stage 1 residuals: {strategy}")

        chi, xi = self._stage_one(X, y, treatment, strategy, verbose)
        self.residuals_ = {"outcome": chi, "treatment": xi}

        if self.cache_residuals:
            self._cached_treatment_residuals = xi.copy()
            self._cached_outcome_hash = self._hash(y)
            self._cached_treatment_hash = self._hash(treatment)
            self._cached_n = T

        self.results_ = fit_stage_two(
            outcome_residuals=chi,
            treatment_residuals=xi,
            estimation_method=self.estimation_method,
            n_blocks=self.n_blocks,
            main_blocks=self.blocks_.main_blocks,
            include_constant=self.include_constant,
            use_hac=self.use_hac,
            horizon=0,
            confidence_level=self.confidence_level,
            hac_kernel=self.hac_kernel,
            hac_bandwidth_rule=self.hac_bandwidth_rule,
            hac_bandwidth_value=self.hac_bandwidth_value,
            use_fixed_b_critical=self.use_fixed_b_critical,
        )
        self.theta_ = float(self.results_["coef"])
        self.block_statistics_ = block_statistics(
            chi, xi, self.blocks_.main_blocks,
            include_constant=self.include_constant,
            use_hac=self.use_hac,
            hac_lag=1 if self.use_hac else None,
            confidence_level=self.confidence_level,
            hac_kernel=self.hac_kernel,
        )
        self.is_fitted_ = True

        if verbose:
            self.summary()
        return self

    # -------------------------------------------------------------- stage 1 -- #

    def _stage_one(self, X, y, treatment, strategy, verbose):
        T = X.shape[0]
        chi = np.zeros(T)

        reuse = (strategy == "reuse_treatment" and self._cached_treatment_residuals is not None)
        xi = self._cached_treatment_residuals.copy() if reuse else np.zeros(T)

        bs = self.blocks_
        K = self.n_blocks

        for k in range(K):
            main = np.asarray(bs.main_blocks[k])
            direction = fold_direction(k, K)

            if verbose:
                print(f"  fold {k}: {direction:<7} "
                      f"(main {main[0]}..{main[-1]})")

            if direction == "reverse":
                aux = np.asarray(bs.aux_right_blocks[k])
                if aux.size == 0:
                    continue
                chi[main], xi_k = self._fit_predict(
                    X, y, treatment, aux, main, reverse=True, k=k, skip_treatment=reuse)
                if not reuse:
                    xi[main] = xi_k

            elif direction == "forward":
                aux = np.asarray(bs.aux_left_blocks[k])
                if aux.size == 0:
                    continue
                chi[main], xi_k = self._fit_predict(
                    X, y, treatment, aux, main, reverse=False, k=k, skip_treatment=reuse)
                if not reuse:
                    xi[main] = xi_k

            else:  # central fold, odd K: average the two directions
                aux_r = np.asarray(bs.aux_right_blocks[k])
                aux_l = np.asarray(bs.aux_left_blocks[k])
                chi_r = xi_r = chi_l = xi_l = None

                if aux_r.size:
                    chi_r, xi_r = self._fit_predict(
                        X, y, treatment, aux_r, main, reverse=True, k=k, skip_treatment=reuse)
                if aux_l.size:
                    chi_l, xi_l = self._fit_predict(
                        X, y, treatment, aux_l, main, reverse=False, k=k, skip_treatment=reuse)

                if chi_r is not None and chi_l is not None:
                    chi[main] = 0.5 * (chi_r + chi_l)
                    if not reuse:
                        xi[main] = 0.5 * (xi_r + xi_l)
                elif chi_r is not None:
                    chi[main] = chi_r
                    if not reuse:
                        xi[main] = xi_r
                elif chi_l is not None:
                    chi[main] = chi_l
                    if not reuse:
                        xi[main] = xi_l

        return chi, xi

    def _fit_predict(self, X, y, treatment, aux, main, reverse, k, skip_treatment):
        """
        Fit the two nuisances on ``aux`` and residualise on ``main``.

        When ``reverse`` is True both index sets are read backwards before the
        learner sees them, and the residuals are flipped back into calendar
        order.  For a learner that is invariant to row order the flip is a
        no-op; it matters for learners with any sequential component, and it
        makes the time-reversal explicit rather than implicit.
        """
        if reverse:
            aux_idx = np.flip(aux)
            main_idx = np.flip(main)
        else:
            aux_idx, main_idx = aux, main

        g = clone(self.outcome_learners[k])
        g.fit(X[aux_idx], y[aux_idx])
        chi = y[main_idx] - g.predict(X[main_idx])

        if skip_treatment:
            xi = None
        else:
            m = clone(self.treatment_learners[k])
            m.fit(X[aux_idx], treatment[aux_idx])
            xi = treatment[main_idx] - m.predict(X[main_idx])

        if reverse:
            chi = chi[::-1]
            if xi is not None:
                xi = xi[::-1]
        return chi, xi

    # --------------------------------------------------------------- output -- #

    def get_residuals(self) -> Dict[str, np.ndarray]:
        """Out-of-sample residuals ``{'outcome': ..., 'treatment': ...}``."""
        self._check_fitted()
        return self.residuals_

    def summary(self) -> None:
        """Print a compact estimation table."""
        self._check_fitted()
        r = self.results_
        stars = _stars(r["p_value"])
        print("\n" + "=" * 72)
        print("Reverse Cross-Fitting DML  --  partially linear model")
        print("=" * 72)
        print(f"observations         : {self.blocks_.n_samples}")
        print(f"folds (K)            : {self.n_blocks}   "
              f"block size {self.blocks_.block_size}")
        print(f"stage-2 method       : {self.estimation_method}")
        print(f"HAC                  : {self.hac_kernel} "
              f"(rule '{self.hac_bandwidth_rule}')" if self.use_hac else "HAC: off")
        print("-" * 72)
        print(f"{'':<12}{'coef':>12}{'std err':>12}{'t':>10}{'P>|t|':>10}"
              f"{'':>4}")
        print(f"{'theta':<12}{r['coef']:>12.6f}{r['std_error']:>12.6f}"
              f"{r['t_stat']:>10.3f}{r['p_value']:>10.4f}{stars:>4}")
        pct = int(round(self.confidence_level * 100))
        print(f"{pct}% CI: [{r['ci_lower']:.6f}, {r['ci_upper']:.6f}]")
        if r.get("block_coefs"):
            coefs = ", ".join(f"{c:.4f}" for c in r["block_coefs"])
            print(f"fold estimates: [{coefs}]")
        print("=" * 72 + "\n")

    def to_frame(self) -> pd.DataFrame:
        """One-row :class:`pandas.DataFrame` with the headline result."""
        self._check_fitted()
        r = self.results_
        return pd.DataFrame([{
            "coefficient": r["coef"],
            "std_error": r["std_error"],
            "t_stat": r["t_stat"],
            "p_value": r["p_value"],
            "ci_lower": r["ci_lower"],
            "ci_upper": r["ci_upper"],
            "n_obs": self.blocks_.n_samples,
            "n_blocks": self.n_blocks,
        }])

    def block_frame(self) -> pd.DataFrame:
        """Per-fold estimates as a :class:`pandas.DataFrame`."""
        self._check_fitted()
        return pd.DataFrame(self.block_statistics_)

    def plot_structure(self, **kwargs):
        """
        Draw the fold diagram (paper, Figure 1).

        Thin wrapper around :func:`tsdml.plots.plot_block_structure`.
        """
        from .plots import plot_block_structure
        n = self.blocks_.n_samples if self.blocks_ is not None else None
        return plot_block_structure(n=n, K=self.n_blocks, **kwargs)

    # ------------------------------------------------------------ internals -- #

    def clear_cache(self) -> None:
        """Drop cached policy residuals, forcing a full stage one next fit."""
        self._cached_treatment_residuals = None
        self._cached_outcome_hash = None
        self._cached_treatment_hash = None
        self._cached_n = None

    @staticmethod
    def _as_array(data) -> np.ndarray:
        if hasattr(data, "values"):
            return np.asarray(data.values, dtype=float)
        return np.asarray(data, dtype=float)

    @staticmethod
    def _hash(arr: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()

    def _caching_strategy(self, y: np.ndarray, treatment: np.ndarray) -> str:
        if not self.cache_residuals:
            return "no_cache"
        if (self._cached_treatment_residuals is None
                or self._cached_n is None
                or len(y) != self._cached_n
                or len(treatment) != self._cached_n):
            return "fresh"
        same_outcome = self._hash(y) == self._cached_outcome_hash
        same_treatment = self._hash(treatment) == self._cached_treatment_hash
        if same_treatment and not same_outcome:
            return "reuse_treatment"
        return "fresh"

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise ValueError("estimator is not fitted; call fit() first")


def _stars(p: float) -> str:
    """Significance stars: ``***`` p<0.01, ``**`` p<0.05, ``*`` p<0.10."""
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""
