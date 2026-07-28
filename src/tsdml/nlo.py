"""
Neighbors-Left-Out cross-fitting (Semenova et al., 2023) -- the benchmark.

Same partially linear model and same stage two as
:class:`tsdml.rcf.ReverseCrossFitting`, but the auxiliary sample for main block
``k`` deletes block ``k`` together with its two adjacent neighbours.  Buffer
deletion buys approximate independence between training and test folds; the
price is sample use, which is what the paper's simulations quantify:

>>> from tsdml.folds import sample_use_rcf, sample_use_nlo
>>> round(sample_use_rcf(6), 3), round(sample_use_nlo(6), 3)
(0.667, 0.556)

Keep this estimator for two purposes: reproducing the paper's RCF-vs-NLO
comparison, and as the fallback when residual diagnostics say adjacent
auxiliary blocks leak into the main block (see
:func:`tsdml.diagnostics.leakage_test`), where conditional stability fails and
buffering becomes the safer design.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import clone

from .folds import nlo_folds, sample_use_nlo
from .stage2 import block_statistics, fit_stage_two
from .rcf import _stars

__all__ = ["NLOCrossFitting"]


class NLOCrossFitting:
    """
    Neighbors-left-out cross-fitted DML estimator.

    Parameters mirror :class:`tsdml.rcf.ReverseCrossFitting`; see that class for
    details.  ``K >= 4`` is recommended: with ``K = 3`` the middle fold's
    auxiliary sample is empty once both neighbours are deleted.

    Attributes
    ----------
    theta_ : float
    results_ : dict
    residuals_ : dict
    main_blocks_, aux_blocks_ : list of list of int

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LassoCV
    >>> rng = np.random.default_rng(3)
    >>> X = rng.standard_normal((240, 8))
    >>> d = X[:, 0] + rng.standard_normal(240)
    >>> y = 1.0 * d + 0.5 * X[:, 0] + rng.standard_normal(240)
    >>> est = NLOCrossFitting(LassoCV(), LassoCV(), n_blocks=6).fit(X, y, d)
    >>> bool(abs(est.theta_ - 1.0) < 0.3)
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
        random_state: Optional[int] = None,
    ):
        if estimation_method not in ("block", "full"):
            raise ValueError("estimation_method must be 'block' or 'full'")
        if n_blocks < 4:
            raise ValueError(
                "NLO needs n_blocks >= 4; with fewer folds the auxiliary "
                "sample of an interior block is empty after deleting neighbours"
            )

        self.n_blocks = int(n_blocks)
        self.estimation_method = estimation_method
        self.include_constant = bool(include_constant)
        self.use_hac = bool(use_hac)
        self.confidence_level = float(confidence_level)
        self.hac_kernel = hac_kernel
        self.hac_bandwidth_rule = hac_bandwidth_rule
        self.hac_bandwidth_value = hac_bandwidth_value
        self.use_fixed_b_critical = bool(use_fixed_b_critical)
        self.random_state = random_state

        if block_specific_learners is not None:
            self.outcome_learners = list(block_specific_learners["outcome_learners"])
            self.treatment_learners = list(block_specific_learners["treatment_learners"])
            if len(self.outcome_learners) != n_blocks or len(self.treatment_learners) != n_blocks:
                raise ValueError(f"block_specific_learners must supply {n_blocks} learners")
        else:
            if outcome_learner is None or treatment_learner is None:
                raise ValueError("provide both learners, or block_specific_learners")
            self.outcome_learners = [outcome_learner] * n_blocks
            self.treatment_learners = [treatment_learner] * n_blocks

        self.theta_: Optional[float] = None
        self.results_: Optional[Dict[str, Any]] = None
        self.residuals_: Optional[Dict[str, np.ndarray]] = None
        self.main_blocks_: Optional[List[List[int]]] = None
        self.aux_blocks_: Optional[List[List[int]]] = None
        self.block_statistics_: Optional[List[Dict[str, Any]]] = None
        self.is_fitted_ = False

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        treatment: Union[np.ndarray, pd.Series],
        verbose: bool = False,
    ) -> "NLOCrossFitting":
        """Fit stage one on buffered auxiliary samples, then stage two."""
        X = np.asarray(getattr(X, "values", X), dtype=float)
        y = np.asarray(getattr(y, "values", y), dtype=float).ravel()
        treatment = np.asarray(getattr(treatment, "values", treatment), dtype=float).ravel()

        T = X.shape[0]
        if not (len(y) == len(treatment) == T):
            raise ValueError("X, y and treatment must have the same length")

        main, aux = nlo_folds(T, self.n_blocks)
        self.main_blocks_, self.aux_blocks_ = main, aux

        chi = np.zeros(T)
        xi = np.zeros(T)

        if verbose:
            print(f"NLO-DML: T={T}, K={self.n_blocks}, "
                  f"sample use={sample_use_nlo(self.n_blocks):.3f}")

        for k in range(self.n_blocks):
            m_idx = np.asarray(main[k])
            a_idx = np.asarray(aux[k])
            if a_idx.size == 0:
                raise ValueError(f"fold {k}: empty auxiliary sample; increase n_blocks")

            g = clone(self.outcome_learners[k])
            g.fit(X[a_idx], y[a_idx])
            chi[m_idx] = y[m_idx] - g.predict(X[m_idx])

            m = clone(self.treatment_learners[k])
            m.fit(X[a_idx], treatment[a_idx])
            xi[m_idx] = treatment[m_idx] - m.predict(X[m_idx])

            if verbose:
                print(f"  fold {k}: aux n={a_idx.size}, main n={m_idx.size}")

        self.residuals_ = {"outcome": chi, "treatment": xi}
        self.results_ = fit_stage_two(
            outcome_residuals=chi,
            treatment_residuals=xi,
            estimation_method=self.estimation_method,
            n_blocks=self.n_blocks,
            main_blocks=main,
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
            chi, xi, main, include_constant=self.include_constant,
            use_hac=self.use_hac, hac_lag=1 if self.use_hac else None,
            confidence_level=self.confidence_level, hac_kernel=self.hac_kernel,
        )
        self.is_fitted_ = True
        if verbose:
            self.summary()
        return self

    def get_residuals(self) -> Dict[str, np.ndarray]:
        """Out-of-sample residuals."""
        if not self.is_fitted_:
            raise ValueError("estimator is not fitted; call fit() first")
        return self.residuals_

    def summary(self) -> None:
        """Print a compact estimation table."""
        if not self.is_fitted_:
            raise ValueError("estimator is not fitted; call fit() first")
        r = self.results_
        print("\n" + "=" * 72)
        print("Neighbors-Left-Out DML  --  partially linear model")
        print("=" * 72)
        print(f"folds (K)            : {self.n_blocks}")
        print(f"sample use           : {sample_use_nlo(self.n_blocks):.4f}")
        print("-" * 72)
        print(f"{'':<12}{'coef':>12}{'std err':>12}{'t':>10}{'P>|t|':>10}")
        print(f"{'theta':<12}{r['coef']:>12.6f}{r['std_error']:>12.6f}"
              f"{r['t_stat']:>10.3f}{r['p_value']:>10.4f}{_stars(r['p_value']):>4}")
        pct = int(round(self.confidence_level * 100))
        print(f"{pct}% CI: [{r['ci_lower']:.6f}, {r['ci_upper']:.6f}]")
        print("=" * 72 + "\n")

    def to_frame(self) -> pd.DataFrame:
        """One-row summary frame."""
        if not self.is_fitted_:
            raise ValueError("estimator is not fitted; call fit() first")
        r = self.results_
        return pd.DataFrame([{
            "coefficient": r["coef"], "std_error": r["std_error"],
            "t_stat": r["t_stat"], "p_value": r["p_value"],
            "ci_lower": r["ci_lower"], "ci_upper": r["ci_upper"],
            "n_blocks": self.n_blocks,
        }])
