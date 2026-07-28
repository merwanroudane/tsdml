"""
Nuisance-parameter calibration: the Goldilocks zone.

Section 3 of Ciganovic, D'Amario and Tancioni (2026) shows that in
high-dimensional time series the hyperparameter minimising predictive error is
*not* the one minimising bias in the causal score: predictive tuning
over-shrinks the policy equation and attenuates the partialled-out signal.  The
proposed rule instead targets a locally stable region of the predictive error
profile.

Let :math:`\\Lambda = \\{\\lambda_1 < \\dots < \\lambda_M\\}` be an ordered grid
and :math:`\\mathcal R(\\lambda_i)` the validation RMSE on the auxiliary
validation block.  For window :math:`\\mathcal W_j = \\{j, \\dots, j+S-1\\}`,

.. math::

    \\bar{\\mathcal R}_j = \\frac1S \\sum_{i \\in \\mathcal W_j} \\mathcal R(\\lambda_i),
    \\qquad
    V_j = \\frac1S \\sum_{i \\in \\mathcal W_j}
          \\bigl(\\mathcal R(\\lambda_i) - \\bar{\\mathcal R}_j\\bigr)^2 ,

both min-max normalised across admissible windows to
:math:`\\tilde V_j, \\bar{\\mathcal R}^{\\sim}_j`, and scored by

.. math::

    \\mathcal S_j = \\tilde V_j + \\bar{\\mathcal R}^{\\sim}_j ,
    \\qquad j^\\star = \\arg\\min_j \\mathcal S_j ,
    \\qquad \\lambda^\\star = \\arg\\min_{\\lambda_i \\in \\mathcal W_{j^\\star}}
                              \\mathcal R(\\lambda_i).

The benchmark window size is :math:`S = 3`.

Crucially, the validation block is carved out of the *auxiliary* sample: the
main block :math:`B_k` is never used for tuning or for nuisance fitting, so
everything stays measurable with respect to
:math:`\\mathcal F_{\\mathrm{aux},k}` and the proof architecture of Theorem 2.1
survives.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid

from .folds import reverse_cf_folds

try:  # optional acceleration
    from joblib import Parallel, delayed
    _JOBLIB = True
except ImportError:  # pragma: no cover
    _JOBLIB = False

__all__ = ["Metric", "RMSE", "GoldilocksZone", "Calibrator", "goldilocks_select"]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

class Metric:
    """Base class for calibration metrics.  Lower is better."""

    is_batch_metric = False

    def __init__(self, name: str):
        self.name = name

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r})"


class RMSE(Metric):
    """
    Root mean squared error -- the standard predictive criterion.

    Examples
    --------
    >>> RMSE().calculate(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
    0.7071067811865476
    """

    def __init__(self):
        super().__init__("RMSE")

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def goldilocks_select(rmse_scores, window_size: int = 3) -> int:
    """
    Return the index of the hyperparameter selected by the Goldilocks rule.

    Parameters
    ----------
    rmse_scores : sequence of float
        Validation RMSE for each candidate, **in grid order** (the rule is
        local, so the ordering of the grid is part of the specification).
    window_size : int, default 3
        Window length ``S``.  The paper's benchmark is ``S = 3``.

    Returns
    -------
    int
        Index into ``rmse_scores``.

    Examples
    --------
    A grid with a sharp, unstable minimum at index 1 and a flat, stable region
    at indices 3-5: the rule declines the spike.

    >>> goldilocks_select([0.50, 0.20, 0.60, 0.31, 0.30, 0.31], window_size=3)
    4
    """
    scores = list(map(float, rmse_scores))
    N = len(scores)
    if N == 0:
        raise ValueError("rmse_scores is empty")
    if N == 1:
        return 0

    S = int(window_size)
    if S > N:  # grid shorter than the window: fall back to the plain minimum
        return int(np.argmin(scores))

    windows = []
    for start in range(N - S + 1):
        w = scores[start:start + S]
        windows.append((float(np.var(w)), float(np.mean(w)), start))

    variances = np.array([w[0] for w in windows])
    means = np.array([w[1] for w in windows])

    var_range = variances.max() - variances.min()
    mean_range = means.max() - means.min()
    norm_var = ((variances - variances.min()) / var_range
                if var_range > 1e-10 else np.zeros_like(variances))
    norm_mean = ((means - means.min()) / mean_range
                 if mean_range > 1e-10 else np.zeros_like(means))

    j_star = int(np.argmin(norm_var + norm_mean))
    start = windows[j_star][2]
    idx = list(range(start, start + S))
    return idx[int(np.argmin([scores[i] for i in idx]))]


class GoldilocksZone(Metric):
    """
    Stability-based tuning criterion (paper, Section 3).

    Scores individual candidates by RMSE, then selects across the grid with
    :func:`goldilocks_select`.  Flagged as a *batch* metric because selection
    needs the whole RMSE profile, not one candidate at a time.

    Parameters
    ----------
    window_size : int, default 3
        Window length ``S``.
    """

    is_batch_metric = True

    def __init__(self, window_size: int = 3):
        super().__init__("GoldilocksZone")
        self.window_size = int(window_size)

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def select_best(self, rmse_scores: List[float]) -> int:
        """Index of the selected candidate given the RMSE profile."""
        return goldilocks_select(rmse_scores, self.window_size)


def _resolve_metric(metric: Union[Metric, str, Any], window_size: int) -> Metric:
    if isinstance(metric, Metric):
        return metric
    if isinstance(metric, str):
        key = metric.lower().replace("-", "_")
        if key in ("rmse", "predictive"):
            return RMSE()
        if key in ("goldilocks", "goldilocks_zone", "gz"):
            return GoldilocksZone(window_size=window_size)
        raise ValueError(
            f"Unknown metric '{metric}'. Use 'rmse', 'goldilocks_zone', "
            f"a Metric instance, or a callable f(y_true, y_pred) -> float."
        )
    if callable(metric):
        wrapper = Metric(getattr(metric, "__name__", "custom"))
        wrapper.calculate = metric  # type: ignore[method-assign]
        return wrapper
    raise TypeError(f"Cannot interpret metric of type {type(metric)!r}")


# --------------------------------------------------------------------------- #
# Calibrator
# --------------------------------------------------------------------------- #

class Calibrator:
    """
    Fold-specific tuning of the outcome and policy nuisance learners.

    For every RCF fold ``k`` the auxiliary sample :math:`\\mathcal A_k` is split
    into a training part and a validation block :math:`\\mathcal V_k` *adjacent
    to the main block*: the ``L`` observations of the auxiliary sample closest
    to :math:`B_k`, where ``L = |B_k|``.  Candidates are trained on
    :math:`\\mathcal A_k \\setminus \\mathcal V_k` and scored on
    :math:`\\mathcal V_k`.  The main block is never touched.

    Parameters
    ----------
    metric : {'rmse', 'goldilocks_zone'} or Metric or callable, default 'goldilocks_zone'
        Selection criterion.  ``'goldilocks_zone'`` is the paper's proposal.
    n_blocks : int, default 5
        Number of RCF folds ``K``.
    stability_window_size : int, default 3
        Goldilocks window ``S``.
    verbose : bool, default False
        Print per-fold progress.
    n_jobs : int, default 1
        Parallel candidate evaluation (requires ``joblib``).  ``-1`` uses all
        cores.
    backend : {'threading', 'loky', 'multiprocessing'}, default 'threading'
        ``joblib`` backend.  ``'threading'`` avoids pickling scikit-learn
        estimators.
    random_state : int, optional
        Reserved; block construction is deterministic.

    Attributes
    ----------
    best_outcome_learners_ : list
        One selected (unfitted-parameters, fitted-on-training) learner per fold.
    best_treatment_learners_ : list
        Same for the policy equation.
    block_specific_learners_ : dict
        ``{'outcome_learners': [...], 'treatment_learners': [...]}`` -- pass
        straight to :class:`tsdml.rcf.ReverseCrossFitting` or
        :class:`tsdml.lp.DMLLocalProjections`.
    calibration_scores_ : dict
        Per-fold scores and selected hyperparameters.
    rmse_profiles_ : dict
        ``{k: {'outcome': [...], 'treatment': [...]}}`` -- the validation RMSE
        profile over the grid for every fold.  Plot it with
        :func:`tsdml.plots.plot_goldilocks_profile`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Lasso
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((120, 5))
    >>> d = X @ np.r_[1.0, 0.5, np.zeros(3)] + rng.standard_normal(120)
    >>> y = 2.0 * d + X @ np.r_[0.8, np.zeros(4)] + rng.standard_normal(120)
    >>> cal = Calibrator(metric='goldilocks_zone', n_blocks=4)
    >>> _ = cal.calibrate(X, y, d,
    ...                   outcome_learner_class=Lasso,
    ...                   outcome_param_grid={'alpha': [0.001, 0.01, 0.1, 1.0]},
    ...                   treatment_learner_class=Lasso,
    ...                   treatment_param_grid={'alpha': [0.001, 0.01, 0.1, 1.0]})
    >>> len(cal.best_outcome_learners_)
    4
    """

    def __init__(
        self,
        metric: Union[Metric, str, Any] = "goldilocks_zone",
        n_blocks: int = 5,
        random_state: Optional[int] = None,
        verbose: bool = False,
        n_jobs: int = 1,
        backend: str = "threading",
        stability_window_size: int = 3,
    ):
        self.n_blocks = int(n_blocks)
        if self.n_blocks < 2:
            raise ValueError("n_blocks must be at least 2")
        self.random_state = random_state
        self.verbose = bool(verbose)
        self.n_jobs = int(n_jobs)
        self.backend = backend
        self.stability_window_size = int(stability_window_size)
        self.metric = _resolve_metric(metric, self.stability_window_size)

        self.best_outcome_learners_: Optional[List[Any]] = None
        self.best_treatment_learners_: Optional[List[Any]] = None
        self.block_specific_learners_: Optional[Dict[str, List[Any]]] = None
        self.calibration_scores_: Dict[int, Dict[str, Any]] = {}
        self.rmse_profiles_: Dict[int, Dict[str, List[float]]] = {}

    # ---------------------------------------------------------------- API -- #

    def calibrate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        treatment: np.ndarray,
        outcome_learner_class: Optional[type] = None,
        treatment_learner_class: Optional[type] = None,
        outcome_param_grid: Optional[Dict[str, List]] = None,
        treatment_param_grid: Optional[Dict[str, List]] = None,
        outcome_estimator_grid: Optional[List[Any]] = None,
        treatment_estimator_grid: Optional[List[Any]] = None,
        outcome_estimators_with_params: Optional[List[Tuple[type, Dict[str, List]]]] = None,
        treatment_estimators_with_params: Optional[List[Tuple[type, Dict[str, List]]]] = None,
    ) -> Dict[str, Any]:
        """
        Select one outcome and one policy learner per fold.

        Candidates may be supplied in three ways, per equation:

        1. ``learner_class`` + ``param_grid`` (expanded with ``ParameterGrid``);
        2. ``estimator_grid`` -- a list of pre-configured instances;
        3. ``estimators_with_params`` -- a list of ``(class, param_grid)``
           pairs, to mix model families.

        Returns
        -------
        dict
            ``block_specific_learners``, ``calibration_scores``,
            ``best_params_outcome``, ``best_params_treatment``,
            ``rmse_profiles``.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        treatment = np.asarray(treatment, dtype=float).ravel()
        if not (X.shape[0] == y.shape[0] == treatment.shape[0]):
            raise ValueError("X, y and treatment must have the same length")

        outcome_grid = self._build_grid(
            outcome_learner_class, outcome_param_grid, outcome_estimator_grid,
            "outcome", outcome_estimators_with_params,
        )
        treatment_grid = self._build_grid(
            treatment_learner_class, treatment_param_grid, treatment_estimator_grid,
            "treatment", treatment_estimators_with_params,
        )

        blocks = reverse_cf_folds(X.shape[0], self.n_blocks)

        if self.verbose:
            print("=" * 78)
            print(f"tsdml Calibrator  |  metric = {self.metric.name}")
            print("=" * 78)
            print(f"folds K            : {self.n_blocks}")
            print(f"outcome candidates : {len(outcome_grid)}")
            print(f"policy candidates  : {len(treatment_grid)}")

        best_out, best_trt = [], []
        params_out, params_trt = [], []

        for k in range(self.n_blocks):
            train_idx, val_idx = self._train_val_indices(k, blocks)
            if self.verbose:
                print(f"\n-- fold {k + 1}/{self.n_blocks} "
                      f"(train {len(train_idx)}, validate {len(val_idx)})")

            o_est, o_score, o_params, o_profile = self._calibrate_one(
                X, y, train_idx, val_idx, outcome_grid, "outcome")
            t_est, t_score, t_params, t_profile = self._calibrate_one(
                X, treatment, train_idx, val_idx, treatment_grid, "treatment")

            best_out.append(o_est)
            best_trt.append(t_est)
            params_out.append(o_params)
            params_trt.append(t_params)

            self.calibration_scores_[k] = {
                "outcome_score": o_score,
                "treatment_score": t_score,
                "outcome_params": o_params,
                "treatment_params": t_params,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
            }
            self.rmse_profiles_[k] = {"outcome": o_profile, "treatment": t_profile}

            if self.verbose:
                print(f"   outcome  RMSE {o_score:.6f}")
                print(f"   policy   RMSE {t_score:.6f}")

        self.best_outcome_learners_ = best_out
        self.best_treatment_learners_ = best_trt
        self.block_specific_learners_ = {
            "outcome_learners": best_out,
            "treatment_learners": best_trt,
        }
        return {
            "block_specific_learners": self.block_specific_learners_,
            "calibration_scores": self.calibration_scores_,
            "best_params_outcome": params_out,
            "best_params_treatment": params_trt,
            "rmse_profiles": self.rmse_profiles_,
        }

    def get_block_specific_learners(self) -> Dict[str, List[Any]]:
        """Return the per-fold learner dict, raising if not calibrated yet."""
        if self.block_specific_learners_ is None:
            raise ValueError("call calibrate() first")
        return self.block_specific_learners_

    def summary(self) -> None:
        """Print a per-fold table of selected scores and key hyperparameters."""
        if self.block_specific_learners_ is None:
            print("Calibrator has not been fitted. Call calibrate() first.")
            return
        print("\n" + "=" * 78)
        print("CALIBRATION SUMMARY")
        print("=" * 78)
        print(f"{'fold':<6}{'n_train':>9}{'n_val':>7}"
              f"{'outcome RMSE':>16}{'policy RMSE':>15}")
        print("-" * 78)
        for k in range(self.n_blocks):
            row = self.calibration_scores_.get(k, {})
            print(f"{k:<6}{row.get('n_train', 0):>9}{row.get('n_val', 0):>7}"
                  f"{row.get('outcome_score', float('nan')):>16.6f}"
                  f"{row.get('treatment_score', float('nan')):>15.6f}")
        print("=" * 78 + "\n")

    def selected_params_frame(self):
        """
        Per-fold selected hyperparameters as a tidy :class:`pandas.DataFrame`.

        Only parameters that actually vary across the grid are reported, so a
        Lasso grid over ``alpha`` gives a compact two-column table.
        """
        import pandas as pd

        rows = []
        for k in range(self.n_blocks):
            row = self.calibration_scores_.get(k, {})
            entry = {"fold": k}
            for eq, key in (("outcome", "outcome_params"), ("policy", "treatment_params")):
                params = row.get(key) or {}
                for name in ("alpha", "l1_ratio", "n_estimators", "max_depth", "C"):
                    if name in params:
                        entry[f"{eq}_{name}"] = params[name]
                entry[f"{eq}_rmse"] = row.get(
                    "outcome_score" if eq == "outcome" else "treatment_score")
            rows.append(entry)
        return pd.DataFrame(rows)

    # ----------------------------------------------------------- internals -- #

    @staticmethod
    def _build_grid(learner_class, param_grid, estimator_grid, name,
                    estimators_with_params) -> List[Any]:
        if estimators_with_params is not None:
            out: List[Any] = []
            for cls, grid in estimators_with_params:
                for params in ParameterGrid(grid):
                    out.append(cls(**params))
            return out
        if estimator_grid is not None:
            return list(estimator_grid)
        if learner_class is None or param_grid is None:
            raise ValueError(
                f"provide estimator_grid, estimators_with_params, or "
                f"(learner_class, param_grid) for the {name} equation"
            )
        return [learner_class(**params) for params in ParameterGrid(param_grid)]

    def _train_val_indices(self, k: int, blocks) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validation block = the ``L`` auxiliary observations adjacent to ``B_k``.

        The main block is excluded from both training and validation, so tuning
        remains measurable with respect to the auxiliary sigma-field.
        """
        L = len(blocks.main_blocks[k])
        aux_right = sorted(blocks.aux_right_blocks[k])
        aux_left = sorted(blocks.aux_left_blocks[k])
        K = self.n_blocks

        if k <= K // 2 - 1:
            val, train = aux_right[:L], aux_right[L:]
        elif k == K // 2 and K % 2 == 1:
            if aux_right:
                val, train = aux_right[:L], aux_right[L:] + aux_left
            else:
                val, train = aux_left[-L:], aux_left[:-L]
        else:
            val, train = aux_left[-L:], aux_left[:-L]

        if len(train) == 0:
            raise ValueError(
                f"fold {k}: no training observations left after carving out a "
                f"validation block of length {L}. Reduce K or extend the sample."
            )
        return np.asarray(train), np.asarray(val)

    def _calibrate_one(self, X, target, train_idx, val_idx, grid, name):
        X_tr, y_tr = X[train_idx], target[train_idx]
        X_va, y_va = X[val_idx], target[val_idx]

        def evaluate(est):
            try:
                fitted = clone(est)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted.fit(X_tr, y_tr)
                score = float(self.metric.calculate(y_va, fitted.predict(X_va)))
                params = est.get_params() if hasattr(est, "get_params") else None
                return {"ok": True, "score": score, "estimator": fitted, "params": params}
            except Exception as exc:  # pragma: no cover - defensive
                return {"ok": False, "score": np.inf, "estimator": None,
                        "params": None, "error": str(exc)}

        if self.n_jobs != 1 and _JOBLIB:
            results = Parallel(n_jobs=self.n_jobs, backend=self.backend)(
                delayed(evaluate)(est) for est in grid)
        else:
            results = [evaluate(est) for est in grid]

        valid = [r for r in results if r["ok"]]
        if not valid:
            raise ValueError(f"every candidate failed for the {name} equation")
        if len(valid) < len(results):
            warnings.warn(f"{len(results) - len(valid)} {name} candidates failed and "
                          f"were dropped from the grid")

        profile = [r["score"] for r in valid]

        if getattr(self.metric, "is_batch_metric", False):
            best_idx = self.metric.select_best(profile)
        else:
            best_idx = int(np.argmin(profile))

        best = valid[best_idx]
        return best["estimator"], float(best["score"]), best["params"], profile
