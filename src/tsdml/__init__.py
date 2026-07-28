"""
tsdml -- Double Machine Learning for time series.

A faithful implementation of Ciganovic, D'Amario and Tancioni (2026),
*Double Machine Learning for Time Series*, The Econometrics Journal
(arXiv:2603.10999).

The package supplies the two contributions of the paper and the plumbing around
them:

**Reverse Cross-Fitting (RCF)** -- :class:`ReverseCrossFitting`
    Deterministic block cross-fitting that trains nuisance functions on
    time-reversed auxiliary blocks and deletes no buffer between the auxiliary
    and main samples, so it uses more of a short macro sample than
    neighbor-deletion designs at moderate ``K``.

**Goldilocks-zone tuning** -- :class:`Calibrator`
    Selects the nuisance hyperparameter inside a locally stable region of the
    validation-error profile rather than at its global minimum, which reduces
    small-sample bias in the causal score when the predictive optimum
    over-shrinks the policy equation.

Plus: :class:`DMLLocalProjections` for dynamic responses,
:class:`NLOCrossFitting` as the benchmark design, :mod:`tsdml.diagnostics` for
the conditional-stability screen, and :mod:`tsdml.plots` / :mod:`tsdml.tables`
for journal-ready output.

Quick start
-----------
>>> import numpy as np
>>> from sklearn.linear_model import Lasso
>>> from tsdml import Calibrator, ReverseCrossFitting, simulate_plr
>>> sim = simulate_plr(T=180, p=40, theta=1.0, seed=0)
>>> cal = Calibrator(metric='goldilocks_zone', n_blocks=6).calibrate(
...     sim['X'], sim['y'], sim['d'],
...     outcome_learner_class=Lasso,
...     outcome_param_grid={'alpha': np.linspace(1e-4, 0.3, 25), 'max_iter': [50_000]},
...     treatment_learner_class=Lasso,
...     treatment_param_grid={'alpha': np.linspace(1e-4, 0.3, 25), 'max_iter': [50_000]})
>>> est = ReverseCrossFitting(
...     n_blocks=6, block_specific_learners=cal['block_specific_learners']
... ).fit(sim['X'], sim['y'], sim['d'])
>>> bool(abs(est.theta_ - 1.0) < 0.2)
True

Full walkthrough: see ``docs/STEP_BY_STEP_GUIDE.md`` in the repository.
"""

from __future__ import annotations

from .calibration import Calibrator, GoldilocksZone, Metric, RMSE, goldilocks_select
from .datasets import (
    load_macroprudential,
    macroprudential_spec,
    simulate_plr,
    simulate_svar,
)
from .diagnostics import (
    boundary_leakage_test,
    diagnose,
    residual_autocorrelation_test,
)
from .folds import (
    BlockStructure,
    fold_direction,
    nlo_folds,
    reverse_cf_folds,
    sample_use_nlo,
    sample_use_rcf,
)
from .hac import compute_hac_bandwidth, fixed_b_critical_value, hac_lrv
from .lp import DMLLocalProjections, cumulate_residuals, scale_irfs
from .nlo import NLOCrossFitting
from .plots import (
    plot_block_structure,
    plot_goldilocks_profile,
    plot_irf,
    plot_irf_comparison,
    plot_irf_panel,
    plot_residuals,
    plot_sample_use,
)
from .prep import DataProcessor, TRANSFORM_CODES, transform_series
from .rcf import ReverseCrossFitting
from .stage2 import fit_stage_two
from .style import use_journal_style
from .tables import (
    calibration_table,
    estimation_table,
    irf_table,
    sample_use_table,
    to_latex,
    to_markdown,
)

__version__ = "0.1.0"
__author__ = "Merwan Roudane"
__email__ = "merwanroudane920@gmail.com"

#: Reference for the method implemented here.
CITATION = (
    "Ciganovic, M., D'Amario, F. and Tancioni, M. (2026). "
    "Double Machine Learning for Time Series. The Econometrics Journal. "
    "arXiv:2603.10999."
)

__all__ = [
    # estimators
    "ReverseCrossFitting",
    "NLOCrossFitting",
    "DMLLocalProjections",
    # tuning
    "Calibrator",
    "GoldilocksZone",
    "RMSE",
    "Metric",
    "goldilocks_select",
    # folds and inference
    "BlockStructure",
    "reverse_cf_folds",
    "nlo_folds",
    "fold_direction",
    "sample_use_rcf",
    "sample_use_nlo",
    "hac_lrv",
    "compute_hac_bandwidth",
    "fixed_b_critical_value",
    "fit_stage_two",
    "cumulate_residuals",
    "scale_irfs",
    # data
    "DataProcessor",
    "TRANSFORM_CODES",
    "transform_series",
    "load_macroprudential",
    "macroprudential_spec",
    "simulate_svar",
    "simulate_plr",
    # diagnostics
    "diagnose",
    "boundary_leakage_test",
    "residual_autocorrelation_test",
    # output
    "use_journal_style",
    "plot_irf",
    "plot_irf_panel",
    "plot_irf_comparison",
    "plot_block_structure",
    "plot_goldilocks_profile",
    "plot_sample_use",
    "plot_residuals",
    "irf_table",
    "estimation_table",
    "calibration_table",
    "sample_use_table",
    "to_latex",
    "to_markdown",
    # meta
    "__version__",
    "CITATION",
]
