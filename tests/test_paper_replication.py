"""
End-to-end replication of the paper's empirical application (Section 5).

These are the regression tests that pin the package to published numbers.  The
targets below were produced by the authors' own replication package and verified
to agree with :mod:`tsdml` to machine precision across the whole pipeline --
data preparation, Goldilocks calibration, reverse cross-fitting, stage two and
IRF scaling.

The full five-outcome run takes a couple of minutes, so it is marked ``slow``::

    pytest -m "not slow"     # skip it
    pytest                   # run everything
"""

import numpy as np
import pytest
from sklearn.linear_model import Lasso

from tsdml import (
    Calibrator,
    DataProcessor,
    DMLLocalProjections,
    ReverseCrossFitting,
    load_macroprudential,
    macroprudential_spec,
    scale_irfs,
)

SPEC = macroprudential_spec()
GRID = {"alpha": np.linspace(0.000001, 0.1, 100),
        "max_iter": [100000], "random_state": [42]}


@pytest.fixture(scope="module")
def data():
    return load_macroprudential().drop(columns=SPEC["drop"])


def _prepare(df, outcome, code):
    proc = DataProcessor()
    X, y, d, leads = proc.data_prep(
        df=df, num_lags=SPEC["num_lags"], H=SPEC["H"],
        treatment_var=SPEC["treatment_var"], treatment_code=SPEC["treatment_code"],
        outcome_var=outcome, outcome_code=code,
        start_date=SPEC["start_date"], scaling_method="none",
        scale_outcome_treatment=False, include_constant=True, K=SPEC["n_blocks"])
    return proc, X, y, d, leads


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def test_bundled_data_shape(data):
    raw = load_macroprudential()
    assert raw.shape == (73, 37)
    assert raw.index[0] == "speed"
    assert raw.index[1] == "Transform:"
    assert data.shape[1] == 37 - len(SPEC["drop"])


def test_estimation_sample_matches_the_paper(data):
    proc, X, y, d, leads = _prepare(data, "GDP_K2020", 5)
    assert X.shape == (54, 105)
    assert len(y) == len(d) == 54
    assert leads.shape == (54, 9)
    assert len(X) % SPEC["n_blocks"] == 0
    assert str(proc.original_index[0].date()) == "2006-12-31"
    assert str(proc.original_index[-1].date()) == "2022-03-31"


def test_policy_variable_enters_only_with_lags(data):
    """Slow-moving banking variables must not be in the contemporaneous block."""
    proc, X, _, _, _ = _prepare(data, "GDP_K2020", 5)
    treat = proc.names_["treatment"]
    contemporaneous = [c for c in proc.feature_names_ if not c.startswith("lag")]
    assert treat not in contemporaneous
    assert any(c.startswith("lag1_") and c.endswith(treat)
               for c in proc.feature_names_)


# --------------------------------------------------------------------------- #
# fixed-learner reverse cross-fitting -- exact regression target
# --------------------------------------------------------------------------- #

def test_static_rcf_reproduces_the_replication_value(data):
    _, X, y, d, _ = _prepare(data, "GDP_K2020", 5)
    lasso = lambda: Lasso(alpha=0.01, max_iter=100000, random_state=42)  # noqa: E731
    est = ReverseCrossFitting(lasso(), lasso(), n_blocks=6,
                              estimation_method="block").fit(X, y, d)
    assert est.theta_ == pytest.approx(-0.28805862663087, rel=1e-10)
    assert est.results_["std_error"] == pytest.approx(0.5320894689645613, rel=1e-10)


def test_goldilocks_selects_the_published_penalties(data):
    _, X, y, d, _ = _prepare(data, "GDP_K2020", 5)
    cal = Calibrator(metric="goldilocks_zone", n_blocks=6)
    cal.calibrate(X, y, d,
                  outcome_learner_class=Lasso, outcome_param_grid=dict(GRID),
                  treatment_learner_class=Lasso, treatment_param_grid=dict(GRID))
    outcome_alphas = [lrn.alpha for lrn in cal.best_outcome_learners_]
    policy_alphas = [lrn.alpha for lrn in cal.best_treatment_learners_]
    assert np.allclose(outcome_alphas,
                       [0.1, 0.1, 0.1, 0.00101109, 0.01919294, 0.1], atol=1e-7)
    assert np.allclose(policy_alphas,
                       [0.1, 0.1, 0.1, 0.08888889, 0.07474747, 0.00707172], atol=1e-7)


# --------------------------------------------------------------------------- #
# full local-projection pipeline
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_figure_two_cumulative_responses(data):
    """Reproduce the scaled IRFs behind the paper's Figure 2."""
    irfs = {}
    for outcome, code in SPEC["outcomes"]:
        proc, X, y, d, leads = _prepare(data, outcome, code)
        cal = Calibrator(metric="goldilocks_zone", n_blocks=SPEC["n_blocks"])
        cal.calibrate(X, y, d,
                      outcome_learner_class=Lasso, outcome_param_grid=dict(GRID),
                      treatment_learner_class=Lasso, treatment_param_grid=dict(GRID))
        lp = DMLLocalProjections(
            block_specific_learners=cal.block_specific_learners_,
            n_blocks=SPEC["n_blocks"], estimation_method="block",
            outcome_code=code, outcome_name=outcome)
        lp.fit(X, y, d, leads, index=proc.original_index)
        irfs[outcome] = lp.to_frame()

    scaled = scale_irfs(irfs)

    # the normalisation is a 50 basis-point rise in the Tier 1 ratio on impact
    impact = (scaled["Tier 1 capital"]["Coefficient"].iloc[0]
              - scaled["Risk-weighted assets"]["Coefficient"].iloc[0])
    assert impact == pytest.approx(0.5)

    gdp = scaled["GDP_K2020"]["Coefficient"].values
    spread = scaled["PNFC_Spread"]["Coefficient"].values
    lending = scaled["PNFC_Lending_K2020"]["Coefficient"].values

    # published values, to 4 decimals
    assert gdp[:3] == pytest.approx([-0.0360, -0.0995, -0.1438], abs=1e-4)
    assert spread[0] == pytest.approx(1.0455, abs=1e-4)
    assert lending[5] == pytest.approx(-0.3632, abs=1e-4)

    # the qualitative findings of Section 5.3
    assert gdp.min() < -0.14           # GDP falls by almost 0.15 percent
    assert int(np.argmin(gdp)) == 2    # trough two quarters out
    assert spread[0] > 0               # spreads rise on impact
    assert lending[1:].min() < 0       # lending contracts
