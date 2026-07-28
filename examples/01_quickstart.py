"""
tsdml quickstart -- estimate one causal parameter in a high-dimensional time series.

Run:
    python examples/01_quickstart.py

The five steps every tsdml analysis follows:

    1. get time-ordered X, y, d
    2. tune the nuisance learners per fold  (Calibrator)
    3. cross-fit and estimate                (ReverseCrossFitting)
    4. read the result                       (summary / estimation_table)
    5. check the maintained assumption       (diagnostics)

Step 5 is not optional. Reverse cross-fitting deletes no buffer between the
auxiliary and main samples, so its validity rests on conditional stability:
after conditioning on X_t, neighbouring observations must not predict the
main-block residual innovations. This script shows the check passing on a
well-specified design and failing on one built to violate it.
"""

import warnings

import numpy as np
from sklearn.linear_model import Lasso

from tsdml import (
    Calibrator,
    ReverseCrossFitting,
    boundary_leakage_test,
    estimation_table,
    simulate_plr,
)

warnings.filterwarnings("ignore")

K = 6                                    # folds
PENALTIES = {"alpha": np.linspace(1e-6, 0.4, 100),   # wide enough that the
             "max_iter": [100_000], "random_state": [42]}   # optimum is interior


def fit(X, y, d, metric):
    """Tune per fold, then cross-fit and estimate."""
    calibrator = Calibrator(metric=metric, n_blocks=K)
    calibrator.calibrate(
        X, y, d,
        outcome_learner_class=Lasso, outcome_param_grid=dict(PENALTIES),
        treatment_learner_class=Lasso, treatment_param_grid=dict(PENALTIES),
    )
    model = ReverseCrossFitting(
        n_blocks=K,
        block_specific_learners=calibrator.block_specific_learners_,
        estimation_method="block",
    ).fit(X, y, d)
    return calibrator, model


# --------------------------------------------------------------------------- #
# 1. data -- 200 periods, 100 controls, true effect 1.5
# --------------------------------------------------------------------------- #
sim = simulate_plr(T=200, p=100, theta=1.5, rho=0.5, seed=0)
X, y, d, truth = sim["X"], sim["y"], sim["d"], sim["theta"]

print(f"X {X.shape}   y {y.shape}   d {d.shape}")
print(f"true theta = {truth},  p/T = {X.shape[1] / X.shape[0]:.2f} "
      f"(high-dimensional)\n")

# --------------------------------------------------------------------------- #
# 2-3. tune and estimate
# --------------------------------------------------------------------------- #
calibrator, goldilocks = fit(X, y, d, "goldilocks_zone")
calibrator.summary()
print("penalty selected per fold:")
print(calibrator.selected_params_frame().round(5).to_string(index=False))

goldilocks.summary()

# the same pipeline with plain predictive tuning, for contrast
_, predictive = fit(X, y, d, "rmse")

# --------------------------------------------------------------------------- #
# 4. read the result
# --------------------------------------------------------------------------- #
print(estimation_table({"RCF (Goldilocks)": goldilocks,
                        "RCF (RMSE)": predictive}).to_string())
print(f"\ntrue theta = {truth}")
for label, model in (("Goldilocks", goldilocks), ("RMSE", predictive)):
    r = model.results_
    inside = "covers" if r["ci_lower"] <= truth <= r["ci_upper"] else "MISSES"
    print(f"  {label:<11} {model.theta_:.4f}   error {model.theta_ - truth:+.4f}   "
          f"95% CI {inside} the truth")
print("""
One draw tells you little: the sampling error of theta-hat here is about the
size of the reported standard error, so both estimates sit a fraction of a
standard error from 1.5. Averaged over draws the bias of either rule on this
well-specified design is 1-2 percent. The paper's bias gap between the two
tuning rules opens up in short, mis-specified, high-dimensional samples --
see examples/03_tuning_and_design_comparison.py for the Monte Carlo.""")

# --------------------------------------------------------------------------- #
# 5. conditional stability
# --------------------------------------------------------------------------- #


def leakage_pvalue(model):
    out = boundary_leakage_test(model.residuals_["treatment"],
                                n_blocks=model.n_blocks, max_lag=2)
    row = out.loc[out["fold"] == "pooled", "p_value"]
    return float(row.iloc[0]) if len(row) else float("nan")


print("\n" + "=" * 72)
print("Conditional stability check (Assumption 2.4)")
print("=" * 72)
print(f"well-specified design          : p = {leakage_pvalue(goldilocks):.4f}")

# now break it on purpose: give the disturbances their own AR(1) dynamics, so
# lagged residuals predict current ones even after conditioning on X
broken = simulate_plr(T=200, p=100, theta=1.5, rho=0.5,
                     resid_persistence=0.7, seed=0)
_, broken_fit = fit(broken["X"], broken["y"], broken["d"], "goldilocks_zone")
print(f"persistent-disturbance design  : p = {leakage_pvalue(broken_fit):.4f}")

print("""
A small p-value says neighbouring blocks still predict the main-block residuals.
Two remedies, in the order the paper suggests them:

  1. enrich the conditioning set -- more lags, factor proxies, regime
     indicators -- so X_t absorbs the predictable component;
  2. if the leakage is short-memory and will not go away, switch to
     NLOCrossFitting, which deletes a buffer around each main block and pays
     for it in sample use.
""")
