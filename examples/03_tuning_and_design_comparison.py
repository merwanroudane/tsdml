"""
Why the two contributions of the paper matter, on data you can regenerate.

Two experiments:

  A. Tuning rule -- Goldilocks zone versus predictive RMSE.
     Section 3: in high dimensions the RMSE-optimal penalty over-shrinks the
     policy equation, attenuates the partialled-out signal, and leaves bias in
     the causal score. Stability-based tuning reduces it.

  B. Cross-fitting design -- Reverse Cross-Fitting versus Neighbors-Left-Out.
     Section 2.1: RCF keeps the buffer blocks that NLO deletes, so at moderate
     K it trains on more of a short sample.

Run:
    python examples/03_tuning_and_design_comparison.py

Writes ``output/comparison_bias.pdf`` and prints a Monte Carlo table.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

from tsdml import (
    Calibrator,
    NLOCrossFitting,
    ReverseCrossFitting,
    plot_sample_use,
    sample_use_nlo,
    sample_use_rcf,
    simulate_svar,
)

warnings.filterwarnings("ignore")

OUT = Path("output")
OUT.mkdir(exist_ok=True)

REPS = 60            # raise to 1000+ for publication-grade numbers
T = 60               # short sample, the regime the paper targets
N_VARS = 40          # yields p = 78 controls, so p/T > 1: high-dimensional
K = 6
THETA = 0.5

# The correctly specified ordering puts the policy variable immediately before
# the outcome, so the PLR estimand equals THETA and "bias" is interpretable.
# Switch to 'misspecified' for the paper's stress test, where part of any gap
# is an estimand wedge rather than estimator bias.
SPECIFICATION = "specified"

PENALTIES = {"alpha": np.linspace(1e-6, 0.2, 60),
             "max_iter": [50_000], "random_state": [42]}


def tuned_learners(X, y, d, metric):
    cal = Calibrator(metric=metric, n_blocks=K)
    cal.calibrate(
        X, y, d,
        outcome_learner_class=Lasso, outcome_param_grid=dict(PENALTIES),
        treatment_learner_class=Lasso, treatment_param_grid=dict(PENALTIES),
    )
    return cal.block_specific_learners_


# --------------------------------------------------------------------------- #
# Monte Carlo
# --------------------------------------------------------------------------- #
records = []

for rep in range(REPS):
    sim = simulate_svar(n=N_VARS, T=T, theta=THETA, specification=SPECIFICATION,
                        seed=rep, seed_err=7000 + rep)
    X, y, d = sim["X"], sim["y"], sim["d"]

    for label, metric in (("RCF-GZ", "goldilocks_zone"), ("RCF-RMSE", "rmse")):
        learners = tuned_learners(X, y, d, metric)
        est = ReverseCrossFitting(
            n_blocks=K, block_specific_learners=learners).fit(X, y, d)
        r = est.results_
        records.append({
            "design": label, "rep": rep, "theta": est.theta_,
            "covered": int(r["ci_lower"] <= THETA <= r["ci_upper"]),
        })

    learners = tuned_learners(X, y, d, "rmse")
    est = NLOCrossFitting(n_blocks=K, block_specific_learners=learners).fit(X, y, d)
    r = est.results_
    records.append({
        "design": "NLO-RMSE", "rep": rep, "theta": est.theta_,
        "covered": int(r["ci_lower"] <= THETA <= r["ci_upper"]),
    })

    if (rep + 1) % 5 == 0:
        print(f"  {rep + 1}/{REPS} replications")

frame = pd.DataFrame(records)
summary = (frame.groupby("design")
           .apply(lambda g: pd.Series({
               "Bias (%)": 100 * (g["theta"].mean() - THETA) / THETA,
               "|Bias| (%)": 100 * abs(g["theta"].mean() - THETA) / THETA,
               "RMSE": float(np.sqrt(np.mean((g["theta"] - THETA) ** 2))),
               "Coverage (%)": 100 * g["covered"].mean(),
           }), include_groups=False)
           .loc[["RCF-GZ", "RCF-RMSE", "NLO-RMSE"]])

n_controls = sim["X"].shape[1]
print("\n" + "=" * 72)
print(f"Monte Carlo: T={T}, p={n_controls} controls (p/T = {n_controls / T:.1f}), "
      f"K={K}")
print(f"{REPS} replications, {SPECIFICATION} design, true theta = {THETA}")
print("=" * 72)
print(summary.round(2).to_string())
print("=" * 72)

gz = summary.loc["RCF-GZ", "|Bias| (%)"]
rmse_ = summary.loc["RCF-RMSE", "|Bias| (%)"]
nlo_ = summary.loc["NLO-RMSE", "|Bias| (%)"]
if rmse_ > 0:
    print(f"\nA. Goldilocks vs RMSE tuning, same RCF design:")
    print(f"     |bias| falls by {100 * (1 - gz / rmse_):.0f}% "
          f"({rmse_:.1f}% -> {gz:.1f}%).  Paper's Table 2: about 24%.")
if nlo_ > 0:
    print(f"B. RCF vs NLO design, same RMSE tuning:")
    print(f"     |bias| falls by {100 * (1 - rmse_ / nlo_):.0f}% "
          f"({nlo_:.1f}% -> {rmse_:.1f}%).  Paper's Table 2: about 7%.")
    print(f"   Both together: {100 * (1 - gz / nlo_):.0f}% "
          f"({nlo_:.1f}% -> {gz:.1f}%).  Paper: over 22%.")
print(f"\nWith {REPS} replications these numbers carry real Monte Carlo noise --")
print("the paper uses 10,000. Expect the ordering to hold and the magnitudes to move.")

# --------------------------------------------------------------------------- #
# sample use, the mechanical part of the RCF-vs-NLO story
# --------------------------------------------------------------------------- #
print(f"\nAt K={K}: RCF trains on {sample_use_rcf(K):.1%} of the sample, "
      f"NLO on {sample_use_nlo(K):.1%}.")
plot_sample_use(range(3, 16), save_path=str(OUT / "comparison_sample_use.pdf"))
print(f"Figure written to {(OUT / 'comparison_sample_use.pdf').resolve()}")
