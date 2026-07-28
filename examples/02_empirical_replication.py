"""
Full replication of the paper's empirical application (Section 5).

Estimates the dynamic response of Italian GDP, corporate lending, corporate
lending spreads, Tier 1 capital and risk-weighted assets to a prudential
capital shock, by RCF-DML local projections with Goldilocks-tuned Lasso
nuisances, and reproduces Figure 2 and the accompanying tables.

Run:
    python examples/02_empirical_replication.py

Outputs land in ``output/``:
    figure2.pdf              five-panel cumulative IRFs, 50bp Tier 1 shock
    figure1_blocks.pdf       the reverse cross-fitting scheme
    sample_use.pdf           RCF vs NLO sample use across K
    irf_<variable>.csv       scaled impulse responses
    table_irf_gdp.tex        LaTeX table of the GDP response
    table_calibration.tex    per-fold selected penalties

Takes a couple of minutes: five outcomes x six folds x 100 penalties x nine
horizons.
"""

import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import Lasso

from tsdml import (
    Calibrator,
    DataProcessor,
    DMLLocalProjections,
    calibration_table,
    irf_table,
    load_macroprudential,
    macroprudential_spec,
    plot_block_structure,
    plot_goldilocks_profile,
    plot_irf_panel,
    plot_sample_use,
    scale_irfs,
    to_latex,
)

warnings.filterwarnings("ignore")

OUT = Path("output")
OUT.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# specification -- exactly the paper's
# --------------------------------------------------------------------------- #
spec = macroprudential_spec()
data = load_macroprudential().drop(columns=spec["drop"])

K = spec["n_blocks"]          # 6 folds
LAGS = spec["num_lags"]       # 3 lags of every control, the outcome and the policy
H = spec["H"]                 # horizons 0..8

PENALTIES = {"alpha": np.linspace(0.000001, 0.1, 100),
             "max_iter": [100_000], "random_state": [42]}

print(f"policy variable : {spec['treatment_var']}")
print(f"folds K         : {K}")
print(f"lags            : {LAGS}     horizons: 0..{H}")
print(f"controls dropped: {len(spec['drop'])} (mechanically tied to the capital ratio)\n")

# --------------------------------------------------------------------------- #
# estimate one local projection per outcome
# --------------------------------------------------------------------------- #
irfs, calibrators, first_processor = {}, {}, None

for outcome, code in spec["outcomes"]:
    print(f"--- {outcome}  (transformation code {code})")

    processor = DataProcessor()
    X, y, d, leads = processor.data_prep(
        df=data,
        num_lags=LAGS,
        H=H,
        treatment_var=spec["treatment_var"],
        treatment_code=spec["treatment_code"],
        outcome_var=outcome,
        outcome_code=code,
        start_date=spec["start_date"],
        scaling_method="none",
        scale_outcome_treatment=False,
        include_constant=True,
        K=K,
    )
    first_processor = first_processor or processor
    print(f"    sample: X {X.shape}, "
          f"{processor.original_index[0].date()} to {processor.original_index[-1].date()}")

    calibrator = Calibrator(metric="goldilocks_zone", n_blocks=K)
    calibrator.calibrate(
        X, y, d,
        outcome_learner_class=Lasso, outcome_param_grid=dict(PENALTIES),
        treatment_learner_class=Lasso, treatment_param_grid=dict(PENALTIES),
    )
    calibrators[outcome] = calibrator

    lp = DMLLocalProjections(
        block_specific_learners=calibrator.block_specific_learners_,
        n_blocks=K,
        estimation_method="block",
        outcome_code=code,          # drives cumulation of the response
        confidence_level=0.95,
        outcome_name=outcome,
    )
    lp.fit(X, y, d, leads, index=processor.original_index)
    irfs[outcome] = lp.to_frame()
    print(f"    impact effect {lp.irf_[0]:+.5f},  peak |effect| "
          f"{np.max(np.abs(lp.irf_)):.5f}\n")

# --------------------------------------------------------------------------- #
# normalise to a 50 basis-point rise in the Tier 1 ratio
# --------------------------------------------------------------------------- #
scaled = scale_irfs(
    irfs,
    numerator_var="Tier 1 capital",
    denominator_var="Risk-weighted assets",
    basis_point_vars=("PNFC_Spread",),
    target_shock=0.5,
)

print("Cumulative responses to a 50bp rise in the Tier 1 capital ratio")
print("-" * 78)
import pandas as pd  # noqa: E402

table = pd.DataFrame({v: scaled[v]["Coefficient"].values for v in scaled})
table.index.name = "h"
print(table.round(4).to_string())

for name, frame in scaled.items():
    frame.to_csv(OUT / f"irf_{name.replace(' ', '_')}.csv", index=False)

# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
plot_irf_panel(
    scaled,
    order=[name for name, _ in spec["outcomes"]],
    titles=spec["labels"],
    ylabels=spec["ylabels"],
    layout=(2, 3),
    highlight="GDP_K2020",
    save_path=str(OUT / "figure2.pdf"),
)

plot_block_structure(K=5, save_path=str(OUT / "figure1_blocks.pdf"))
plot_sample_use(range(3, 16), save_path=str(OUT / "sample_use.pdf"))

# the tuning picture for the GDP equation, fold 0
plot_goldilocks_profile(
    calibrators["GDP_K2020"].rmse_profiles_[0]["treatment"],
    grid=PENALTIES["alpha"],
    window_size=3,
    title="Policy equation, fold 0: RMSE profile and Goldilocks zone",
    save_path=str(OUT / "goldilocks_profile.pdf"),
)

# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
to_latex(
    irf_table(scaled["GDP_K2020"], digits=3),
    caption="Cumulative response of Italian GDP to a 50 basis-point rise in the "
            "Tier 1 capital ratio",
    label="tab:gdp_irf",
    notes="RCF-DML local projections with Goldilocks-tuned Lasso nuisances, "
          "$K=6$ folds, three lags of all controls. Newey-West standard errors "
          "with bandwidth $m=\\min(h+1,24)$ in parentheses. "
          "*** $p<0.01$, ** $p<0.05$, * $p<0.10$.",
    save_path=str(OUT / "table_irf_gdp.tex"),
)

to_latex(
    calibration_table(calibrators["GDP_K2020"]),
    caption="Fold-specific penalties selected by the Goldilocks-zone criterion",
    label="tab:calibration",
    notes="Validation RMSE is computed on the auxiliary block adjacent to each "
          "main block; the main block is never used for tuning.",
    save_path=str(OUT / "table_calibration.tex"),
)

print(f"\nWritten to {OUT.resolve()}")
for path in sorted(OUT.iterdir()):
    print("   ", path.name)
