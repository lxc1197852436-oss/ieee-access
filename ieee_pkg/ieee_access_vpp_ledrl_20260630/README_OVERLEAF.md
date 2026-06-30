# IEEE Access VPP LE-DRL Submission Package

This package is a dedicated IEEE Access draft converted from the VPP/LE-DRL master's thesis materials. It is intentionally separate from the earlier power-data-asset trading manuscript.

## Main Files

- `main.tex`: IEEE Access manuscript draft.
- `ieeeaccess.cls`: IEEE Access class file copied from the existing package.
- `figures/ieee_vpp_ledrl_architecture.png`: redrawn English framework figure.
- `figures/classical_baseline_results.csv`: rule, semantic heuristic, rolling-horizon, random, and Soft-Q baseline results.
- `figures/combined_baseline_summary.csv`: combined cross-scenario summary including trained SAC variants.
- `figures/combined_scenario_core_baselines.csv`: scenario-level comparison for the core baselines.

## Current Evidence Boundary

The current evidence supports the following claim:

> LLM-derived textual risk semantics improve SAC dispatch relative to a numerical-state SAC and a text-ablation SAC under the calibrated VPP simulation, especially in price-spike and renewable-curtailment scenarios.

After the semantic safety-layer sweep, the evidence also supports:

> LE-DRL-SAC with a semantic safety layer at `w=0.75` outperforms the implemented Rule-Based and Rolling-Horizon baselines in average total reward across the four current scenarios.

The current evidence does **not** support claiming that the pure SAC actor alone is the best overall dispatch method. The safety layer is necessary for the final controller to pass the strong baselines.

## Data Boundary

The modeled 15-minute VPP time series is public-data-calibrated dispatch data. The project metadata state that the load, PV, temperature, and event trajectories are generated/calibrated sequences, not verified private VPP telemetry and not a complete Guangdong spot-market clearing dataset.

Use this wording:

> The case study uses a public-data-calibrated 15-minute VPP dispatch dataset. Public disclosures calibrate the market and system background, while the dispatch trajectories support reproducible algorithm evaluation.

Avoid this wording:

> The paper uses real Guangdong VPP 15-minute operating data.
