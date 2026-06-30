# Overleaf IEEE Access Package

Main file:

- `main.tex`

Recommended Overleaf settings:

- Compiler: `pdfLaTeX`
- TeX Live: latest available
- Main document: `main.tex`

Included files:

- `ieeeaccess.cls`: IEEE Access class file used by `\documentclass{ieeeaccess}`.
- `IEEEtran.cls`: included as a compatibility backup from the original paper folder.
- `figures/`: only the 12 figures referenced by `main.tex`.
- `figures/seed_robustness_*.csv/png`: supplementary seed-robustness outputs
  included for review, although the manuscript uses a compact table rather than
  the PNG figure.

Before submission, replace or verify:

- Affiliations, emails, biographies, and corresponding author email.
- Funding/grant statement is currently set to no external funding.
- Public repository URL in the Data and Code Availability paragraph.
- DOI/access-date formatting for online references if IEEE requests it.
- Any institution-specific statements required by your advisor or school.

Current paper positioning:

- This is framed as an end-to-end applied framework for power data asset trading.
- Do not change the abstract back to claiming TransModal-ValueNet is best across all metrics.
- The privacy result is intentionally reported as a stress-test finding: uniform DP outperforms the current smoothed adaptive allocation under full-year LMP.
