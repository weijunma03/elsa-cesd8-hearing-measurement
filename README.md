# Hearing-group comparability of CES-D8 in ELSA Wave 7

Public code and non-disclosive aggregate tables accompanying the manuscript by Weijun Ma, Ying Cheng, Heng Li, and Hui Lv. Correspondence: weijunma@mail.xjtu.edu.cn.

## Data source and access

The participant data are from the English Longitudinal Study of Ageing (ELSA), Wave 7, UK Data Service study 5050, 50th edition, DOI: 10.5255/UKDA-SN-5050-37. ELSA data are safeguarded. Each researcher must obtain their own authorised access from the UK Data Service and follow its End User Licence. No ELSA participant records, licensed source archives, person-level intermediate files, fitted participant-level objects, or account credentials are included here.

## Contents

- factor_grid: local data loader, survey-design-aware factor model grid, and interval code.
- simulation: calibration and operating-characteristic simulation code.
- mnlfa: exploratory covariate-adjusted moderated nonlinear factor analysis pilot.
- mnlfa_inference: working-design uncertainty analyses and anchor checks.
- anchor_uncertainty: joint anchor sensitivity and uncertainty analyses.
- aggregate_tables: non-disclosive tables corresponding to manuscript Tables 1-3 and Supplementary Tables S1-S9.
- SHA256SUMS: checksums for release files.

The scripts preserve the source files used during staged local analyses. They are provided for methodological inspection and adaptation. They are not a standalone one-command reproduction pipeline: executed scripts depend on frozen local manifests, authorised ELSA input, and stage outputs that cannot be included here. Path assumptions and plan hashes reflect the original workspace. Licensed researchers must rebuild the local input frame, adapt paths, verify each stage's frozen plan and data dictionary, and reproduce each stage in sequence. The data loader expects the UKDS-distributed Wave 7 Stata data and was designed to enforce participant-level work locally.

## Analysis boundaries

The study is a secondary, cross-sectional measurement analysis. Hearing-screen groups are based on self-reported hearing difficulty and hearing-aid information, not random treatment allocation. Results do not estimate the causal effect of hearing-aid use or validate an invariant anchor set. Aggregate tables should be interpreted with the manuscript's model diagnostics and sensitivity analyses.

## Software recorded in local execution

R 4.6.1 with lavaan 0.7-3 was recorded for the factor-model grid. Python 3.12.14, NumPy 2.5.4, SciPy 1.18.1, and pandas 3.0.6 were recorded for the MNLFA stages. These versions document the original environment; other dependencies and operating-system details may require adaptation.

## Release notes

The aggregate CSV files are summaries, not a substitute for access to the underlying licensed data. They have been checked for obvious small cells: the smallest displayed group count in Table 1 is 14, and all released S9 item counts exceed 10. This is an archival code release with no general software licence granted; contact the corresponding author about reuse beyond viewing and verification.
