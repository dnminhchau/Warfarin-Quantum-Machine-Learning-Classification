# Bàn giao GĐ1–GĐ3 — Warfarin QML

**Status:** READY FOR GĐ4 IMPLEMENTATION (RETROSPECTIVE RESEARCH ONLY)

## Completed

- GĐ1: `docs/project_spec.md`, `docs/protocol.md`, `docs/experiment_matrix.csv`.
- GĐ2: IWPC raw workbook (5,700 rows, 68 columns), SHA256 source manifest, dictionary and quality audits.
- GĐ3: 5,410 eligible and labeled records; Low 1,786, Intermediate 2,936, High 688. Four unit tests and prior row-level raw verification passed on user's machine.
- Feature policy: `configs/feature_policy.yaml` approves exactly 10 X features for retrospective research only. Clinical use is prohibited; temporal evidence remains UNVERIFIED.
- Outliers: 21 weekly doses >=100 mg retained in primary analysis, with sensitivity analysis planned.

## GĐ4 MUST

1. Verify workbook hash in `docs/reference/source_manifest.json` and the explicit raw filename.
2. Restrict inputs to `approved_X_columns` in YAML; do not include IDs, dose, stable flag, post-outcome INR or target-derived flags.
3. Lock stratified 20% final test once; avoid using it for feature/model selection.
4. Fit imputation, encoding, scaling, and PCA on training folds only.
5. Preserve Age groups, NA vs 0 and rare genotype handling.
6. Record the temporal limitation and outlier sensitivity study separately.
7. Verify local package versions; do not rerun one-time initializer scripts over approved documents.

## Compact packaging

The active policy is `configs/feature_policy.yaml`. Historical decisions and auxiliary regeneration scripts are preserved in `archive/supporting_materials.zip`; do not rerun the obsolete draft builder against the approved policy. This package was reorganized from the user-provided ZIP.
