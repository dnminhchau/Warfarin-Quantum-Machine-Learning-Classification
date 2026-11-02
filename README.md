# Warfarin QML — GĐ1–GĐ3 (compact handover)

This is a **reorganized handover copy** of the user-provided project ZIP. The original uploaded ZIP was not changed. The project is approved for **retrospective research only**, **not clinical use**; genotype and clinical feature timing remains `UNVERIFIED`.

## Where to look

| Path | Purpose |
|---|---|
| `docs/project_spec.md`, `docs/protocol.md`, `docs/experiment_matrix.csv` | GĐ1 protocol and experimental contract |
| `docs/eda_summary.md` | GĐ2 data-quality summary |
| `docs/leakage_audit.md` | GĐ3 leakage review |
| `docs/research_decisions.md` | Combined prediction-time contract, historical proposals, temporal conclusion, approved research amendment, outlier policy |
| `docs/GD1_GD3_handover.md` | GĐ4 handover instructions |
| `docs/reference/` | Source manifest, completed data dictionary, 68-column review, timing evidence review |
| `docs/results/` | Eligible-patient summary, class balance, high-dose cases |
| `configs/feature_policy.yaml` | **Authoritative**, 10 approved X columns for retrospective research only |
| `scripts/` | Only current label and policy verification scripts |
| `data/raw/`, `data/interim/` | Original IWPC data (unchanged), validated preliminary labels |

## Verify from project root

```powershell
python -m scripts.verify_research_policy
python -m scripts.verify_gd3_labels
python -m unittest discover -s tests -p "test_labels.py" -v
```

If reading `.xls` fails, install a compatible XLS engine such as `xlrd` in your own environment (the active label scripts currently specify `engine="xlrd"`).

**Historical supporting materials:** The previously referenced archive is not present in this checkout. If supplied separately, review historical scripts before running them; older builders may regenerate draft files.

**Important**: `training_allowed: true` is a policy declaration, not an automatic security control; GĐ4 pipeline must enforce `approved_X_columns` and the training-fold preprocessing protocol. High-dose outliers (21 cases ≥100 mg/week) remain in main data, with planned sensitivity analysis. Do not publish patient-level files or inferred identifiers without checking ClinPGx data usage terms.

## GĐ9 — Development CV experiments (k=4)

This section documents completed development-stage experiments.
Research use only; not intended for clinical dose recommendations.

### Main benchmark E01 — Matched-input

- Dataset: Quantum Development Pool, N=600.
- Representation: PCA k=4, fitted within training folds.
- Evaluation: shared stratified 5-fold CV.
- Primary models: Logistic Regression, SVM, MLP,
  QSVC (C=1), VQC (seed=42).
- QSVC C-grid, balanced models and additional VQC seeds
  are reported separately as exploratory analyses.

### Strong Classical Reference E02

- Dataset: full Development, N=4328.
- Features: approved full feature representation, no PCA.
- Models: Logistic Regression, SVM and MLP.
- Evaluation: stratified 5-fold CV.
- E02 is a contextual reference, not a matched-input comparison.

### Classical Learning Curve

- Models: Logistic Regression, SVM and MLP.
- Training sizes: 25%, 50%, 100% of train_base.
- Samples: 120, 240 and 480, respectively.
- Evaluation: 3 stratified holdout repeats, with validation
  fixed within each repeat.
- Results include metrics, predictions, split IDs and a figure.

### Reproducibility

Install dependencies:

    python -m pip install -r requirements.txt

Run unit tests:

    python -m unittest discover -s tests -p "test_*.py" -v

Experiment notebooks are located in `notebooks/`.
Primary result files are located in `docs/results/`.

Key outputs:

- `main_benchmark_k4_e01_primary.csv`
- `main_benchmark_k4_e01_additional.csv`
- `main_benchmark_k4_e02_reference.csv`
- `main_benchmark_k4_metrics_runtime_summary.csv`
- `main_benchmark_k4_primary_reporting_lock.json`
- `classical_learning_curve_k4_summary.csv`
- `classical_learning_curve_k4_macro_f1.png`

All paths above are relative to `docs/results/`.

### Evaluation status

Development CV results and primary reporting configurations
have been recorded. Final model selection remains pending.

The locked Final Test (N=1082) has not been used for
model fitting or evaluation in these experiments.

Temporal availability of clinical/genetic features remains
unverified. Results are retrospective research only.

### Artifact availability

The repository includes the development CV summary tables,
configuration records, and learning-curve figure.

Some row-level prediction CSVs, split-ID reports, and VQC
diagnostic JSON files are not included in this results update.

The primary reporting lock preserves original checksums,
including references to restricted local artifacts.
Full checksum verification and prediction-level audits
require those artifacts through an authorized channel.

The Final Test remains locked and was not evaluated.
