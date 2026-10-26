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
| `archive/supporting_materials.zip` | Earlier decision files, detailed audit tables, old one-time audit/initializer scripts for recovery |

## Verify from project root

```powershell
python -m scripts.verify_research_policy
python -m scripts.verify_gd3_labels
python -m unittest discover -s tests -p "test_labels.py" -v
```

If reading `.xls` fails, install a compatible XLS engine such as `xlrd` in your own environment (the active label scripts currently specify `engine="xlrd"`).

**Do not run historical scripts** in `archive/supporting_materials.zip` against current approved documents without reviewing their outputs; earlier builder scripts can regenerate *draft* files. The archive can be opened to inspect any original report or script.

**Important**: `training_allowed: true` is a policy declaration, not an automatic security control; GĐ4 pipeline must enforce `approved_X_columns` and the training-fold preprocessing protocol. High-dose outliers (21 cases ≥100 mg/week) remain in main data, with planned sensitivity analysis. Do not publish patient-level files or inferred identifiers without checking ClinPGx data usage terms.
