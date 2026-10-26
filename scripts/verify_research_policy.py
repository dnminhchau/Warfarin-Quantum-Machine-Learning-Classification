"""Validate the current retrospective research policy in the compact project.
Run from the repository root: python -m scripts.verify_research_policy
"""
from pathlib import Path
import csv
import json
import hashlib
import yaml

ROOT = Path(__file__).resolve().parent.parent
with (ROOT / "configs/feature_policy.yaml").open(encoding="utf-8") as f:
    policy = yaml.safe_load(f)
with (ROOT / policy["source_review"]).open(encoding="utf-8-sig", newline="") as f:
    review = list(csv.DictReader(f))

assert len(policy["columns"]) == len(review) == 68
assert [(c["column_index"], c["column_name"]) for c in policy["columns"]] == [
    (int(r["column_index"]), r["column_name"]) for r in review
]
assert policy["policy_status"] == "RESEARCH_APPROVED_WITH_TEMPORAL_ASSUMPTION"
assert policy["training_allowed"] is True
assert policy["research_training_allowed"] is True
assert policy["clinical_deployment_allowed"] is False
assert policy["outlier_policy"] == "RETAIN_WITH_SENSITIVITY_ANALYSIS"
assert (ROOT / policy["protocol_amendment"]).is_file()
assert (ROOT / policy["prediction_contract"]).is_file()
assert (ROOT / policy["temporal_review"]).is_file()

approved = [r["column_name"] for r in review if r["proposed_decision"] == "CANDIDATE"]
assert len(approved) == len(set(approved)) == 10
assert policy["approved_X_columns"] == approved
selected = [c["column_name"] for c in policy["columns"] if c["include_in_X"] is True]
assert selected == approved
for c in policy["columns"]:
    if c["column_name"] in approved:
        assert c["approval_status"] == "RESEARCH_APPROVED_WITH_ASSUMPTION"
        assert c["timing_evidence"] == "UNVERIFIED"
    else:
        assert c["include_in_X"] is False

for name in (
    "PharmGKB Subject ID", "PharmGKB Sample ID",
    "Subject Reached Stable Dose of Warfarin", "Therapeutic Dose of Warfarin",
    "INR on Reported Therapeutic Dose of Warfarin",
):
    assert name not in approved

manifest = json.loads((ROOT / "docs/reference/source_manifest.json").read_text(encoding="utf-8-sig"))
raw = ROOT / manifest["raw_file"]["path"]
assert raw.is_file(), f"Raw workbook missing: {raw}"
assert raw.stat().st_size == manifest["raw_file"]["size_bytes"]
assert hashlib.sha256(raw.read_bytes()).hexdigest() == manifest["raw_file"]["sha256"]

with (ROOT / "docs/results/class_balance.csv").open(encoding="utf-8-sig", newline="") as f:
    balance = list(csv.DictReader(f))
assert {r["target_group"]: int(r["count"]) for r in balance} == {
    "Low": 1786, "Intermediate": 2936, "High": 688
}
print("[PASS] 68 columns and 10 research-approved features")
print("[PASS] No target/IDs in X; temporal UNVERIFIED; clinical use blocked")
print("[PASS] Active policy paths resolve and outlier policy is consistent")
print("[PASS] IWPC raw workbook SHA-256 matches source manifest")
print("[PASS] Class distribution matches validated results")
print("ALL RESEARCH POLICY CHECKS PASSED")
