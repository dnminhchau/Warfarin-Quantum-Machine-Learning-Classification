"""Thống kê mô tả phục vụ tiền xử lý, chỉ trên tập Development.

Báo cáo độ lệch của phân bố liều, tỉ lệ thiếu của từng cột (và phần do trung tâm bỏ trống cả cột),
số giá trị VKORC1 phục hồi được từ SNP liên kết, và số bệnh nhân đủ 10 cột.
Chạy: python -m scripts.analyze_preprocessing
Kết quả: results/data_validation/preprocessing_report.txt
"""
from __future__ import annotations

from pathlib import Path
import argparse
import contextlib
import io

import numpy as np
import pandas as pd
from scipy.stats import skew

from src.dataset import load_dataset, load_policy_columns, site_labels, split_ids
from src.preprocessing import PGxFeatureBuilder

ROOT = Path(__file__).resolve().parent.parent


def analyze():
    df = load_dataset(ROOT); x = load_policy_columns(ROOT)
    dev = df.loc[np.intersect1d(df.index.to_numpy(), split_ids(ROOT, "development"))]
    dose = dev["dose"].dropna()
    print(f"Liều: trung vị {dose.median():.0f} · trung bình {dose.mean():.1f} · max {dose.max():.0f}")
    print(f"Độ lệch (skew): liều {skew(dose):.2f} -> căn bậc hai {skew(np.sqrt(dose)):.2f} (gần 0 = cân đối)\n")

    st = pd.Series(site_labels(dev), index=dev.index)
    print(f"{'Cột':42s} %thiếu  %thiếu từ site bỏ trống nguyên cột")
    for c in x:
        miss = dev[c].isna()
        g = dev.groupby(st)[c].apply(lambda s: s.isna().mean())
        full_sites = g[g >= 0.95].index
        share = miss[st.isin(full_sites)].sum() / max(miss.sum(), 1)
        print(f"{c[:42]:42s} {100*miss.mean():5.1f}   {100*share:4.0f}%")

    v_before = dev["VKORC1 -1639 consensus"].map({"G/G": 0, "A/G": 1, "A/A": 2}).notna().sum()
    v_after = pd.notna(PGxFeatureBuilder(use_snp_recovery=True).transform(dev)["VKORC1"]).sum()
    print(f"\nVKORC1 -1639 có sẵn: {int(v_before)} | sau phục hồi SNP: {int(v_after)} "
          f"(+{int(v_after - v_before)} ô)")
    print(f"Complete-case đủ 10 cột: {int(dev[x].notna().all(axis=1).sum())}/{len(dev)}")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    report = io.StringIO()
    with contextlib.redirect_stdout(report):
        analyze()
    text = report.getvalue()
    out = ROOT / "results/data_validation/preprocessing_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
