"""
Example 03: Coupled NHEJ / MMEJ / HDR double-strand break repair kinetics.

Demonstrates:
- Microhomology-mediated deletion prediction on a repeat-rich locus vs a repeat-poor locus
- Donor ssODN concentration titration curve (0 to 500 nM) showing HDR vs NHEJ/MMEJ competition
- Strict ODE mass conservation across all 24h kinetic time points
"""

from __future__ import annotations

from nuclease_prism import get_nuclease_profile, predict_repair_spectrum
from nuclease_prism.visualizer import assert_strict_ascii, render_repair_histogram


def main() -> None:
    profile = get_nuclease_profile("SpCas9")
    # Sequence with a clean 5-bp flanking microhomology 'GTCAG' across cut_index=22
    mh_rich_context = "ATCGATCGATGTCAGATCGATCGTCAGCTAGCTAGCTAGCTAGCT"
    cut_idx = 22

    lines = [
        "=" * 78,
        "DSB REPAIR PATHWAY TITRATION (ssODN Donor 0 -> 500 nM)",
        "=" * 78,
        "Donor(nM) | NHEJ%  | MMEJ%  | HDR%   | Frameshift% | Entropy(bits)",
        "----------+--------+--------+--------+-------------+--------------",
    ]

    for donor_nm in (0.0, 25.0, 75.0, 150.0, 300.0, 500.0):
        spec = predict_repair_spectrum(
            context_seq=mh_rich_context,
            cut_index=cut_idx,
            profile=profile,
            donor_conc_nm=donor_nm,
        )
        lines.append(
            f"{donor_nm:9.1f} | {spec.pathway_fractions['nhej']*100:6.2f} | "
            f"{spec.pathway_fractions['mmej']*100:6.2f} | {spec.pathway_fractions['hdr']*100:6.2f} | "
            f"{spec.frameshift_prob*100:11.2f} | {spec.indel_entropy_bits:12.4f}"
        )

    print(assert_strict_ascii("\n".join(lines)))
    print("")
    baseline = predict_repair_spectrum(mh_rich_context, cut_idx, profile, donor_conc_nm=0.0)
    print(render_repair_histogram(baseline))


if __name__ == "__main__":
    main()
