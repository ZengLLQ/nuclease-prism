"""
Example 02: Precision Base Editing (ABE8e / CBE4max) vs Prime Editing (PE3) & AsCas12a.

Demonstrates:
- Multi-nuclease scanning across a pathogenic point-mutation locus
- Base editor bystander editing window quantification (A4-A8 / C4-C8)
- Prime editor pegRNA Primer Binding Site (PBS) Tm & Reverse Transcriptase Template (RTT) tuning
- 5'-PAM AsCas12a staggered-overhang comparison
"""

from __future__ import annotations

from nuclease_prism import GuideDesignPipeline, build_synthetic_benchmark_genome
from nuclease_prism.visualizer import assert_strict_ascii


def main() -> None:
    # Locus engineered to carry targetable A/C tracts adjacent to NGG and TTTV PAMs
    target_locus = (
        "GCTAGCTAGCTTTGACAGGATCCAGTAGCTAGCATCGATCGAAAGGCTTGGATCCAGT"
        "TTTAGCTAGCATCGATCGATCGATCGATCGGATCCTAGCTGATCGATCGATCGATCGA"
        "CGTACGTACGTAAACCCTAGAGGATCCAGTCTAGCTAGCTAGCATCGATCGAGGTCAG"
        "TTTCGATCGATCGAAAGGCTAGCGATCGATCGATCGATCGGGATCCTAGCTAGCTGAT"
    )
    contigs, intervals = build_synthetic_benchmark_genome(target_locus_name="PCSK9_LOCUS", seed=77)
    contigs["PCSK9_LOCUS"] = target_locus

    pipeline = GuideDesignPipeline(reference_contigs=contigs, intervals=intervals)
    report = pipeline.design_for_locus(
        locus_name="PCSK9_LOCUS",
        nuclease_names=("ABE8e", "CBE4max", "PE3", "AsCas12a"),
        top_k=16,
    )

    lines = [
        "=" * 86,
        "PRECISION EDITOR COMPARISON (ABE8e / CBE4max / PE3 / AsCas12a)",
        "=" * 86,
        f"Total PAM candidates scanned: {report.total_candidates}",
        "Nuclease  | Guide ID                      | OnTgt | Spec  | Bystander/PBS_Tm",
        "-" * 86,
    ]
    for ev in report.evaluations:
        ed = ev.editing_profile
        detail = "-"
        if ed is not None:
            if ed.optimal_pbs_tm_celsius is not None:
                detail = f"PBS_Tm={ed.optimal_pbs_tm_celsius:.1f}C RTT={ed.optimal_rtt_length}nt"
            else:
                detail = f"Pos={list(ed.target_positions)} BystRisk={ed.bystander_risk:.3f}"
        lines.append(
            f"{ev.candidate.nuclease_name:<9} | {ev.candidate.guide_id[:29]:<29} | "
            f"{ev.on_target_score:5.1f} | {ev.specificity_score:5.1f} | {detail}"
        )

    out = assert_strict_ascii("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
