"""
Example 01: SpCas9 therapeutic knockout guide design on a TP53 exon locus.

Demonstrates:
- Multi-chromosome benchmark genome construction with implanted homologous off-targets
- Protospacer extraction across plus and minus strands
- Nearest-neighbor R-loop thermodynamics and Markov cleavage scoring
- Genome-wide FM-index mismatch & 1-nt DNA bulge off-target enumeration
- Pareto ranking and 7-bit ASCII report + CSV export verification
"""

from __future__ import annotations

from nuclease_prism import (
    GuideDesignPipeline,
    build_synthetic_benchmark_genome,
    report_to_csv,
)
from nuclease_prism.visualizer import assert_strict_ascii, render_design_report


def main() -> None:
    contigs, intervals = build_synthetic_benchmark_genome(
        target_locus_name="TP53_EXON5",
        locus_len=360,
        background_len=2800,
        seed=101,
    )
    pipeline = GuideDesignPipeline(
        reference_contigs=contigs,
        intervals=intervals,
        max_offtarget_mismatches=3,
        include_bulges=True,
    )
    report = pipeline.design_for_locus(
        locus_name="TP53_EXON5",
        nuclease_names=("SpCas9",),
        donor_conc_nm=0.0,
        top_k=12,
    )

    ascii_report = render_design_report(report)
    assert_strict_ascii(ascii_report)
    print(ascii_report)

    csv_text = report_to_csv(report)
    csv_lines = csv_text.strip().splitlines()
    print(f"\nExported CSV rows (including header): {len(csv_lines)}")
    print(f"Pareto Frontier Rank-1 Guides ({len(report.pareto_frontier_ids)}): {report.pareto_frontier_ids[:5]}")


if __name__ == "__main__":
    main()
