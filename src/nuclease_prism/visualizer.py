"""
Pure 7-bit ASCII terminal visualizer for NucleasePrism biophysical & genomic telemetry.

Strictly restricted to 7-bit printable ASCII characters (codes 32..126 and newline)
to guarantee crash-free rendering on Windows cp1252, headless CI runners, and POSIX terminals.
"""

from __future__ import annotations

from typing import Sequence

from .types import (
    BatchDesignReport,
    GuideEvaluation,
    OffTargetHit,
    RepairSpectrum,
    ScreenAnalysisReport,
    ThermodynamicProfile,
)


def assert_strict_ascii(text: str) -> str:
    """
    Verify that ``text`` contains only 7-bit ASCII characters (ord < 128).

    Raises:
        ValueError: If any non-ASCII code point is encountered.
    """
    for idx, ch in enumerate(text):
        if ord(ch) >= 128:
            raise ValueError(f"Non-ASCII character {ch!r} (ord={ord(ch)}) at index {idx}")
    return text


def render_rloop_energy_profile(
    thermo: ThermodynamicProfile,
    spacer_seq: str,
    width: int = 50,
) -> str:
    """
    Render an ASCII horizontal bar profile of cumulative R-loop free energy ddG(s).
    """
    profile = thermo.positional_dg_profile
    if not profile:
        return "[empty R-loop profile]"

    min_e = min(profile)
    max_e = max(profile)
    span = max(1.0, max_e - min_e)
    lines = [
        f"R-loop Free-Energy Trajectory (ddG_net={thermo.ddg_net_rloop_kcal:.2f} kcal/mol, "
        f"Tm={thermo.tm_hybrid_celsius:.1f} C, P_cleave={thermo.kinetic_cleavage_prob:.4f})",
        f"Spacer 5'->3': {spacer_seq}",
        "Step | ddG(kcal) | Profile",
        "-----+-----------+-" + "-" * width,
    ]
    for step, val in enumerate(profile):
        zero_col = int(round(((0.0 - min_e) / span) * (width - 1)))
        val_col = int(round(((val - min_e) / span) * (width - 1)))
        row = [" "] * width
        if 0 <= zero_col < width:
            row[zero_col] = "|"
        lo = min(zero_col, val_col)
        hi = max(zero_col, val_col)
        fill_ch = "=" if val <= 0.0 else "+"
        for c in range(max(0, lo), min(width, hi + 1)):
            row[c] = fill_ch
        if 0 <= val_col < width:
            row[val_col] = "*"
        lines.append(f"{step:4d} | {val:9.3f} | {''.join(row)}")

    return assert_strict_ascii("\n".join(lines))


def render_offtarget_alignments(
    spacer_seq: str,
    hits: Sequence[OffTargetHit],
    max_rows: int = 10,
) -> str:
    """
    Render an ASCII alignment table highlighting mismatches ('^') and bulges ('v').
    """
    lines = [
        f"Off-Target Alignment Summary (showing top {min(len(hits), max_rows)} of {len(hits)} hits)",
        f"Guide 5'->3': {spacer_seq}",
    ]
    if not hits:
        lines.append("  [No PAM-licensed off-targets detected within mismatch/bulge thresholds]")
        return assert_strict_ascii("\n".join(lines))

    lines.append("Loc                        | PAM | MM | Bulge     | CFD    | MIT    | Alignment")
    lines.append("-" * 92)
    for h in hits[:max_rows]:
        markers = []
        for g_ch, t_ch in zip(h.aligned_spacer, h.aligned_target):
            if g_ch == "-" or t_ch == "-":
                markers.append("v")
            elif g_ch != t_ch:
                markers.append("^")
            else:
                markers.append("|")
        loc_str = f"{h.chrom}:{h.position}({h.strand.value})"[:26]
        lines.append(
            f"{loc_str:<26} | {h.pam_seq:<3} | {h.mismatches:2d} | {h.bulge_type.value:<9} | "
            f"{h.cfd_score:6.4f} | {h.mit_score:6.4f} | {h.aligned_target}"
        )
        lines.append(f"{'':<69}   {''.join(markers)}")

    return assert_strict_ascii("\n".join(lines))


def render_repair_histogram(spectrum: RepairSpectrum, bar_width: int = 36) -> str:
    """
    Render an ASCII bar chart of top predicted DSB DNA repair outcomes and pathway split.
    """
    fracs = spectrum.pathway_fractions
    lines = [
        "DNA DSB Repair Outcome Spectrum (24h Kinetic Resolution)",
        (
            f"Pathways -> NHEJ: {fracs.get('nhej', 0.0)*100:5.1f}% | "
            f"MMEJ: {fracs.get('mmej', 0.0)*100:5.1f}% | "
            f"HDR: {fracs.get('hdr', 0.0)*100:5.1f}%"
        ),
        (
            f"Frameshift: {spectrum.frameshift_prob*100:5.1f}% | "
            f"In-Frame: {spectrum.inframe_prob*100:5.1f}% | "
            f"Entropy: {spectrum.indel_entropy_bits:.3f} bits | "
            f"Precision: {spectrum.precision_score:.3f}"
        ),
        "Allele Label        | Path | Indel | Prob   | Distribution",
        "--------------------+------+-------+--------+-" + "-" * bar_width,
    ]
    max_p = max((a.probability for a in spectrum.top_alleles), default=1.0)
    max_p = max(1e-6, max_p)
    for a in spectrum.top_alleles[:10]:
        n_bars = int(round((a.probability / max_p) * bar_width))
        fill = "#" if a.is_frameshift else (":" if a.pathway.value == "hdr" else "=")
        bar = fill * max(1, n_bars) if a.probability > 0.002 else "."
        fs_tag = "*" if a.is_frameshift else " "
        lines.append(
            f"{a.allele_label[:19]:<19} | {a.pathway.value.upper():<4} | {a.indel_size:+5d} | "
            f"{a.probability*100:5.2f}% | {bar} {fs_tag}"
        )
    lines.append("Legend: [#]=frameshift, [=]=in-frame indel, [:]=HDR, [*]=frameshift flag")
    return assert_strict_ascii("\n".join(lines))


def render_pareto_scatter(evaluations: Sequence[GuideEvaluation], width: int = 52, height: int = 14) -> str:
    """
    Render an ASCII 2D Pareto frontier plot of On-Target Efficacy (X) vs Specificity (Y).
    """
    if not evaluations:
        return "[no guide evaluations to plot]"

    grid = [[" " for _ in range(width)] for _ in range(height)]
    for ev in evaluations:
        col = int(round((ev.on_target_score / 100.0) * (width - 1)))
        row = height - 1 - int(round((ev.specificity_score / 100.0) * (height - 1)))
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))
        sym = "*" if ev.pareto_rank == 1 else ("+" if ev.pareto_rank == 2 else ".")
        if grid[row][col] != "*":
            grid[row][col] = sym

    lines = ["Pareto Frontier Scatter: On-Target Efficacy (X: 0..100) vs Specificity (Y: 0..100)"]
    for r_idx, row_chars in enumerate(grid):
        y_val = int(round(100.0 * (height - 1 - r_idx) / max(1, height - 1)))
        lines.append(f"{y_val:3d} |{''.join(row_chars)}|")
    lines.append("    +" + "-" * width + "+")
    lines.append("    0" + " " * (width // 2 - 3) + "Efficacy" + " " * (width - width // 2 - 8) + "100")
    lines.append("Legend: [*]=Pareto Rank 1 (non-dominated), [+]=Rank 2, [.]=Rank >=3")
    return assert_strict_ascii("\n".join(lines))


def render_design_report(report: BatchDesignReport) -> str:
    """
    Render a comprehensive 7-bit ASCII engineering summary for a BatchDesignReport.
    """
    lines = [
        "=" * 88,
        f"NUCLEARPRISM / NUCLEASEPRISM DESIGN REPORT :: Locus={report.locus_name} ({report.locus_length_bp} bp)",
        f"Indexed Reference: {report.reference_size_bp} bp | Nucleases: {', '.join(report.nucleases_evaluated)}",
        (
            f"Candidates Scanned: {report.total_candidates} | Reported: {len(report.evaluations)} | "
            f"Pareto-1: {len(report.pareto_frontier_ids)} | Runtime: {report.runtime_ms:.1f} ms"
        ),
        "=" * 88,
        "Rank | Guide ID                           | Nuc      | OnTgt | Spec  | FS%  | Comp  | Spacer",
        "-" * 88,
    ]
    for ev in report.evaluations[:12]:
        c = ev.candidate
        lines.append(
            f"P{ev.pareto_rank:<3} | {c.guide_id[:34]:<34} | {c.nuclease_name:<8} | "
            f"{ev.on_target_score:5.1f} | {ev.specificity_score:5.1f} | "
            f"{ev.repair_spectrum.frameshift_prob*100:4.0f} | {ev.composite_score:5.1f} | {c.spacer_seq}"
        )

    if report.evaluations:
        top = report.evaluations[0]
        lines.append("")
        lines.append(render_pareto_scatter(report.evaluations))
        lines.append("")
        lines.append(render_rloop_energy_profile(top.thermodynamics, top.candidate.spacer_seq))
        lines.append("")
        lines.append(render_offtarget_alignments(top.candidate.spacer_seq, top.off_target_hits))
        lines.append("")
        lines.append(render_repair_histogram(top.repair_spectrum))

    return assert_strict_ascii("\n".join(lines))


def render_screen_report(report: ScreenAnalysisReport, top_n: int = 12) -> str:
    """
    Render an ASCII gene-ranking bar chart and QC summary for a pooled CRISPR screen.
    """
    lines = [
        "=" * 82,
        f"POOLED CRISPR SCREEN RRA REPORT :: Guides={report.n_guides} | Genes={report.n_genes}",
        (
            f"QC -> Gini(ctrl)={report.gini_index_control:.4f} | ZeroDropout={report.zero_dropout_fraction*100:.2f}% | "
            f"NB(alpha={report.nb_dispersion_alpha:.4f}, beta={report.nb_dispersion_beta:.3f})"
            + (f" | Essential AUROC={report.essential_auroc:.4f}" if report.essential_auroc is not None else "")
        ),
        "=" * 82,
        "Rank | Gene       | nGuides | MeanLFC | RRA(-log10) | FDR(dep)  | Signal",
        "-" * 82,
    ]
    max_rra = max((g.rra_score for g in report.gene_results[:top_n]), default=1.0)
    max_rra = max(1e-6, max_rra)
    for g in report.gene_results[:top_n]:
        bar_len = int(round((g.rra_score / max_rra) * 20))
        bar_ch = "#" if g.mean_lfc < 0 else "+"
        lines.append(
            f"{g.rank:4d} | {g.gene[:10]:<10} | {g.n_guides:7d} | {g.mean_lfc:+7.3f} | "
            f"{g.rra_score:11.3f} | {g.fdr_depletion:9.2e} | {bar_ch * max(1, bar_len)}"
        )
    return assert_strict_ascii("\n".join(lines))
