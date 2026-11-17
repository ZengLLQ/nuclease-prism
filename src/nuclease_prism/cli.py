"""
Command-line interface for NucleasePrism.

Provides both ``nuclease-prism`` console script and ``python -m nuclease_prism.cli``
entry points with 6 operational subcommands:
    - design:    Multi-nuclease guide scoring, off-target search, and Pareto ranking
    - offtarget: FM-index mismatch & bulge off-target enumeration for a guide spacer
    - repair:    Coupled NHEJ/MMEJ/HDR DNA double-strand break repair simulation
    - screen:    Pooled CRISPR screen Negative Binomial + RRA gene ranking
    - calibrate: R-loop kinetic parameter calibration and Sobol sensitivity benchmark
    - demo:      Comprehensive end-to-end biophysical & screen demonstration
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .calibration import (
    bootstrap_confidence_interval,
    calibrate_rloop_kinetics,
    generate_synthetic_cleavage_dataset,
    sobol_rloop_sensitivity,
)
from .indexing import GenomeFMIndex
from .nucleases import get_nuclease_profile, list_supported_nucleases
from .pipeline import (
    GuideDesignPipeline,
    build_synthetic_benchmark_genome,
    report_to_csv,
    report_to_json,
)
from .repair import predict_repair_spectrum
from .scoring import compute_cfd_score, compute_mit_score
from .screens import analyze_pooled_screen
from .sequence import normalize_dna, parse_fasta_text
from .thermodynamics import KineticParameters, analyze_rloop_thermodynamics
from .types import BulgeType, GenomicInterval, OffTargetHit
from .visualizer import (
    assert_strict_ascii,
    render_design_report,
    render_offtarget_alignments,
    render_repair_histogram,
    render_screen_report,
)


def _configure_ascii_stdout() -> None:
    """Reconfigure stdout safely if encoding supports it; output is always 7-bit ASCII."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except Exception:
            pass


def _load_fasta_or_synthetic(fasta_path: Optional[str], seed: int) -> tuple[Dict[str, str], str]:
    if fasta_path:
        text = Path(fasta_path).read_text(encoding="utf-8")
        contigs = parse_fasta_text(text)
        first_key = next(iter(contigs.keys()))
        return contigs, first_key
    contigs, _ = build_synthetic_benchmark_genome(seed=seed)
    return contigs, "TP53_EXON5"


def _cmd_design(args: argparse.Namespace) -> int:
    if args.fasta:
        contigs, default_locus = _load_fasta_or_synthetic(args.fasta, args.seed)
        locus = args.locus or default_locus
        intervals: List[GenomicInterval] = []
    else:
        contigs, intervals = build_synthetic_benchmark_genome(
            target_locus_name=args.locus or "TP53_EXON5",
            seed=args.seed,
        )
        locus = args.locus or "TP53_EXON5"

    if args.sequence:
        contigs[locus] = normalize_dna(args.sequence)

    pipeline = GuideDesignPipeline(
        reference_contigs=contigs,
        intervals=intervals,
        max_offtarget_mismatches=args.max_mm,
        include_bulges=not args.no_bulges,
    )
    report = pipeline.design_for_locus(
        locus_name=locus,
        nuclease_names=args.nucleases,
        donor_conc_nm=args.donor_nm,
        allow_non_canonical_pam=args.non_canonical_pam,
        top_k=args.top_k,
    )

    if args.output_json:
        Path(args.output_json).write_text(report_to_json(report), encoding="utf-8")
    if args.output_csv:
        Path(args.output_csv).write_text(report_to_csv(report), encoding="utf-8")

    if args.json:
        print(report_to_json(report))
    else:
        print(render_design_report(report))
    return 0


def _cmd_offtarget(args: argparse.Namespace) -> int:
    profile = get_nuclease_profile(args.nuclease)
    spacer = normalize_dna(args.spacer)
    if len(spacer) != profile.spacer_length:
        raise ValueError(
            f"Spacer length {len(spacer)} does not match {profile.name} ({profile.spacer_length} nt)"
        )
    contigs, _ = _load_fasta_or_synthetic(args.fasta, args.seed)
    fm = GenomeFMIndex(contigs)
    raw = fm.search_pam_offtargets(
        spacer_seq=spacer,
        profile=profile,
        max_mismatches=args.max_mm,
        include_bulges=not args.no_bulges,
    )
    hits: List[OffTargetHit] = []
    for chrom, pos, strand, aln_sp, aln_tg, pam_obs, btype, mm_c, mm_pos, pam_w in raw:
        cfd, ddg = compute_cfd_score(spacer, aln_sp, aln_tg, pam_w, btype, profile)
        mit = compute_mit_score(mm_pos, pam_w, btype, profile.spacer_length)
        hits.append(
            OffTargetHit(
                chrom=chrom,
                position=pos,
                strand=strand,
                aligned_spacer=aln_sp,
                aligned_target=aln_tg,
                pam_seq=pam_obs,
                pam_weight=pam_w,
                mismatches=mm_c,
                bulge_type=btype,
                bulge_size=0 if btype == BulgeType.NONE else 1,
                mismatch_positions=mm_pos,
                cfd_score=cfd,
                mit_score=mit,
                ddg_penalty_kcal=ddg,
            )
        )
    hits.sort(key=lambda h: (-h.cfd_score, h.mismatches))
    if args.json:
        payload = [
            {
                "chrom": h.chrom,
                "position": h.position,
                "strand": h.strand.value,
                "pam": h.pam_seq,
                "mismatches": h.mismatches,
                "bulge": h.bulge_type.value,
                "cfd": h.cfd_score,
                "mit": h.mit_score,
                "aligned_target": h.aligned_target,
            }
            for h in hits
        ]
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    else:
        print(render_offtarget_alignments(spacer, hits, max_rows=args.max_rows))
    return 0


def _cmd_repair(args: argparse.Namespace) -> int:
    profile = get_nuclease_profile(args.nuclease)
    seq = normalize_dna(args.sequence)
    cut_idx = args.cut_index if args.cut_index is not None else len(seq) // 2
    spectrum = predict_repair_spectrum(
        context_seq=seq,
        cut_index=cut_idx,
        profile=profile,
        donor_conc_nm=args.donor_nm,
    )
    if args.json:
        payload = {
            "pathway_fractions": spectrum.pathway_fractions,
            "frameshift_prob": spectrum.frameshift_prob,
            "inframe_prob": spectrum.inframe_prob,
            "hdr_efficiency": spectrum.hdr_efficiency,
            "indel_entropy_bits": spectrum.indel_entropy_bits,
            "precision_score": spectrum.precision_score,
            "top_alleles": [
                {
                    "label": a.allele_label,
                    "pathway": a.pathway.value,
                    "indel_size": a.indel_size,
                    "probability": a.probability,
                    "is_frameshift": a.is_frameshift,
                }
                for a in spectrum.top_alleles
            ],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    else:
        print(render_repair_histogram(spectrum))
    return 0


def _generate_synthetic_screen(
    n_genes: int = 80,
    guides_per_gene: int = 4,
    seed: int = 42,
) -> tuple[np.ndarray, List[str], set[str], set[str]]:
    rng = np.random.default_rng(seed)
    n_guides = n_genes * guides_per_gene
    genes: List[str] = []
    essential: set[str] = set()
    non_essential: set[str] = set()
    counts = np.zeros((n_guides, 4), dtype=np.float64)

    for g_idx in range(n_genes):
        if g_idx < 8:
            g_name = f"ESS_{g_idx+1:02d}"
            essential.add(g_name)
            trt_mult = 0.16
        elif g_idx < 14:
            g_name = f"RES_{g_idx-7:02d}"
            trt_mult = 2.85
        else:
            g_name = f"GENE_{g_idx+1:03d}"
            if g_idx < 40:
                non_essential.add(g_name)
            trt_mult = 1.0

        for _ in range(guides_per_gene):
            row = len(genes)
            genes.append(g_name)
            base_mu = float(rng.uniform(350.0, 900.0))
            counts[row, 0] = rng.negative_binomial(25.0, 25.0 / (25.0 + base_mu))
            counts[row, 1] = rng.negative_binomial(25.0, 25.0 / (25.0 + base_mu * 1.03))
            trt_mu = base_mu * trt_mult
            counts[row, 2] = rng.negative_binomial(18.0, 18.0 / (18.0 + trt_mu))
            counts[row, 3] = rng.negative_binomial(18.0, 18.0 / (18.0 + trt_mu * 0.97))

    return counts, genes, essential, non_essential


def _cmd_screen(args: argparse.Namespace) -> int:
    counts, genes, ess, non_ess = _generate_synthetic_screen(
        n_genes=args.n_genes,
        guides_per_gene=args.guides_per_gene,
        seed=args.seed,
    )
    report = analyze_pooled_screen(
        counts=counts,
        guide_genes=genes,
        control_cols=[0, 1],
        treatment_cols=[2, 3],
        essential_genes=ess,
        non_essential_genes=non_ess,
    )
    if args.json:
        payload = {
            "n_guides": report.n_guides,
            "n_genes": report.n_genes,
            "gini_index_control": report.gini_index_control,
            "essential_auroc": report.essential_auroc,
            "top_genes": [
                {
                    "rank": g.rank,
                    "gene": g.gene,
                    "mean_lfc": g.mean_lfc,
                    "rra_score": g.rra_score,
                    "fdr_depletion": g.fdr_depletion,
                }
                for g in report.gene_results[: args.top_n]
            ],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    else:
        print(render_screen_report(report, top_n=args.top_n))
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    profile = get_nuclease_profile(args.nuclease)
    true_params = KineticParameters(
        k_zip_forward=145.0,
        protein_stabilization_per_nt=-0.36,
        seed_mismatch_multiplier=1.55,
    )
    dataset = generate_synthetic_cleavage_dataset(
        profile=profile,
        true_params=true_params,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    init_guess = KineticParameters(
        k_zip_forward=90.0,
        protein_stabilization_per_nt=-0.22,
        seed_mismatch_multiplier=1.20,
    )
    cal = calibrate_rloop_kinetics(dataset, profile, initial_params=init_guess, maxiter=35)
    sobol = sobol_rloop_sensitivity(
        dataset[0].spacer_seq,
        dataset[0].target_sense_seq,
        profile,
        n_samples=32,
        seed=args.seed,
    )
    obs_vals = [d.observed_cleavage for d in dataset]
    mean_cl, ci_lo, ci_hi = bootstrap_confidence_interval(obs_vals, seed=args.seed)

    lines = [
        "=" * 78,
        f"NUCLEASEPRISM CALIBRATION & SOBOL SENSITIVITY :: {profile.name}",
        "=" * 78,
        f"Observations: {cal.n_observations} | Optimizer iterations: {cal.n_iterations}",
        f"Initial RMSE: {cal.initial_rmse:.5f} -> Fitted RMSE: {cal.fitted_rmse:.5f} | R^2: {cal.r_squared:.4f}",
        (
            f"Fitted Params: k_zip={cal.fitted_params.k_zip_forward:.2f} s^-1, "
            f"E_prot={cal.fitted_params.protein_stabilization_per_nt:.4f} kcal/mol, "
            f"seed_mult={cal.fitted_params.seed_mismatch_multiplier:.3f}"
        ),
        f"Mean Cleavage Bootstrap 95% CI: {mean_cl:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]",
        "First-Order Sobol Sensitivity Indices (S_i):",
    ]
    for param_name, s_val in sorted(sobol.items(), key=lambda kv: -kv[1]):
        bar = "#" * max(1, int(round(s_val * 30)))
        lines.append(f"  {param_name:<30} : {s_val:6.4f} | {bar}")
    print(assert_strict_ascii("\n".join(lines)))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    contigs, intervals = build_synthetic_benchmark_genome(seed=args.seed)
    pipeline = GuideDesignPipeline(reference_contigs=contigs, intervals=intervals)
    report = pipeline.design_for_locus(
        locus_name="TP53_EXON5",
        nuclease_names=("SpCas9", "AsCas12a", "ABE8e"),
        donor_conc_nm=60.0,
        top_k=10,
    )
    print(render_design_report(report))
    print("")
    counts, genes, ess, non_ess = _generate_synthetic_screen(n_genes=60, guides_per_gene=4, seed=args.seed)
    screen_rep = analyze_pooled_screen(
        counts=counts,
        guide_genes=genes,
        control_cols=[0, 1],
        treatment_cols=[2, 3],
        essential_genes=ess,
        non_essential_genes=non_ess,
    )
    print(render_screen_report(screen_rep, top_n=8))
    # Quick thermodynamic check summary
    sp_profile = get_nuclease_profile("SpCas9")
    if report.evaluations:
        ev0 = report.evaluations[0]
        t_mm = analyze_rloop_thermodynamics(
            ev0.candidate.spacer_seq,
            sp_profile,
            target_sense_seq="A" + ev0.candidate.spacer_seq[1:],
        )
        print(
            assert_strict_ascii(
                f"\nSupported nucleases ({len(list_supported_nucleases())}): "
                f"{', '.join(list_supported_nucleases())}\n"
                f"Distal 1-mm kinetic cleavage modulation check: P_cleave={t_mm.kinetic_cleavage_prob:.4f}"
            )
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level NucleasePrism CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="nuclease-prism",
        description="CRISPR guide-RNA R-loop thermodynamics, FM-index off-target alignment, and repair simulator.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_des = sub.add_parser("design", help="Design and Pareto-rank CRISPR guides for a target locus.")
    p_des.add_argument("--fasta", type=str, default=None, help="Optional reference FASTA path.")
    p_des.add_argument("--locus", type=str, default=None, help="Target contig/locus name.")
    p_des.add_argument("--sequence", type=str, default=None, help="Direct target DNA sequence.")
    p_des.add_argument("--nucleases", nargs="+", default=["SpCas9"], help="Nucleases to evaluate.")
    p_des.add_argument("--donor-nm", type=float, default=0.0, help="HDR donor template concentration (nM).")
    p_des.add_argument("--max-mm", type=int, default=3, help="Max off-target mismatches.")
    p_des.add_argument("--no-bulges", action="store_true", help="Disable 1-nt DNA/RNA bulge off-target search.")
    p_des.add_argument("--non-canonical-pam", action="store_true", help="Include wobble PAM candidates.")
    p_des.add_argument("--top-k", type=int, default=15, help="Maximum guides in report.")
    p_des.add_argument("--seed", type=int, default=42, help="Seed for synthetic benchmark genome.")
    p_des.add_argument("--json", action="store_true", help="Print JSON output to stdout.")
    p_des.add_argument("--output-json", type=str, default=None, help="Write JSON report to path.")
    p_des.add_argument("--output-csv", type=str, default=None, help="Write CSV summary to path.")
    p_des.set_defaults(func=_cmd_design)

    p_ot = sub.add_parser("offtarget", help="Run FM-index mismatch & bulge off-target search.")
    p_ot.add_argument("--spacer", required=True, help="5'->3' guide spacer sequence.")
    p_ot.add_argument("--nuclease", default="SpCas9", help="Nuclease profile name.")
    p_ot.add_argument("--fasta", type=str, default=None, help="Optional reference FASTA path.")
    p_ot.add_argument("--max-mm", type=int, default=3, help="Max Hamming mismatches.")
    p_ot.add_argument("--no-bulges", action="store_true", help="Disable bulge search.")
    p_ot.add_argument("--max-rows", type=int, default=15, help="Max displayed alignment rows.")
    p_ot.add_argument("--seed", type=int, default=42, help="Seed for synthetic genome.")
    p_ot.add_argument("--json", action="store_true", help="Print JSON output.")
    p_ot.set_defaults(func=_cmd_offtarget)

    p_rep = sub.add_parser("repair", help="Simulate DSB NHEJ/MMEJ/HDR repair allele spectrum.")
    p_rep.add_argument("--sequence", required=True, help="Local DNA context sequence (>=12 bp).")
    p_rep.add_argument("--cut-index", type=int, default=None, help="0-based cleavage index in sequence.")
    p_rep.add_argument("--nuclease", default="SpCas9", help="Nuclease profile name.")
    p_rep.add_argument("--donor-nm", type=float, default=0.0, help="HDR donor template concentration (nM).")
    p_rep.add_argument("--json", action="store_true", help="Print JSON output.")
    p_rep.set_defaults(func=_cmd_repair)

    p_scr = sub.add_parser("screen", help="Analyze pooled CRISPR screen with NB dispersion & RRA.")
    p_scr.add_argument("--n-genes", type=int, default=80, help="Synthetic gene count.")
    p_scr.add_argument("--guides-per-gene", type=int, default=4, help="Guides per gene.")
    p_scr.add_argument("--top-n", type=int, default=12, help="Top ranked genes to display.")
    p_scr.add_argument("--seed", type=int, default=42, help="RNG seed.")
    p_scr.add_argument("--json", action="store_true", help="Print JSON output.")
    p_scr.set_defaults(func=_cmd_screen)

    p_cal = sub.add_parser("calibrate", help="Calibrate R-loop kinetics and compute Sobol sensitivity.")
    p_cal.add_argument("--nuclease", default="SpCas9", help="Nuclease profile name.")
    p_cal.add_argument("--n-samples", type=int, default=20, help="Calibration dataset size.")
    p_cal.add_argument("--seed", type=int, default=42, help="RNG seed.")
    p_cal.set_defaults(func=_cmd_calibrate)

    p_demo = sub.add_parser("demo", help="Run full multi-nuclease design & pooled screen demonstration.")
    p_demo.add_argument("--seed", type=int, default=42, help="RNG seed.")
    p_demo.set_defaults(func=_cmd_demo)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point returning exit status code."""
    _configure_ascii_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
