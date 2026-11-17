"""
End-to-end CRISPR guide design pipeline, Pareto ranking, and synthetic genome generator.

Orchestrates protospacer scanning, GenomeFMIndex construction, R-loop thermodynamic master-equation
scoring, bulge-aware off-target enumeration, coupled NHEJ/MMEJ/HDR repair spectrum simulation,
multi-objective Pareto frontier extraction, and JSON/CSV telemetry serialization.
"""

from __future__ import annotations

import csv
import io
import json
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .indexing import GenomeFMIndex
from .nucleases import get_nuclease_profile
from .repair import predict_repair_spectrum
from .scoring import (
    analyze_precision_editing_window,
    evaluate_off_targets,
    score_on_target_efficacy,
)
from .sequence import normalize_dna, scan_protospacers
from .thermodynamics import KineticParameters, analyze_rloop_thermodynamics
from .types import (
    BatchDesignReport,
    GenomicInterval,
    GuideEvaluation,
    NucleaseClass,
    NucleaseProfile,
)


def compute_pareto_ranks(
    evaluations: Sequence[GuideEvaluation],
) -> List[int]:
    """
    Assign 1-based non-dominated Pareto frontier ranks across 3 objectives (all maximized):
      1. on_target_score
      2. specificity_score
      3. repair_precision (frameshift_prob for DSB knockouts, precision_score otherwise)

    Args:
        evaluations: Sequence of GuideEvaluation records.

    Returns:
        List of integer Pareto ranks aligned with ``evaluations`` (1 = non-dominated frontier).
    """
    n = len(evaluations)
    if n == 0:
        return []

    objs: List[Tuple[float, float, float]] = []
    for ev in evaluations:
        repair_obj = (
            ev.repair_spectrum.frameshift_prob * 100.0
            if ev.repair_spectrum.hdr_efficiency < 0.05
            else ev.repair_spectrum.hdr_efficiency * 100.0
        )
        objs.append((ev.on_target_score, ev.specificity_score, repair_obj))

    remaining = set(range(n))
    ranks = [0] * n
    cur_rank = 1

    while remaining:
        frontier: List[int] = []
        for i in remaining:
            dominated = False
            oi = objs[i]
            for j in remaining:
                if i == j:
                    continue
                oj = objs[j]
                if (
                    oj[0] >= oi[0]
                    and oj[1] >= oi[1]
                    and oj[2] >= oi[2]
                    and (oj[0] > oi[0] or oj[1] > oi[1] or oj[2] > oi[2])
                ):
                    dominated = True
                    break
            if not dominated:
                frontier.append(i)
        for idx in frontier:
            ranks[idx] = cur_rank
            remaining.remove(idx)
        cur_rank += 1

    return ranks


def build_synthetic_benchmark_genome(
    target_locus_name: str = "TP53_EXON5",
    locus_len: int = 320,
    background_len: int = 2400,
    seed: int = 42,
) -> Tuple[Dict[str, str], List[GenomicInterval]]:
    """
    Generate a deterministic multi-chromosome reference genome with annotated intervals
    and realistic implanted near-cognate off-targets (1-2 mismatches and 1-nt bulges).

    Args:
        target_locus_name: Contig name for the primary target locus.
        locus_len: Length of target exon sequence in bp.
        background_len: Length of background chromosome sequence in bp.
        seed: Random seed for reproducibility.

    Returns:
        Tuple ``(contigs_dict, genomic_intervals_list)``.
    """
    rng = np.random.default_rng(seed)
    bases = np.array(["A", "C", "G", "T"])

    # Biased GC (~52%) exon-like target locus
    probs_exon = np.array([0.24, 0.26, 0.26, 0.24])
    target_seq = "".join(rng.choice(bases, size=locus_len, p=probs_exon))
    bg_chr2 = "".join(rng.choice(bases, size=background_len, p=np.array([0.27, 0.23, 0.23, 0.27])))
    bg_chr7 = "".join(rng.choice(bases, size=background_len // 2, p=np.array([0.26, 0.24, 0.24, 0.26])))

    # Implant a near-cognate 2-mismatch SpCas9 off-target from target_seq into chr2 (exon)
    # and a 1-nt DNA bulge site into chr7 (intergenic) so off-target scoring exercises all modes
    gg_idx = target_seq.find("GG", 40, locus_len - 40)
    if gg_idx != -1 and gg_idx >= 21:
        protospacer_20 = target_seq[gg_idx - 21 : gg_idx - 1]
        pam_3 = target_seq[gg_idx - 1 : gg_idx + 2]
        # Mutate 2 distal positions (indices 1 and 4) for a realistic high-CFD off-target
        mut_list = list(protospacer_20)
        mut_list[1] = "A" if mut_list[1] != "A" else "G"
        mut_list[4] = "T" if mut_list[4] != "T" else "C"
        off_2mm = "".join(mut_list) + pam_3
        bg_chr2 = bg_chr2[:300] + off_2mm + bg_chr2[300 + len(off_2mm) :]

        # Implant a 1-nt DNA bulge off-target preserving the 8-nt PAM-proximal seed
        bulge_site = protospacer_20[:8] + "A" + protospacer_20[8:] + pam_3
        bg_chr7 = bg_chr7[:200] + bulge_site + bg_chr7[200 + len(bulge_site) :]

    contigs = {
        target_locus_name: target_seq,
        "chr2_offtarget_region": bg_chr2,
        "chr7_intergenic": bg_chr7,
    }
    intervals = [
        GenomicInterval(
            chrom=target_locus_name,
            start=0,
            end=locus_len,
            annotation="exon",
            accessibility=0.92,
            gene_symbol=target_locus_name.split("_")[0],
        ),
        GenomicInterval(
            chrom="chr2_offtarget_region",
            start=250,
            end=450,
            annotation="exon",
            accessibility=0.80,
            gene_symbol="HOMOLOG2",
        ),
        GenomicInterval(
            chrom="chr2_offtarget_region",
            start=600,
            end=1100,
            annotation="promoter",
            accessibility=0.75,
            gene_symbol="PROM2",
        ),
        GenomicInterval(
            chrom="chr7_intergenic",
            start=0,
            end=len(bg_chr7),
            annotation="intergenic",
            accessibility=0.35,
            gene_symbol="NONE",
        ),
    ]
    return contigs, intervals


class GuideDesignPipeline:
    """
    Multi-nuclease CRISPR guide design, off-target alignment, and repair simulation orchestrator.
    """

    def __init__(
        self,
        reference_contigs: Dict[str, str],
        intervals: Optional[List[GenomicInterval]] = None,
        kinetic_params: Optional[KineticParameters] = None,
        max_offtarget_mismatches: int = 3,
        include_bulges: bool = True,
    ) -> None:
        self.reference_contigs = {k: normalize_dna(v) for k, v in reference_contigs.items()}
        self.intervals = intervals or []
        self.kinetic_params = kinetic_params or KineticParameters()
        self.max_offtarget_mismatches = max_offtarget_mismatches
        self.include_bulges = include_bulges
        self.fm_index = GenomeFMIndex(self.reference_contigs, intervals=self.intervals)

    def design_for_locus(
        self,
        locus_name: str,
        nuclease_names: Sequence[str] = ("SpCas9",),
        target_seq: Optional[str] = None,
        donor_conc_nm: float = 0.0,
        allow_non_canonical_pam: bool = False,
        top_k: int = 25,
        weights: Tuple[float, float, float] = (0.45, 0.35, 0.20),
    ) -> BatchDesignReport:
        """
        Run the complete design pipeline for a named target locus.

        Args:
            locus_name: Contig key in ``reference_contigs`` (or custom locus label if ``target_seq`` supplied).
            nuclease_names: Iterable of nuclease identifiers (e.g., ``['SpCas9', 'AsCas12a']``).
            target_seq: Optional explicit target DNA sequence overriding ``reference_contigs[locus_name]``.
            donor_conc_nm: Optional HDR ssODN donor concentration in nanomolar.
            allow_non_canonical_pam: Whether to include wobble PAM candidates.
            top_k: Maximum number of top-ranked GuideEvaluation items retained.
            weights: Composite score weights ``(w_on_target, w_specificity, w_repair_precision)``.

        Returns:
            Populated BatchDesignReport.
        """
        t0 = time.perf_counter()
        seq = normalize_dna(target_seq) if target_seq is not None else self.reference_contigs[locus_name]

        profiles: List[NucleaseProfile] = [get_nuclease_profile(name) for name in nuclease_names]
        raw_evaluations: List[GuideEvaluation] = []
        total_candidates = 0

        w_on, w_spec, w_rep = weights
        w_sum = max(1e-9, w_on + w_spec + w_rep)
        w_on, w_spec, w_rep = w_on / w_sum, w_spec / w_sum, w_rep / w_sum

        for profile in profiles:
            candidates = scan_protospacers(
                sequence=seq,
                profile=profile,
                chrom=locus_name,
                intervals=self.intervals,
                allow_non_canonical_pam=allow_non_canonical_pam,
            )
            total_candidates += len(candidates)

            for cand in candidates:
                thermo = analyze_rloop_thermodynamics(
                    spacer_seq=cand.spacer_seq,
                    profile=profile,
                    params=self.kinetic_params,
                )
                on_score = score_on_target_efficacy(cand, thermo, profile)
                spec_score, cfd_burden, hits = evaluate_off_targets(
                    candidate=cand,
                    profile=profile,
                    fm_index=self.fm_index,
                    max_mismatches=self.max_offtarget_mismatches,
                    include_bulges=self.include_bulges,
                )
                repair = predict_repair_spectrum(
                    context_seq=cand.context_seq,
                    cut_index=cand.cut_index_in_context,
                    profile=profile,
                    donor_conc_nm=donor_conc_nm,
                )
                editing = analyze_precision_editing_window(cand, profile, on_score)

                if profile.nuclease_class in (NucleaseClass.DSB_CAS9, NucleaseClass.DSB_CAS12A):
                    rep_metric = (
                        repair.hdr_efficiency * 100.0 if donor_conc_nm > 0.0 else repair.frameshift_prob * 100.0
                    )
                elif profile.nuclease_class == NucleaseClass.BASE_EDITOR and editing is not None:
                    best_eff = max(editing.per_position_efficiency.values(), default=0.0)
                    rep_metric = 100.0 * best_eff * (1.0 - 0.5 * editing.bystander_risk)
                else:
                    rep_metric = repair.precision_score * 100.0

                composite = round(w_on * on_score + w_spec * spec_score + w_rep * rep_metric, 2)
                raw_evaluations.append(
                    GuideEvaluation(
                        candidate=cand,
                        thermodynamics=thermo,
                        on_target_score=on_score,
                        specificity_score=spec_score,
                        aggregate_cfd_burden=cfd_burden,
                        off_target_hits=hits,
                        repair_spectrum=repair,
                        editing_profile=editing,
                        composite_score=composite,
                        pareto_rank=1,
                    )
                )

        # Assign Pareto frontier ranks across all candidates
        p_ranks = compute_pareto_ranks(raw_evaluations)
        ranked_evals: List[GuideEvaluation] = [
            GuideEvaluation(
                candidate=ev.candidate,
                thermodynamics=ev.thermodynamics,
                on_target_score=ev.on_target_score,
                specificity_score=ev.specificity_score,
                aggregate_cfd_burden=ev.aggregate_cfd_burden,
                off_target_hits=ev.off_target_hits,
                repair_spectrum=ev.repair_spectrum,
                editing_profile=ev.editing_profile,
                composite_score=ev.composite_score,
                pareto_rank=p_ranks[idx],
            )
            for idx, ev in enumerate(raw_evaluations)
        ]

        ranked_evals.sort(key=lambda e: (e.pareto_rank, -e.composite_score, e.candidate.guide_id))
        trimmed = ranked_evals[:top_k]
        pareto_ids = [e.candidate.guide_id for e in trimmed if e.pareto_rank == 1]

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return BatchDesignReport(
            locus_name=locus_name,
            locus_length_bp=len(seq),
            reference_size_bp=self.fm_index.total_bp,
            nucleases_evaluated=[p.name for p in profiles],
            total_candidates=total_candidates,
            evaluations=trimmed,
            pareto_frontier_ids=pareto_ids,
            runtime_ms=elapsed_ms,
        )


def report_to_json(report: BatchDesignReport) -> str:
    """Serialize a BatchDesignReport to formatted JSON (pure 7-bit ASCII)."""
    return json.dumps(asdict(report), indent=2, ensure_ascii=True)


def report_to_csv(report: BatchDesignReport) -> str:
    """Serialize ranked GuideEvaluation summaries to RFC-4180 CSV."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "pareto_rank",
            "guide_id",
            "nuclease",
            "chrom",
            "start",
            "end",
            "strand",
            "spacer_seq",
            "pam_seq",
            "gc_content",
            "ddg_rloop_kcal",
            "tm_celsius",
            "p_cleave",
            "on_target_score",
            "specificity_score",
            "off_target_count",
            "frameshift_prob",
            "hdr_efficiency",
            "composite_score",
        ]
    )
    for ev in report.evaluations:
        c = ev.candidate
        t = ev.thermodynamics
        r = ev.repair_spectrum
        writer.writerow(
            [
                ev.pareto_rank,
                c.guide_id,
                c.nuclease_name,
                c.chrom,
                c.start,
                c.end,
                c.strand.value,
                c.spacer_seq,
                c.pam_seq,
                c.gc_content,
                t.ddg_net_rloop_kcal,
                t.tm_hybrid_celsius,
                t.kinetic_cleavage_prob,
                ev.on_target_score,
                ev.specificity_score,
                len(ev.off_target_hits),
                r.frameshift_prob,
                r.hdr_efficiency,
                ev.composite_score,
            ]
        )
    return buf.getvalue()
