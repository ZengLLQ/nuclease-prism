"""
Strongly typed data structures and domain models for NucleasePrism.

All biophysical quantities adhere to explicit SI/biochemical unit conventions:
- Free energies (deltaG, deltaH): kcal/mol
- Entropies (deltaS): cal/(mol*K)
- Temperatures: Kelvin (K) or Celsius (C) where explicitly named
- Rate constants (k_on, k_off, k_cleave, k_nhej, k_mmej, k_hdr): s^-1 or min^-1
- Genomic coordinates: 0-based half-open intervals [start, end)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Strand(str, Enum):
    """Genomic DNA strand orientation."""

    PLUS = "+"
    MINUS = "-"


class NucleaseClass(str, Enum):
    """Mechanistic classification of programmable CRISPR systems."""

    DSB_CAS9 = "dsb_cas9"
    DSB_CAS12A = "dsb_cas12a"
    RNA_CAS13D = "rna_cas13d"
    BASE_EDITOR = "base_editor"
    PRIME_EDITOR = "prime_editor"


class RepairPathway(str, Enum):
    """Cellular DNA double-strand break repair pathways."""

    NHEJ = "nhej"
    MMEJ = "mmej"
    HDR = "hdr"


class BulgeType(str, Enum):
    """Classification of off-target alignment gaps."""

    NONE = "none"
    DNA_BULGE = "dna_bulge"
    RNA_BULGE = "rna_bulge"


@dataclass(frozen=True)
class NucleaseProfile:
    """
    Biophysical specification for a CRISPR nuclease or precision editor.

    Attributes:
        name: Canonical identifier (e.g., 'SpCas9', 'AsCas12a', 'ABE8e', 'PE3').
        nuclease_class: Mechanistic category.
        pam_pattern: IUPAC-degenerate PAM motif (e.g., 'NGG', 'TTTV').
        non_canonical_pams: Mapping from secondary PAM motifs to relative penalty weight (0, 1].
        pam_side: Either '3prime' (Cas9) or '5prime' (Cas12a).
        spacer_length: Canonical protospacer length in nucleotides.
        cut_offset_from_pam: Signed offset from PAM boundary to DSB cleavage site.
        seed_range: 1-based inclusive PAM-proximal seed window (start, end) in spacer coordinates.
        cleavage_overhang_bp: Staggered overhang length in bp (0 for blunt SpCas9, 4-5 for Cas12a).
        editing_window: Optional 1-based spacer window (start, end) for base editors.
        target_base: Target nucleobase for base editors ('A' or 'C'), else None.
        edited_base: Resulting product nucleobase ('G' or 'T'), else None.
        rloop_initiation_penalty: Energetic barrier in kcal/mol to nucleate R-loop after PAM binding.
        cleavage_rate_max: Intrinsic catalytic cleavage rate k_cat (s^-1) when fully zippered.
        dissociation_rate_base: Basal R-loop dissociation rate scale (s^-1).
    """

    name: str
    nuclease_class: NucleaseClass
    pam_pattern: str
    non_canonical_pams: Dict[str, float]
    pam_side: str
    spacer_length: int
    cut_offset_from_pam: int
    seed_range: Tuple[int, int]
    cleavage_overhang_bp: int = 0
    editing_window: Optional[Tuple[int, int]] = None
    target_base: Optional[str] = None
    edited_base: Optional[str] = None
    rloop_initiation_penalty: float = 2.4
    cleavage_rate_max: float = 1.8
    dissociation_rate_base: float = 0.25

    def __post_init__(self) -> None:
        if self.pam_side not in ("3prime", "5prime"):
            raise ValueError(f"pam_side must be '3prime' or '5prime', got {self.pam_side!r}")
        if self.spacer_length < 12 or self.spacer_length > 32:
            raise ValueError(f"spacer_length out of biophysical bounds [12, 32]: {self.spacer_length}")


@dataclass(frozen=True)
class GenomicInterval:
    """
    Annotated genomic interval in 0-based half-open coordinates.

    Attributes:
        chrom: Chromosome or contig identifier.
        start: 0-based inclusive start coordinate.
        end: 0-based exclusive end coordinate.
        annotation: Functional region label ('exon', 'promoter', 'intron', 'intergenic').
        accessibility: Normalized chromatin accessibility weight in [0.05, 1.0].
        gene_symbol: Associated gene symbol if annotated.
    """

    chrom: str
    start: int
    end: int
    annotation: str = "exon"
    accessibility: float = 1.0
    gene_symbol: str = "TARGET"


@dataclass(frozen=True)
class ProtospacerCandidate:
    """
    Extracted candidate CRISPR target site from a genomic locus.

    Attributes:
        guide_id: Unique deterministic identifier for this candidate.
        nuclease_name: Name of the NucleaseProfile that matched this site.
        chrom: Source chromosome/contig name.
        start: 0-based start of the protospacer on the forward genomic coordinate.
        end: 0-based end of the protospacer.
        strand: Genomic strand ('+' or '-').
        spacer_seq: 5'->3' RNA-equivalent spacer DNA sequence (A/C/G/T).
        pam_seq: Matched 5'->3' PAM sequence on the target strand.
        pam_weight: Canonical/non-canonical PAM activity multiplier in (0, 1].
        context_seq: Extended local genomic sequence (typically +/- 20 bp flanking) for MFE and repair modeling.
        cut_index_in_context: 0-based index in context_seq immediately 5' of the cleavage cut.
        gc_content: Fraction of G/C nucleotides in spacer_seq.
        pol3_terminator_risk: Penalty in [0, 1] for internal U6/Pol-III TTTT runs.
        hairpin_mfe_kcal: Minimum free energy (kcal/mol) of spacer + scaffold self-folding.
        accessibility: Local chromatin accessibility score in [0.05, 1.0].
        annotation: Genomic feature annotation ('exon', 'intron', etc.).
    """

    guide_id: str
    nuclease_name: str
    chrom: str
    start: int
    end: int
    strand: Strand
    spacer_seq: str
    pam_seq: str
    pam_weight: float
    context_seq: str
    cut_index_in_context: int
    gc_content: float
    pol3_terminator_risk: float
    hairpin_mfe_kcal: float
    accessibility: float = 1.0
    annotation: str = "exon"


@dataclass(frozen=True)
class ThermodynamicProfile:
    """
    Nearest-neighbor R-loop thermodynamics and kinetic cleavage profile.

    Attributes:
        dg_rna_dna_kcal: Hybrid RNA:DNA duplex free energy at 37 C (kcal/mol).
        dg_dna_dna_kcal: Displaced DNA:DNA duplex free energy at 37 C (kcal/mol).
        ddg_net_rloop_kcal: Net strand-invasion free energy (kcal/mol), more negative = more stable.
        tm_hybrid_celsius: Estimated melting temperature of the RNA:DNA hybrid (Celsius).
        positional_dg_profile: Per-position cumulative R-loop free energy trajectory (len = spacer_length + 1).
        state_occupancy: Stationary/quasi-steady-state probability vector across R-loop zipping states.
        kinetic_cleavage_prob: Kinetic cleavage probability P_cleave from master equation competition.
        seed_stability_kcal: Integrated free energy across the PAM-proximal seed region.
    """

    dg_rna_dna_kcal: float
    dg_dna_dna_kcal: float
    ddg_net_rloop_kcal: float
    tm_hybrid_celsius: float
    positional_dg_profile: Tuple[float, ...]
    state_occupancy: Tuple[float, ...]
    kinetic_cleavage_prob: float
    seed_stability_kcal: float


@dataclass(frozen=True)
class OffTargetHit:
    """
    Aligned genomic off-target match from FM-index or bulge-aware search.

    Attributes:
        chrom: Contig identifier of the off-target site.
        position: 0-based start coordinate in reference contig.
        strand: Target genomic strand.
        aligned_spacer: Guide sequence representation with '-' for DNA bulge gaps.
        aligned_target: Genomic target sequence representation with '-' for RNA bulge gaps.
        pam_seq: Observed PAM sequence at the genomic site.
        pam_weight: Relative PAM licensing weight in (0, 1].
        mismatches: Count of nucleotide mismatches.
        bulge_type: Bulge gap classification ('none', 'dna_bulge', 'rna_bulge').
        bulge_size: Number of gap positions (0 or 1).
        mismatch_positions: 1-based guide positions carrying mismatches.
        cfd_score: Cutting Frequency Determination off-target score in [0, 1].
        mit_score: Hsu/MIT positional clustering off-target score in [0, 1].
        ddg_penalty_kcal: Thermodynamic destabilization penalty relative to on-target (kcal/mol).
        annotation: Genomic context weight ('exon', 'promoter', 'intron', 'intergenic').
    """

    chrom: str
    position: int
    strand: Strand
    aligned_spacer: str
    aligned_target: str
    pam_seq: str
    pam_weight: float
    mismatches: int
    bulge_type: BulgeType
    bulge_size: int
    mismatch_positions: Tuple[int, ...]
    cfd_score: float
    mit_score: float
    ddg_penalty_kcal: float
    annotation: str = "intergenic"


@dataclass(frozen=True)
class RepairAllele:
    """
    Individual predicted DNA repair editing outcome allele.

    Attributes:
        allele_label: Compact notation (e.g., '+1:T', '-4:MH(TCGA)', 'HDR:donor', '-2:NHEJ').
        pathway: Dominant repair mechanism producing this allele.
        indel_size: Net length change in bp (+ for insertion, - for deletion, 0 for point/HDR).
        probability: Normalized allele frequency in [0, 1].
        is_frameshift: True if indel_size % 3 != 0 and pathway != HDR.
        microhomology_len: Length of flanking microhomology repeat if MMEJ-driven.
        resulting_seq: Reconstructed local DNA sequence after repair.
    """

    allele_label: str
    pathway: RepairPathway
    indel_size: int
    probability: float
    is_frameshift: bool
    microhomology_len: int
    resulting_seq: str


@dataclass(frozen=True)
class RepairSpectrum:
    """
    Complete post-cleavage DNA repair kinetic prediction for a cut site.

    Attributes:
        pathway_fractions: Normalized proportions for NHEJ, MMEJ, and HDR (summing to 1.0).
        frameshift_prob: Total probability of out-of-frame coding disruption.
        inframe_prob: Total probability of in-frame non-HDR indels.
        hdr_efficiency: Predicted precise HDR conversion efficiency.
        indel_entropy_bits: Shannon entropy H = -sum(p_i * log2(p_i)) of allele distribution.
        precision_score: Concentration metric (1 - normalized entropy) for predictable single-allele editing.
        top_alleles: Top predicted RepairAllele outcomes sorted descending by probability.
        kinetic_time_course: Time grid (hours) and cumulative edited/unrepaired fractions.
    """

    pathway_fractions: Dict[str, float]
    frameshift_prob: float
    inframe_prob: float
    hdr_efficiency: float
    indel_entropy_bits: float
    precision_score: float
    top_alleles: Tuple[RepairAllele, ...]
    kinetic_time_course: Dict[str, Tuple[float, ...]]


@dataclass(frozen=True)
class EditingWindowProfile:
    """
    Precision base-editing or prime-editing window profile.

    Attributes:
        editor_name: Name of base/prime editor.
        target_positions: 1-based positions of editable target bases in spacer.
        per_position_efficiency: Predicted deamination or RT incorporation efficiency per position.
        bystander_risk: Probability of undesired collateral edits within the activity window.
        optimal_pbs_tm_celsius: Prime editing Primer Binding Site melting temperature (if PE3).
        optimal_rtt_length: Recommended Reverse Transcriptase Template length (if PE3).
    """

    editor_name: str
    target_positions: Tuple[int, ...]
    per_position_efficiency: Dict[int, float]
    bystander_risk: float
    optimal_pbs_tm_celsius: Optional[float] = None
    optimal_rtt_length: Optional[int] = None


@dataclass(frozen=True)
class GuideEvaluation:
    """
    Comprehensive multi-objective evaluation for a single CRISPR guide RNA.

    Attributes:
        candidate: Underlying ProtospacerCandidate geometry and sequence metadata.
        thermodynamics: R-loop stability and kinetic master-equation profile.
        on_target_score: Calibrated efficacy score in [0, 100].
        specificity_score: Genome-wide off-target specificity score in [0, 100] (100 = zero off-targets).
        aggregate_cfd_burden: Sum of weighted CFD off-target risks across the reference genome.
        off_target_hits: Enumerated off-target hits passing reporting thresholds.
        repair_spectrum: Predicted NHEJ/MMEJ/HDR repair outcome spectrum.
        editing_profile: Optional base/prime editor window analytics.
        composite_score: Weighted multi-objective utility score in [0, 100].
        pareto_rank: Non-dominated Pareto frontier rank (1 = Pareto-optimal).
    """

    candidate: ProtospacerCandidate
    thermodynamics: ThermodynamicProfile
    on_target_score: float
    specificity_score: float
    aggregate_cfd_burden: float
    off_target_hits: Tuple[OffTargetHit, ...]
    repair_spectrum: RepairSpectrum
    editing_profile: Optional[EditingWindowProfile] = None
    composite_score: float = 0.0
    pareto_rank: int = 1


@dataclass(frozen=True)
class GeneScreenResult:
    """
    Gene-level statistical summary from pooled CRISPR screen RRA analysis.

    Attributes:
        gene: Gene symbol.
        n_guides: Number of guides targeting this gene.
        mean_lfc: Mean log2 fold change across guides.
        median_lfc: Median log2 fold change across guides.
        rra_depletion_pvalue: Exact beta order-statistic RRA p-value for negative selection.
        rra_enrichment_pvalue: Exact beta order-statistic RRA p-value for positive selection.
        fdr_depletion: Benjamini-Hochberg adjusted FDR for depletion.
        fdr_enrichment: Benjamini-Hochberg adjusted FDR for enrichment.
        rra_score: -log10(min(rra_depletion_pvalue, rra_enrichment_pvalue)).
        rank: 1-based significance rank across all genes.
    """

    gene: str
    n_guides: int
    mean_lfc: float
    median_lfc: float
    rra_depletion_pvalue: float
    rra_enrichment_pvalue: float
    fdr_depletion: float
    fdr_enrichment: float
    rra_score: float
    rank: int


@dataclass
class ScreenAnalysisReport:
    """
    Complete pooled CRISPR screen statistical report and QC diagnostics.

    Attributes:
        n_guides: Total guides analyzed.
        n_genes: Total unique genes analyzed.
        gini_index_control: Gini inequality index of baseline control counts (QC target < 0.15).
        zero_dropout_fraction: Fraction of guides with zero counts in control samples.
        nb_dispersion_alpha: Fitted negative binomial dispersion slope parameter.
        nb_dispersion_beta: Fitted negative binomial baseline dispersion asymptote.
        essential_auroc: Optional AUROC separating known essential vs non-essential genes.
        gene_results: Ranked list of GeneScreenResult objects.
    """

    n_guides: int
    n_genes: int
    gini_index_control: float
    zero_dropout_fraction: float
    nb_dispersion_alpha: float
    nb_dispersion_beta: float
    essential_auroc: Optional[float]
    gene_results: List[GeneScreenResult] = field(default_factory=list)


@dataclass
class BatchDesignReport:
    """
    Top-level output of the NucleasePrism guide design pipeline.

    Attributes:
        locus_name: Name of the analyzed target locus.
        locus_length_bp: Length of the target sequence in bp.
        reference_size_bp: Total indexed reference genome size in bp.
        nucleases_evaluated: List of nuclease identifiers evaluated.
        total_candidates: Total PAM-matched candidates prior to filtering.
        evaluations: Ranked list of GuideEvaluation records.
        pareto_frontier_ids: Guide IDs lying on Pareto frontier rank 1.
        runtime_ms: Total execution duration in milliseconds.
    """

    locus_name: str
    locus_length_bp: int
    reference_size_bp: int
    nucleases_evaluated: List[str]
    total_candidates: int
    evaluations: List[GuideEvaluation]
    pareto_frontier_ids: List[str]
    runtime_ms: float
