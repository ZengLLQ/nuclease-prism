"""
Biophysical profiles and PAM/cleavage geometry for CRISPR nucleases and precision editors.

Includes canonical profiles for:
- SpCas9 (Streptococcus pyogenes Cas9, 3' NGG PAM, blunt cut -3 bp upstream of PAM)
- SaCas9 (Staphylococcus aureus Cas9, 3' NNGRRT PAM, 21-nt spacer)
- AsCas12a (Acidaminococcus sp. Cas12a / Cpf1, 5' TTTV PAM, 23-nt spacer, 5-bp staggered cut)
- RfxCas13d (CasRx RNA-targeting nuclease, PFS-independent 23-nt spacer)
- ABE8e (Adenine Base Editor 8e on SpCas9n nickase, A4-A8 editing window)
- PE3 (Prime Editor 3 with SpCas9 H840A nickase + engineered M-MLV RT)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .types import NucleaseClass, NucleaseProfile


# IUPAC ambiguity code expansions for DNA matching
IUPAC_CODES: Dict[str, str] = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "U": "T",
    "R": "AG",
    "Y": "CT",
    "S": "GC",
    "W": "AT",
    "K": "GT",
    "M": "AC",
    "B": "CGT",
    "D": "AGT",
    "H": "ACT",
    "V": "ACG",
    "N": "ACGT",
}


def matches_iupac(seq: str, pattern: str) -> bool:
    """
    Return True if DNA sequence ``seq`` satisfies the IUPAC degenerate ``pattern``.

    Args:
        seq: Concrete uppercase DNA string (e.g., 'AGG').
        pattern: IUPAC pattern of equal length (e.g., 'NGG').

    Returns:
        Boolean match flag.
    """
    if len(seq) != len(pattern):
        return False
    for base, code in zip(seq.upper(), pattern.upper()):
        allowed = IUPAC_CODES.get(code)
        if allowed is None or base not in allowed:
            return False
    return True


def build_nuclease_registry() -> Dict[str, NucleaseProfile]:
    """
    Construct the canonical registry of supported CRISPR nucleases and precision editors.

    Returns:
        Dictionary mapping canonical nuclease names to NucleaseProfile definitions.
    """
    return {
        "SpCas9": NucleaseProfile(
            name="SpCas9",
            nuclease_class=NucleaseClass.DSB_CAS9,
            pam_pattern="NGG",
            non_canonical_pams={
                "NAG": 0.259,
                "NGA": 0.069,
                "NCG": 0.022,
                "NTG": 0.015,
            },
            pam_side="3prime",
            spacer_length=20,
            cut_offset_from_pam=-3,
            seed_range=(13, 20),
            cleavage_overhang_bp=0,
            rloop_initiation_penalty=2.35,
            cleavage_rate_max=2.10,
            dissociation_rate_base=0.22,
        ),
        "SaCas9": NucleaseProfile(
            name="SaCas9",
            nuclease_class=NucleaseClass.DSB_CAS9,
            pam_pattern="NNGRRT",
            non_canonical_pams={
                "NNGRRC": 0.12,
                "NNGAAT": 0.08,
            },
            pam_side="3prime",
            spacer_length=21,
            cut_offset_from_pam=-3,
            seed_range=(14, 21),
            cleavage_overhang_bp=0,
            rloop_initiation_penalty=2.60,
            cleavage_rate_max=1.65,
            dissociation_rate_base=0.28,
        ),
        "AsCas12a": NucleaseProfile(
            name="AsCas12a",
            nuclease_class=NucleaseClass.DSB_CAS12A,
            pam_pattern="TTTV",
            non_canonical_pams={
                "TTTT": 0.09,
                "CTTV": 0.14,
                "TCTV": 0.07,
                "TTCV": 0.06,
            },
            pam_side="5prime",
            spacer_length=23,
            cut_offset_from_pam=18,
            seed_range=(1, 8),
            cleavage_overhang_bp=5,
            rloop_initiation_penalty=2.15,
            cleavage_rate_max=1.45,
            dissociation_rate_base=0.19,
        ),
        "RfxCas13d": NucleaseProfile(
            name="RfxCas13d",
            nuclease_class=NucleaseClass.RNA_CAS13D,
            pam_pattern="N",
            non_canonical_pams={},
            pam_side="3prime",
            spacer_length=23,
            cut_offset_from_pam=11,
            seed_range=(11, 20),
            cleavage_overhang_bp=0,
            rloop_initiation_penalty=1.55,
            cleavage_rate_max=2.40,
            dissociation_rate_base=0.16,
        ),
        "ABE8e": NucleaseProfile(
            name="ABE8e",
            nuclease_class=NucleaseClass.BASE_EDITOR,
            pam_pattern="NGG",
            non_canonical_pams={"NAG": 0.21, "NGA": 0.05},
            pam_side="3prime",
            spacer_length=20,
            cut_offset_from_pam=-3,
            seed_range=(13, 20),
            cleavage_overhang_bp=0,
            editing_window=(4, 8),
            target_base="A",
            edited_base="G",
            rloop_initiation_penalty=2.40,
            cleavage_rate_max=1.95,
            dissociation_rate_base=0.23,
        ),
        "CBE4max": NucleaseProfile(
            name="CBE4max",
            nuclease_class=NucleaseClass.BASE_EDITOR,
            pam_pattern="NGG",
            non_canonical_pams={"NAG": 0.19, "NGA": 0.04},
            pam_side="3prime",
            spacer_length=20,
            cut_offset_from_pam=-3,
            seed_range=(13, 20),
            cleavage_overhang_bp=0,
            editing_window=(4, 8),
            target_base="C",
            edited_base="T",
            rloop_initiation_penalty=2.40,
            cleavage_rate_max=1.80,
            dissociation_rate_base=0.25,
        ),
        "PE3": NucleaseProfile(
            name="PE3",
            nuclease_class=NucleaseClass.PRIME_EDITOR,
            pam_pattern="NGG",
            non_canonical_pams={"NAG": 0.15},
            pam_side="3prime",
            spacer_length=20,
            cut_offset_from_pam=-3,
            seed_range=(13, 20),
            cleavage_overhang_bp=0,
            editing_window=(1, 15),
            rloop_initiation_penalty=2.55,
            cleavage_rate_max=1.50,
            dissociation_rate_base=0.26,
        ),
    }


NUCLEASE_REGISTRY: Dict[str, NucleaseProfile] = build_nuclease_registry()


def get_nuclease_profile(name: str) -> NucleaseProfile:
    """
    Fetch a registered NucleaseProfile by case-insensitive name.

    Args:
        name: Nuclease identifier (e.g., 'SpCas9', 'spcas9', 'AsCas12a').

    Returns:
        Matching NucleaseProfile instance.

    Raises:
        KeyError: If no registered nuclease matches ``name``.
    """
    lookup = {k.lower(): v for k, v in NUCLEASE_REGISTRY.items()}
    key = name.strip().lower()
    if key not in lookup:
        available = ", ".join(sorted(NUCLEASE_REGISTRY.keys()))
        raise KeyError(f"Unknown nuclease {name!r}. Available profiles: {available}")
    return lookup[key]


def evaluate_pam_weight(pam_seq: str, profile: NucleaseProfile) -> float:
    """
    Score a concrete PAM sequence against a nuclease's canonical and non-canonical PAM table.

    Args:
        pam_seq: Uppercase concrete DNA PAM motif.
        profile: NucleaseProfile specifying canonical and wobble PAM tolerances.

    Returns:
        Multiplicative licensing weight in [0.0, 1.0]. Returns 0.0 if unlicensed.
    """
    if matches_iupac(pam_seq, profile.pam_pattern):
        return 1.0
    for alt_pattern, weight in profile.non_canonical_pams.items():
        if matches_iupac(pam_seq, alt_pattern):
            return float(weight)
    return 0.0


def list_supported_nucleases() -> List[str]:
    """Return sorted list of supported nuclease identifiers."""
    return sorted(NUCLEASE_REGISTRY.keys())


def map_spacer_to_pam_distance(pos_1based: int, profile: NucleaseProfile) -> int:
    """
    Convert 1-based 5'->3' spacer index into PAM-proximal distance (1 = closest to PAM).

    For 3'-PAM systems (SpCas9, SaCas9, ABE8e, PE3), position 20 is distance 1 from PAM.
    For 5'-PAM systems (AsCas12a), position 1 is distance 1 from PAM.

    Args:
        pos_1based: 1-based nucleotide index along the 5'->3' spacer.
        profile: Target NucleaseProfile.

    Returns:
        1-based distance from the PAM junction.
    """
    if pos_1based < 1 or pos_1based > profile.spacer_length:
        raise ValueError(
            f"Position {pos_1based} out of range [1, {profile.spacer_length}] for {profile.name}"
        )
    if profile.pam_side == "3prime":
        return profile.spacer_length - pos_1based + 1
    return pos_1based


def pam_zipping_order(profile: NucleaseProfile) -> Tuple[int, ...]:
    """
    Return 0-based spacer indices ordered sequentially from PAM-proximal nucleation to distal end.

    R-loop formation propagates directionally away from the PAM duplex junction:
    - 3'-PAM Cas9: indices [L-1, L-2, ..., 0]
    - 5'-PAM Cas12a: indices [0, 1, ..., L-1]

    Args:
        profile: NucleaseProfile instance.

    Returns:
        Tuple of 0-based spacer positions in R-loop propagation order.
    """
    if profile.pam_side == "3prime":
        return tuple(range(profile.spacer_length - 1, -1, -1))
    return tuple(range(profile.spacer_length))
