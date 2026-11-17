"""Shared deterministic fixtures for NucleasePrism test suites."""

from __future__ import annotations

import pytest

from nuclease_prism.pipeline import build_synthetic_benchmark_genome
from nuclease_prism.types import GenomicInterval


@pytest.fixture(scope="session")
def benchmark_genome() -> tuple[dict[str, str], list[GenomicInterval]]:
    """Deterministic multi-chromosome benchmark genome with implanted off-targets."""
    return build_synthetic_benchmark_genome(
        target_locus_name="TP53_EXON5",
        locus_len=280,
        background_len=1600,
        seed=42,
    )


@pytest.fixture()
def sample_spcas9_spacer() -> str:
    """20-nt balanced GC SpCas9 spacer sequence."""
    return "GCTAGCTAGCATCGATCGAT"


@pytest.fixture()
def microhomology_context() -> tuple[str, int]:
    """44-bp DNA context carrying a 5-bp 'GTCAG' flanking microhomology across cut_index=22."""
    seq = "ATCGATCGATGTCAGATCGATCGTCAGCTAGCTAGCTAGCTAGC"
    return seq, 22
