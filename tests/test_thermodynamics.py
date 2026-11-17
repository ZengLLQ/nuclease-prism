"""Thermodynamic conservation, R-loop master equation, and mismatch destabilization tests."""

from __future__ import annotations

import pytest

from nuclease_prism.nucleases import get_nuclease_profile
from nuclease_prism.thermodynamics import (
    KineticParameters,
    analyze_rloop_thermodynamics,
    compute_hybrid_melting_temp,
    compute_step_rloop_energies,
    solve_rloop_master_equation,
)


def test_master_equation_probability_conservation(sample_spcas9_spacer: str) -> None:
    sp = get_nuclease_profile("SpCas9")
    thermo = analyze_rloop_thermodynamics(sample_spcas9_spacer, sp)
    assert len(thermo.state_occupancy) == sp.spacer_length + 1
    assert len(thermo.positional_dg_profile) == sp.spacer_length + 1
    assert sum(thermo.state_occupancy) == pytest.approx(1.0, abs=1e-4)
    assert all(p >= 0.0 for p in thermo.state_occupancy)
    assert 0.0 < thermo.kinetic_cleavage_prob <= 1.0


def test_seed_mismatch_disrupts_cleavage_more_than_distal(sample_spcas9_spacer: str) -> None:
    sp = get_nuclease_profile("SpCas9")
    perfect = analyze_rloop_thermodynamics(sample_spcas9_spacer, sp)

    # Distal mismatch at position 1 (index 0)
    distal_mm = ("C" if sample_spcas9_spacer[0] != "C" else "A") + sample_spcas9_spacer[1:]
    # PAM-proximal seed mismatch at position 19 (index 18)
    seed_mm = (
        sample_spcas9_spacer[:18]
        + ("C" if sample_spcas9_spacer[18] != "C" else "A")
        + sample_spcas9_spacer[19:]
    )

    t_distal = analyze_rloop_thermodynamics(sample_spcas9_spacer, sp, target_sense_seq=distal_mm)
    t_seed = analyze_rloop_thermodynamics(sample_spcas9_spacer, sp, target_sense_seq=seed_mm)

    assert perfect.kinetic_cleavage_prob > t_distal.kinetic_cleavage_prob > t_seed.kinetic_cleavage_prob
    assert t_seed.ddg_net_rloop_kcal > perfect.ddg_net_rloop_kcal


def test_wobble_rg_dt_more_tolerated_than_purine_purine_clash() -> None:
    sp = get_nuclease_profile("SpCas9")
    # Spacer has G at index 15 (seed)
    spacer = "ATCGATCGATCGATCGCGAT"
    # Target sense 'A' creates rG:dT wobble; 'C' creates rG:dG purine-purine clash
    wobble_target = spacer[:15] + "A" + spacer[16:]
    clash_target = spacer[:15] + "C" + spacer[16:]

    t_wobble = analyze_rloop_thermodynamics(spacer, sp, target_sense_seq=wobble_target)
    t_clash = analyze_rloop_thermodynamics(spacer, sp, target_sense_seq=clash_target)
    assert t_wobble.kinetic_cleavage_prob > t_clash.kinetic_cleavage_prob
    assert t_wobble.ddg_net_rloop_kcal < t_clash.ddg_net_rloop_kcal


def test_melting_temperature_monotonic_with_gc() -> None:
    low_gc = "ATATATATATATATATATAT"
    mid_gc = "ACGTACGTACGTACGTACGT"
    high_gc = "GCGCGCGCGCGCGCGCGCGC"
    tm_low = compute_hybrid_melting_temp(low_gc)
    tm_mid = compute_hybrid_melting_temp(mid_gc)
    tm_high = compute_hybrid_melting_temp(high_gc)
    assert tm_low < tm_mid < tm_high


def test_cas12a_5prime_seed_thermodynamics() -> None:
    cas12a = get_nuclease_profile("AsCas12a")
    spacer23 = "GCTAGCTAGCATCGATCGATCGA"
    # Position 2 (index 1) is inside Cas12a 5' seed (1..8)
    seed_mm = spacer23[:1] + ("A" if spacer23[1] != "A" else "C") + spacer23[2:]
    distal_mm = spacer23[:-1] + ("A" if spacer23[-1] != "A" else "C")

    t_perf = analyze_rloop_thermodynamics(spacer23, cas12a)
    t_seed = analyze_rloop_thermodynamics(spacer23, cas12a, target_sense_seq=seed_mm)
    t_distal = analyze_rloop_thermodynamics(spacer23, cas12a, target_sense_seq=distal_mm)

    assert t_perf.kinetic_cleavage_prob >= t_distal.kinetic_cleavage_prob > t_seed.kinetic_cleavage_prob


def test_kinetic_temperature_and_zip_rate_sensitivity(sample_spcas9_spacer: str) -> None:
    sp = get_nuclease_profile("SpCas9")
    _, _, step_ddg, _ = compute_step_rloop_energies(sample_spcas9_spacer, sample_spcas9_spacer, sp)
    _, p_slow, _ = solve_rloop_master_equation(step_ddg, sp, KineticParameters(k_zip_forward=30.0))
    _, p_fast, _ = solve_rloop_master_equation(step_ddg, sp, KineticParameters(k_zip_forward=300.0))
    assert p_fast >= p_slow
