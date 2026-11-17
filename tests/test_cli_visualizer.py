"""Tests for CLI subcommands, pure 7-bit ASCII terminal visualizer, and Windows cp1252 safety."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from nuclease_prism.cli import main as cli_main
from nuclease_prism.nucleases import get_nuclease_profile
from nuclease_prism.pipeline import GuideDesignPipeline
from nuclease_prism.repair import predict_repair_spectrum
from nuclease_prism.types import GenomicInterval
from nuclease_prism.visualizer import (
    assert_strict_ascii,
    render_design_report,
    render_offtarget_alignments,
    render_pareto_scatter,
    render_repair_histogram,
    render_rloop_energy_profile,
)


def test_visualizer_outputs_are_strict_7bit_ascii(
    benchmark_genome: tuple[dict[str, str], list[GenomicInterval]],
) -> None:
    contigs, intervals = benchmark_genome
    pipeline = GuideDesignPipeline(reference_contigs=contigs, intervals=intervals)
    report = pipeline.design_for_locus("TP53_EXON5", nuclease_names=("SpCas9",), top_k=8)

    full_txt = render_design_report(report)
    assert_strict_ascii(full_txt)
    assert "NUCLEASEPRISM DESIGN REPORT" in full_txt

    top = report.evaluations[0]
    assert_strict_ascii(render_rloop_energy_profile(top.thermodynamics, top.candidate.spacer_seq))
    assert_strict_ascii(render_offtarget_alignments(top.candidate.spacer_seq, top.off_target_hits))
    assert_strict_ascii(render_pareto_scatter(report.evaluations))

    # Also verify repair histogram independently
    spec = predict_repair_spectrum(
        top.candidate.context_seq,
        top.candidate.cut_index_in_context,
        get_nuclease_profile("SpCas9"),
    )
    hist_txt = render_repair_histogram(spec)
    assert_strict_ascii(hist_txt)
    # Confirm cp1252 encoding succeeds without UnicodeEncodeError
    full_txt.encode("cp1252", errors="strict")

    with pytest.raises(ValueError):
        assert_strict_ascii("invalid unicode box \u2500")


def test_cli_subcommands_in_process(capsys: pytest.CaptureFixture[str]) -> None:
    # 1. design (--json)
    rc = cli_main(["design", "--nucleases", "SpCas9", "--top-k", "5", "--json"])
    assert rc == 0
    out_design = json.loads(capsys.readouterr().out)
    assert out_design["locus_name"] == "TP53_EXON5"
    assert len(out_design["evaluations"]) <= 5

    # 2. offtarget
    rc = cli_main(["offtarget", "--spacer", "GCTAGCTAGCATCGATCGAT", "--nuclease", "SpCas9"])
    assert rc == 0
    out_ot = capsys.readouterr().out
    assert_strict_ascii(out_ot)

    # 3. repair
    rc = cli_main(
        [
            "repair",
            "--sequence",
            "ATCGATCGATGTCAGATCGATCGTCAGCTAGCTAGCTAGCTAGC",
            "--cut-index",
            "22",
            "--donor-nm",
            "80",
        ]
    )
    assert rc == 0
    out_rep = capsys.readouterr().out
    assert_strict_ascii(out_rep)
    assert "HDR" in out_rep

    # 4. screen
    rc = cli_main(["screen", "--n-genes", "35", "--guides-per-gene", "4", "--top-n", "6"])
    assert rc == 0
    out_scr = capsys.readouterr().out
    assert_strict_ascii(out_scr)
    assert "POOLED CRISPR SCREEN RRA REPORT" in out_scr

    # 5. calibrate
    rc = cli_main(["calibrate", "--n-samples", "12", "--seed", "7"])
    assert rc == 0
    out_cal = capsys.readouterr().out
    assert_strict_ascii(out_cal)
    assert "Sobol Sensitivity" in out_cal

    # 6. demo
    rc = cli_main(["demo", "--seed", "42"])
    assert rc == 0
    out_demo = capsys.readouterr().out
    assert_strict_ascii(out_demo)


def test_python_module_cli_subprocess() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "nuclease_prism.cli",
            "repair",
            "--sequence",
            "ATCGATCGATCGATCGATCGATCGATCG",
            "--cut-index",
            "14",
            "--json",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert "pathway_fractions" in payload
    assert payload["pathway_fractions"]["nhej"] > 0.0
