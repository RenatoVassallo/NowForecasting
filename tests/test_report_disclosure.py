"""G5 (P1): the report publishes the provisional-calibration qualification.

Both report.md and the TeX/PDF must disclose: the fan calibration is
provisional, horizon coverage rests on 9 to 18 pseudo-real-time cells, the
post-2023 window has been inspected, and prospective validation begins with
the 2026Q2 release. One reporting function feeds both outputs, and the
prospective start comes from ``core.evaluation`` rather than a duplicated
literal. (PDF mode failing closed without a compiler is pinned separately in
tests/test_artifact_contract.py.)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def test_disclosure_wording_and_sources():
    from core.evaluation import EVALUATION_REGIME, PROSPECTIVE_START
    from pipeline.stages.report import calibration_disclosure

    d = calibration_disclosure()
    assert "provisional" in d
    assert "9 to 18" in d
    assert "pseudo-real-time" in d
    assert "final-vintage" in d
    assert "inspected" in d
    assert str(PROSPECTIVE_START) in d
    assert EVALUATION_REGIME in d


def test_prospective_start_is_sourced_not_hardcoded(monkeypatch):
    import core.evaluation as E
    from pipeline.stages import report

    monkeypatch.setattr(E, "PROSPECTIVE_START", pd.Period("2031Q3", freq="Q"))
    assert "2031Q3" in report.calibration_disclosure(), (
        "the disclosure must read PROSPECTIVE_START from core.evaluation")


def test_both_report_surfaces_carry_the_disclosure():
    src = (REPO / "pipeline" / "stages" / "report.py").read_text()
    # the markdown assembly and the tex info paragraph both call the ONE builder
    assert src.count("calibration_disclosure()") >= 2
    tex = (REPO / "pipeline" / "report" / "template.tex").read_text()
    assert "<<INFOSTATE>>" in tex          # the tex carrier of regime + disclosure
    assert "pseudo real time" in src       # the regime sentence stays


def test_template_never_claims_unqualified_real_time_calibration():
    tex = (REPO / "pipeline" / "report" / "template.tex").read_text()
    for line in tex.splitlines():
        if "real-time" in line:
            assert "pseudo-real-time" in line, (
                f"unqualified real-time claim in template: {line.strip()}")
