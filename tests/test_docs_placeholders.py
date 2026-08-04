"""F11: audit reports carry no unresolved formatting placeholders.

`docs/audit/phase3_evaluation.md` shipped with literal ``{:.0f}`` cells where
tables should have been. Every audit document must render real numbers.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PLACEHOLDER = re.compile(r"\{:\.?\d*[fdgse]\}")


def test_audit_docs_have_no_unresolved_placeholders():
    offenders = []
    for f in sorted((REPO / "docs").rglob("*.md")):
        for i, line in enumerate(f.read_text().splitlines(), start=1):
            if PLACEHOLDER.search(line):
                offenders.append(f"{f.relative_to(REPO)}:{i}: {line.strip()[:80]}")
    assert not offenders, ("unresolved formatting placeholders:\n"
                           + "\n".join(offenders))


def test_prospective_evaluation_window_is_declared():
    from core.evaluation import (HOLDOUT_INSPECTED_ON, HOLDOUT_START,
                                 PROSPECTIVE_START)

    assert str(HOLDOUT_START) == "2023Q1"
    assert HOLDOUT_INSPECTED_ON == "2026-08-04"
    assert PROSPECTIVE_START > HOLDOUT_START
