"""The preprocess catalogue must be the PRODUCTION schema, validated on load.

On 2026-07-16 the DSAPM g_invq project's catalogue (columns
``transformation``/``transformation_code``, growth-named variables, no
``need_sa`` and no spec columns) overwrote ``input/metadata.xlsx``. The
defect sat dormant until the first full rebuild, which crashed deep in the
X13 step with ``AttributeError: 'Pandas' object has no attribute
'need_sa'``. The loader now fails at the door, naming the file, the sheet
and the missing columns, so a cross-project overwrite can never again
reach the panel builder.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core import preprocess


def _write_xlsx(path, monthly: pd.DataFrame, quarterly: pd.DataFrame) -> None:
    with pd.ExcelWriter(path) as xl:
        monthly.to_excel(xl, sheet_name="Monthly", index=False)
        quarterly.to_excel(xl, sheet_name="Quarterly", index=False)


def _good_sheet() -> pd.DataFrame:
    return pd.DataFrame([{
        "source_code": "PN00000XX", "variable": "toy", "frequency": "M",
        "group": "Test", "label": "Toy", "description": "", "unit": "Index",
        "need_sa": 1, "spec1": "yoy", "spec2": "yoy", "spec3": "yoy",
        "publication_delay_days": 30, "source": "BCRP", "active": 1,
        "notes": "",
    }])


def test_production_schema_loads(tmp_path):
    p = tmp_path / "metadata.xlsx"
    _write_xlsx(p, _good_sheet(), _good_sheet().assign(frequency="Q"))
    monthly, quarterly = preprocess._load_metadata(p)
    assert "need_sa" in monthly.columns and "spec3" in quarterly.columns


def test_foreign_catalogue_schema_fails_loudly(tmp_path):
    foreign = pd.DataFrame([{
        "source_code": "PN00000XX", "variable": "g_toy", "frequency": "M",
        "group": "Test", "label": "Toy", "description": "", "unit": "Index",
        "transformation_code": 1, "transformation": "Use as published",
        "first_available": "2003-01", "publication_delay_days": 30,
        "source": "BCRP", "active": 1, "notes": "",
    }])
    p = tmp_path / "metadata.xlsx"
    _write_xlsx(p, foreign, foreign)
    with pytest.raises(RuntimeError) as err:
        preprocess._load_metadata(p)
    msg = str(err.value)
    assert "need_sa" in msg and "spec3" in msg
    assert "Monthly" in msg and str(p) in msg
