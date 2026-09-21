"""Offline, deterministic tests for the INEI Avance Coyuntural client.

These avoid network and PDFs (pure logic). One network-guarded test exercises the live
list scraper and skips if INEI is unreachable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sources import inei as di  # noqa: E402


# --------------------------------------------------------------------------------------
# title classification (name-drift tolerance)
# --------------------------------------------------------------------------------------
def test_classify_core_titles_across_name_drift():
    # accents, casing and wording vary across years but keywords are stable
    assert di.classify_title("ÍNDICE DEL VALOR BRUTO DE LA PRODUCCIÓN AGROPECUARIA")[0] == "prod_agropecuaria"
    assert di.classify_title("INDICE DEL VALOR BRUTO DE LA PRODUCCION AGROPECUARIA")[0] == "prod_agropecuaria"
    assert di.classify_title("ÍNDICE DEL SECTOR MANUFACTURA")[0] == "prod_manufactura"
    assert di.classify_title("VARIACIÓN % ANUALIZADA DEL ÍNDICE DE PRECIOS AL CONSUMIDOR DE LIMA")[0] == "ipc_lima"
    key, section, _label, matched = di.classify_title("IMPUESTO GENERAL A LAS VENTAS INTERNO")
    assert key == "igv_interno" and section == "fiscal" and matched


def test_classify_import_credit_vehicle_subcategories():
    # import sub-categories must win over the generic "importaciones" (order matters)
    assert di.classify_title("IMPORTACIONES DE BIENES DE CONSUMO")[0] == "imp_bienes_consumo"
    assert di.classify_title("IMPORTACIONES DE MATERIAS PRIMAS Y PRODUCTOS INTERMEDIOS")[0] == "imp_materias_primas"
    assert di.classify_title("IMPORTACIONES DE BIENES DE CAPITAL Y MATERIALES PARA LA CONSTRUCCIÓN")[0] == "imp_bienes_capital"
    assert di.classify_title("IMPORTACIONES TOTALES")[0] == "importaciones"      # generic = totals only
    # credit by type; the narrow Mivivienda line must not be taken as the broad mortgage series
    assert di.classify_title("CRÉDITOS HIPOTECARIOS PARA VIVIENDA")[0] == "credito_hipotecario"
    assert di.classify_title("CRÉDITOS DE CONSUMO")[0] == "credito_consumo"
    assert di.classify_title("CRÉDITO HIPOTECARIO MIVIVIENDA")[0] != "credito_hipotecario"
    # both era titles map to the one vehicle series
    assert di.classify_title("VENTA DE VEHÍCULOS NUEVOS")[0] == "venta_vehiculos"
    assert di.classify_title("VENTA E INMATRICULACIÓN DE VEHÍCULOS NUEVOS")[0] == "venta_vehiculos"


def test_classify_unmatched_gets_slug():
    key, section, _label, matched = di.classify_title("ALGÚN INDICADOR RARÍSIMO DE PRUEBA")
    assert not matched and section == "otros" and key == "algun_indicador_rarisimo_de_prueba"


def test_moneda_split_rules():
    mn = "CRÉDITO NOMINAL DEL SISTEMA BANCARIO AL SECTOR PRIVADO EN MONEDA NACIONAL"
    me = "CRÉDITO NOMINAL DEL SISTEMA BANCARIO AL SECTOR PRIVADO EN MONEDA EXTRANJERA"
    assert di.classify_title(mn)[0] == "credito_priv_mn"
    assert di.classify_title(me)[0] == "credito_priv_me"


def test_helpers():
    assert di._normalize("Índice   de PRECIOS") == "INDICE DE PRECIOS"
    assert di._to_num("1 234,5") == 1234.5
    assert di._to_num("-73,10") == -73.1
    assert di._to_num("P/") is None
    assert di._month_of("Set.") == 9 and di._month_of("Ene") == 1


# --------------------------------------------------------------------------------------
# report selection (skip redundant, newest-first tiling)
# --------------------------------------------------------------------------------------
def test_select_reports_tiles_history():
    dates = pd.period_range("2003-01", "2026-07", freq="M").to_timestamp()
    index = pd.DataFrame({"report_id": [f"r{i}" for i in range(len(dates))],
                          "pub_date": dates, "pdf_url": "x", "page": 1})
    sel = di.select_reports_for_history(index, overlap_months=12, window_months=36)
    assert 9 <= len(sel) <= 16
    d = sel["pub_date"].sort_values(ascending=False).reset_index(drop=True)
    assert d.iloc[0] == dates.max()                       # newest kept
    gaps = (-d.diff().dt.days.dropna() / 30).round()      # ~24-month steps
    assert gaps.between(20, 28).all()


# --------------------------------------------------------------------------------------
# unifier: within-report YoY, latest vintage wins
# --------------------------------------------------------------------------------------
def _vintage_rows(report_id, pub_date, level_by_period):
    return pd.DataFrame([{
        "report_id": report_id, "pub_date": pd.Timestamp(pub_date),
        "report_ref_month": max(level_by_period), "indicator_key": "prod_x",
        "section": "produccion", "label": "X", "raw_title": "X",
        "reference_period": p, "base_year": 2007, "value_level": v, "var_pct_printed": None,
    } for p, v in level_by_period.items()])


def test_unified_series_latest_vintage_wins():
    periods = pd.period_range("2018-01", "2019-12", freq="M")
    old = _vintage_rows("A", "2020-01-01", {p: 100.0 for p in periods})
    # newer report revises 2019-12 up to 110 (2018-12 stays 100 -> YoY 10%)
    new_levels = {p: 100.0 for p in periods}
    new_levels[pd.Period("2019-12", "M")] = 110.0
    new = _vintage_rows("B", "2020-06-01", new_levels)
    out = di.build_unified_series(pd.concat([old, new], ignore_index=True))
    yoy = out["yoy"]["prod_x"]
    assert round(yoy[pd.Period("2019-12", "M")], 2) == 10.0        # from the newer report
    assert round(out["level_latest"]["prod_x"][pd.Period("2019-12", "M")], 1) == 110.0


def test_rank_indicator_consistency_metrics():
    periods = pd.period_range("2010-01", "2010-12", freq="M")
    v = pd.concat([
        _vintage_rows("A", "2011-03-01", {p: 100.0 for p in periods}),
        _vintage_rows("B", "2012-03-01", {p: 101.0 for p in periods}),
    ], ignore_index=True)
    v["raw_title"] = ["INDICE X"] * len(v)
    rank = di.rank_indicator_consistency(v)
    row = rank[rank["indicator_key"] == "prod_x"].iloc[0]
    assert row["months_covered"] == 12 and row["n_reports"] == 2
    assert row["pct_months_covered"] == 1.0


# --------------------------------------------------------------------------------------
# vintage lake: per-bulletin immutable files, catch-up ingestion
# --------------------------------------------------------------------------------------
def _vrow(report_id, pub, period="2020-01"):
    return pd.DataFrame([{
        "report_id": report_id, "pub_date": pd.Timestamp(pub), "report_ref_month": period,
        "indicator_key": "prod_x", "section": "produccion", "label": "X", "raw_title": "X",
        "reference_period": period, "base_year": 2007, "value_level": 1.0,
        "var_pct_printed": None,
    }])[di.VINTAGE_COLUMNS]


def test_vintage_lake_roundtrip(tmp_path):
    # two bulletins share a publication date; one is older
    di._write_vintage(tmp_path, "rep-A", "2026-06-01", _vrow("rep-A", "2026-06-01"))
    di._write_vintage(tmp_path, "rep-B", "2026-06-01", _vrow("rep-B", "2026-06-01"))
    di._write_vintage(tmp_path, "5011", "2003-05-01", _vrow("5011", "2003-05-01"))
    assert di._stored_report_ids(tmp_path) == {"rep-A", "rep-B", "5011"}
    assert di._stored_frontier(tmp_path) == pd.Timestamp("2026-06-01")   # from filenames
    allv = di._load_all_vintages(tmp_path)
    assert len(allv) == 3 and set(allv["report_id"]) == {"rep-A", "rep-B", "5011"}


def test_reports_to_ingest_catch_up_same_day_and_frontier():
    index = pd.DataFrame({
        "report_id": ["a", "b", "c", "d", "old"],
        "pub_date": pd.to_datetime(["2026-05-01", "2026-05-01", "2026-06-01",
                                    "2026-07-01", "2003-05-01"]),
        "pdf_url": "x", "page": 1})
    frontier = pd.Timestamp("2026-05-01")
    # have both May bulletins -> a skipped June and July are both caught up; the old
    # pre-frontier bulletin the history build skipped is NOT re-pulled
    todo = di._reports_to_ingest(index, {"a", "b"}, frontier)
    assert list(todo["report_id"]) == ["c", "d"]
    # a same-day bulletin we do not yet have is not missed (date-cursor ">" would skip it)
    assert "b" in set(di._reports_to_ingest(index, {"a"}, frontier)["report_id"])
    # idempotent: nothing to do once everything is stored
    assert di._reports_to_ingest(index, {"a", "b", "c", "d", "old"}, frontier).empty


# --------------------------------------------------------------------------------------
# authoritative overrides (curated ground truth layered over parsed values)
# --------------------------------------------------------------------------------------
def test_load_overrides_roundtrip(tmp_path):
    (tmp_path / "foo_ind.csv").write_text(
        "reference_period,value_level\n2020-01,10\n2020-03,12\n")
    ov = di._load_overrides(tmp_path)
    assert set(ov) == {"foo_ind"}
    s = ov["foo_ind"]
    assert s.loc[pd.Period("2020-03", "M")] == 12 and s.index.freqstr == "M"


def test_committed_vehicle_override_present_and_sane():
    # the AAP/SUNARP new-vehicle registrations override ships with the repo
    ov = di._load_overrides()
    assert "venta_vehiculos" in ov
    s = ov["venta_vehiculos"]
    assert s.index.min() <= pd.Period("2016-01", "M") <= s.index.max()
    assert s.index.max() >= pd.Period("2026-01", "M")
    assert 0 <= s.min() and s.max() < 40000              # vehicle units, sane magnitude


# --------------------------------------------------------------------------------------
# live scraper (network; skips on failure)
# --------------------------------------------------------------------------------------
def test_live_list_reports(tmp_path):
    try:
        idx = di.list_reports(cache_dir=tmp_path, refresh=True)
    except Exception as exc:
        pytest.skip(f"INEI unreachable: {exc}")
    assert len(idx) > 250                                  # ~287 bulletins
    assert idx["pdf_url"].str.endswith(".pdf").all()
    assert idx["pub_date"].min().year <= 2004 and idx["pub_date"].max().year >= 2025
