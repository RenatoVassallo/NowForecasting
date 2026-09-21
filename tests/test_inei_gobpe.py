"""I1: INEI Avance Coyuntural discovery moves to the gob.pe collection.

The old m.inei.gob.pe library remains readable but is no longer the
publication surface; discovery now walks
``/institucion/inei/colecciones/6034-avance-coyuntural`` and each item's
publication page, records full provenance (publication date, bulletin URL,
PDF URL, retrieval timestamp, sha256, reference periods) and NEVER
hard-codes a CDN filename (the upload id and cache-buster are dynamic).
Historical runs must never discover bulletins published after their as-of.

Fixtures below are trimmed from the REAL pages captured on 2026-08-06.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]

if not (REPO / "sources").exists():
    pytest.skip("private data layer absent (public clone)",
                allow_module_level=True)

from sources import inei  # noqa: E402

COLLECTION_HTML = """
<html><body>
<a href="/institucion/inei/informes-publicaciones/8433145-avance-coyuntural-de-la-actividad-economica-n-8-agosto-2026">Avance Coyuntural de la Actividad Económica N° 8 - Agosto 2026</a>
<a href="/institucion/inei/informes-publicaciones/8331002-avance-coyuntural-de-la-actividad-economica-n-7-julio-2026">N° 7 Julio 2026</a>
<a href="/institucion/inei/informes-publicaciones/6339941-avance-coyuntural-de-la-actividad-economica-n-1-enero-2025">N° 1 Enero 2025</a>
<a href="/institucion/inei/colecciones/6034-avance-coyuntural?sheet=2">2</a>
</body></html>
"""

PUB_HTML_N8 = """
<html><body>
<h1>Avance Coyuntural de la Actividad Económica N° 8 - Agosto 2026</h1>
<p>1 de agosto de 2026</p>
<a href="https://cdn.www.gob.pe/uploads/document/file/10387452/8433145-avance-coyuntural-de-la-actividad-economica-n-8-agosto-2026.pdf?v=1785588011">PDF</a>
</body></html>
"""

PUB_HTML_N1_2025 = """
<html><body>
<h1>Avance Coyuntural de la Actividad Económica N° 1 - Enero 2025</h1>
<p>3 de enero de 2025</p>
<a href="https://cdn.www.gob.pe/uploads/document/file/7800001/6339941-avance-coyuntural-de-la-actividad-economica-n-1-enero-2025.pdf?v=1735900000">PDF</a>
</body></html>
"""

PUB_HTML_N7 = """
<html><body>
<h1>Avance Coyuntural de la Actividad Económica N° 7 - Julio 2026</h1>
<p>1 de julio de 2026</p>
<a href="https://cdn.www.gob.pe/uploads/document/file/10222001/8331002-avance-coyuntural-de-la-actividad-economica-n-7-julio-2026.pdf?v=1782900000">PDF</a>
</body></html>
"""


def test_discovery_parses_the_collection_page():
    items = inei.discover_gobpe_bulletins(html_pages=[COLLECTION_HTML])
    assert list(items.report_id) == [
        "8433145-avance-coyuntural-de-la-actividad-economica-n-8-agosto-2026",
        "8331002-avance-coyuntural-de-la-actividad-economica-n-7-julio-2026",
        "6339941-avance-coyuntural-de-la-actividad-economica-n-1-enero-2025"]
    top = items.iloc[0]
    assert top.bulletin_no == 8
    assert top.slug_period == pd.Timestamp("2026-08-01")
    assert top.bulletin_url == ("https://www.gob.pe/institucion/inei/"
                                "informes-publicaciones/8433145-avance-"
                                "coyuntural-de-la-actividad-economica-n-8-"
                                "agosto-2026")


def test_publication_page_yields_pdf_url_and_date():
    meta = inei.parse_gobpe_publication(PUB_HTML_N8)
    assert meta["pdf_url"].startswith("https://cdn.www.gob.pe/uploads/document/file/")
    assert meta["pdf_url"].endswith(".pdf?v=1785588011")
    assert meta["pub_date"] == pd.Timestamp("2026-08-01")


def test_publication_page_without_pdf_fails_clearly():
    with pytest.raises(inei.INEIError, match="PDF"):
        inei.parse_gobpe_publication("<html><body>nada</body></html>")


def test_as_of_excludes_later_bulletins_without_fetching_their_pages():
    fetched = []

    def fake_page(url):
        fetched.append(url)
        if "8331002" in url:
            return PUB_HTML_N7
        if "6339941" in url:
            return PUB_HTML_N1_2025
        return PUB_HTML_N8

    idx = inei.gobpe_index_asof(html_pages=[COLLECTION_HTML],
                                page_fetcher=fake_page,
                                as_of=pd.Timestamp("2026-07-15"))
    # the agosto item is excluded by its SLUG period before any page fetch,
    # and only knowable pages were retrieved at all
    assert all("8433145" not in u for u in fetched)
    assert list(idx.report_id) == [
        "8331002-avance-coyuntural-de-la-actividad-economica-n-7-julio-2026",
        "6339941-avance-coyuntural-de-la-actividad-economica-n-1-enero-2025"]
    assert (idx.pub_date <= pd.Timestamp("2026-07-15")).all()


def test_as_of_excludes_by_page_date_when_slug_month_is_ambiguous():
    def fake_page(url):
        return PUB_HTML_N8            # published 2026-08-01

    idx = inei.gobpe_index_asof(
        html_pages=['<a href="/institucion/inei/informes-publicaciones/'
                    '8433145-avance-coyuntural-de-la-actividad-economica-'
                    'n-8-agosto-2026">x</a>'],
        page_fetcher=fake_page, as_of=pd.Timestamp("2026-08-31"))
    assert len(idx) == 1
    idx2 = inei.gobpe_index_asof(
        html_pages=['<a href="/institucion/inei/informes-publicaciones/'
                    '8433145-avance-coyuntural-de-la-actividad-economica-'
                    'n-8-agosto-2026">x</a>'],
        page_fetcher=lambda u: PUB_HTML_N8.replace("1 de agosto", "9 de agosto"),
        as_of=pd.Timestamp("2026-08-05"))
    assert len(idx2) == 0, "a page-dated bulletin after the as-of must not appear"


def test_ingest_records_full_metadata_and_handles_duplicates(tmp_path):
    calls = {"downloads": 0}

    def fake_page(url):
        if "8433145" in url:
            return PUB_HTML_N8
        if "6339941" in url:
            return PUB_HTML_N1_2025
        return PUB_HTML_N7

    def fake_pdf(url, dest_dir):
        calls["downloads"] += 1
        p = Path(dest_dir) / (url.split("/")[-1].split("?")[0])
        p.write_bytes(b"%PDF fake bulletin " + url.encode())
        return p

    first = inei.ingest_gobpe_bulletins(
        cache_dir=tmp_path, html_pages=[COLLECTION_HTML],
        page_fetcher=fake_page, pdf_fetcher=fake_pdf,
        as_of=pd.Timestamp("2026-08-06"),
        frontier_pub=pd.Timestamp("2026-01-01"),
        parse=lambda pdf, rid, pub: (pd.DataFrame(), []))
    assert calls["downloads"] == 2          # n-8 and n-7 (2025 is pre-frontier)
    assert first["n_new_reports"] == 2

    idx = pd.read_parquet(tmp_path / "gobpe_index.parquet")
    row = idx.set_index("report_id").loc[
        "8433145-avance-coyuntural-de-la-actividad-economica-n-8-agosto-2026"]
    assert row.bulletin_url.startswith("https://www.gob.pe/")
    assert row.pdf_url.startswith("https://cdn.www.gob.pe/")
    assert str(pd.Timestamp(row.pub_date).date()) == "2026-08-01"
    assert len(row.sha256) == 64
    assert pd.Timestamp(row.retrieved_at) > pd.Timestamp("2026-01-01")

    second = inei.ingest_gobpe_bulletins(
        cache_dir=tmp_path, html_pages=[COLLECTION_HTML],
        page_fetcher=fake_page, pdf_fetcher=fake_pdf,
        as_of=pd.Timestamp("2026-08-06"),
        frontier_pub=pd.Timestamp("2026-01-01"),
        parse=lambda pdf, rid, pub: (pd.DataFrame(), []))
    assert calls["downloads"] == 2, "already-ingested bulletins never re-download"
    idx2 = pd.read_parquet(tmp_path / "gobpe_index.parquet")
    assert idx2.report_id.is_unique
    assert len(idx2) == len(idx)
    assert second["n_new_reports"] == 0


def test_id_migration_never_duplicates_a_stored_publication_date(tmp_path):
    """A bulletin already in the lake under its OLD-portal report id must not
    re-ingest under its gob.pe slug: publication dates already represented in
    the stored vintages are skipped even when the report_id is new."""
    vdir = tmp_path / "vintages"
    vdir.mkdir()
    # the julio bulletin, stored under the old-portal naming (pub 2026-07-01)
    old = pd.DataFrame({"period": [pd.Timestamp("2026-05-01")], "value": [1.0]})
    old.to_parquet(vdir / "2026-07-01__informe-tecnico-avance-mayo-2026.parquet")

    downloads = []

    def fake_page(url):
        if "8433145" in url:
            return PUB_HTML_N8
        if "6339941" in url:
            return PUB_HTML_N1_2025
        return PUB_HTML_N7

    def fake_pdf(url, dest_dir):
        downloads.append(url)
        p = Path(dest_dir) / (url.split("/")[-1].split("?")[0])
        p.write_bytes(b"%PDF fake")
        return p

    res = inei.ingest_gobpe_bulletins(
        cache_dir=tmp_path, html_pages=[COLLECTION_HTML],
        page_fetcher=fake_page, pdf_fetcher=fake_pdf,
        as_of=pd.Timestamp("2026-08-06"),
        parse=lambda pdf, rid, pub: (pd.DataFrame(), []))
    assert res["added"] == [
        "8433145-avance-coyuntural-de-la-actividad-economica-n-8-agosto-2026"], (
        "only the genuinely new agosto bulletin ingests; julio is already "
        "stored under its old id and enero-2025 is behind the frontier")
    assert all("8331002" not in u for u in downloads)
