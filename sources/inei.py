"""INEI "Avance Coyuntural" scraper and vintage database for Peru monthly indicators.

INEI publishes the monthly bulletin *Avance Coyuntural de la Actividad Económica* (the
earliest official flash estimate of Peru's monthly activity, prices, fiscal, external,
financial and employment indicators), back to 2003, as PDFs at

    https://m.inei.gob.pe/biblioteca-virtual/boletines/avance-coyuntural/<page>/

Each bulletin restates the prior ~3 years, so a reference month appears in many bulletins
with revised (preliminary, ``P/``) values. This module:

- scrapes the paginated list of bulletins (``list_reports``),
- downloads and caches the PDFs (``download_report``),
- selects a small newest-first subset that tiles the whole history with the most-updated
  data (``select_reports_for_history``),
- parses each bulletin's indicator tables from the PDF text layer
  (``parse_report``; uses poppler ``pdftotext -bbox-layout`` for word coordinates, which
  decodes both the 2003 Distiller PDFs and the modern ones),
- keeps every data vintage as one immutable parquet per bulletin under ``<cache>/vintages/``
  (the source of truth; the merged ``inei_vintages.parquet`` and every unified/clean output
  are derived from it) and unifies them into the most updated series per indicator
  (``build_unified_series``),
- ranks indicators by publication consistency (``rank_indicator_consistency``),
- and offers a one-off history build (``build_inei_history``) plus an idempotent monthly
  update (``update_inei_latest``).

Conventions mirror ``satellite/china/utils/data_china.py``: a retrying session, cached raw
responses, a small ``get_*`` surface, and provenance recorded in ``df.attrs``.

Requirements: the poppler ``pdftotext`` binary (``brew install poppler`` /
``apt-get install poppler-utils``). Titles drift over 23 years, so a keyword registry maps
raw titles to canonical keys; anything unmatched is kept under a slug and logged.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import subprocess
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOGGER = logging.getLogger(__name__)

INEI_BASE = "https://m.inei.gob.pe"
INEI_LIST_URL = INEI_BASE + "/biblioteca-virtual/boletines/avance-coyuntural/{page}/"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "input" / "inei"
# Committed authoritative overrides (curated ground truth) that win over parsed values.
DEFAULT_OVERRIDE_DIR = Path(__file__).resolve().parent / "inei_overrides"
DEFAULT_TIMEOUT = 60
MAX_LIST_PAGES = 60  # safety bound; the list is ~29 pages

SPANISH_MONTHS: dict[str, int] = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6, "jul": 7,
    "ago": 8, "set": 9, "sep": 9, "oct": 10, "nov": 11, "dic": 12,
}
_MONTH_WORD = (
    r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
    r"setiembre|septiembre|octubre|noviembre|diciembre)"
)
_COVER_RE = re.compile(_MONTH_WORD + r"\s+(\d{4})", re.IGNORECASE)


class INEIError(RuntimeError):
    """Base error for the INEI client."""


# --------------------------------------------------------------------------------------
# Indicator title registry (absorbs 23 years of name drift)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class IndicatorRule:
    """Map a normalized title to a canonical indicator via required keywords.

    ``unit`` drives how the clean level series is built (see ``build_unified_series``):

    - ``"index"``  a rebased volume index (base year drifts 1994 -> 2007 -> 2012), so its
      single-base level is *reconstructed* from base-safe growth.
    - ``"level"``  a nominal money or price level with no rebasing; used directly.
    - ``"rate"``   a percentage (interest rate, delinquency, spread); its level is the rate.
    - ``"variation"`` published as a year-on-year % (no level in the bulletin); a base-100
      index is reconstructed by chaining that growth forward.
    """

    key: str
    section: str
    label: str
    all_keywords: tuple[str, ...]           # every keyword must be present (normalized)
    exclude: tuple[str, ...] = ()           # reject if any of these appear
    unit: str = "level"


# Ordered; first matching rule wins. Keywords are matched against the accent-stripped,
# upper-cased title. Curated for the core nowcasting indicators; extend as needed.
INDICATOR_RULES: tuple[IndicatorRule, ...] = (
    IndicatorRule("prod_agropecuaria", "produccion", "Producción agropecuaria", ("AGROPECUARIA",), unit="index"),
    IndicatorRule("prod_pesca", "produccion", "Producción pesca", ("PESCA",), unit="index"),
    IndicatorRule("prod_hidrocarburos", "produccion", "Producción hidrocarburos", ("HIDROCARBUROS",), unit="index"),
    IndicatorRule("prod_mineria", "produccion", "Producción minería metálica",
                  ("MINERIA",), exclude=("HIDROCARBUROS",), unit="index"),
    IndicatorRule("prod_manufactura", "produccion", "Sector manufactura", ("MANUFACTURA",), unit="index"),
    IndicatorRule("prod_electricidad", "produccion", "Subsector electricidad", ("ELECTRICIDAD",), unit="index"),
    IndicatorRule("consumo_cemento", "produccion", "Consumo interno de cemento", ("CEMENTO",), unit="index"),
    IndicatorRule("comercio", "produccion", "Comercio", ("COMERCIO",), unit="index"),
    IndicatorRule("servicios", "produccion", "Servicios", ("SERVICIOS",), exclude=("GOBIERNO",), unit="index"),
    IndicatorRule("ipc_lima", "precios", "IPC Lima Metropolitana (índice, base 1er año = 100)",
                  ("PRECIOS", "CONSUMIDOR"), unit="variation"),
    IndicatorRule("tipo_cambio", "precios", "Tipo de cambio (índice, base 1er año = 100)",
                  ("TIPO", "CAMBIO"), unit="variation"),
    IndicatorRule("petroleo_wti", "precios", "Petróleo WTI (US$/barril)", ("WEST", "TEXAS"), unit="level"),
    IndicatorRule("gasto_consumo_gg", "fiscal", "Gasto de consumo del gobierno general",
                  ("CONSUMO", "GOBIERNO"), unit="level"),
    IndicatorRule("gasto_inversion_gg", "fiscal", "Gasto de inversión del gobierno general",
                  ("INVERSION", "GOBIERNO"), unit="level"),
    IndicatorRule("igv_interno", "fiscal", "IGV interno", ("IMPUESTO", "VENTAS"), unit="level"),
    IndicatorRule("exportaciones", "externo", "Exportaciones totales", ("EXPORTACION",),
                  exclude=("TRADICIONAL",), unit="level"),
    # Export composition (leading indicators): non-traditional (manufactured) must precede
    # traditional, since "NO TRADICIONALES" also contains "TRADICIONAL".
    IndicatorRule("exportaciones_no_tradicionales", "externo", "Exportaciones no tradicionales",
                  ("EXPORTACION", "NO", "TRADICIONAL"), unit="level"),
    IndicatorRule("exportaciones_tradicionales", "externo", "Exportaciones tradicionales",
                  ("EXPORTACION", "TRADICIONAL"), exclude=("NO",), unit="level"),
    # Import sub-categories (leading demand indicators). These must precede the generic
    # `importaciones` rule so the specific table wins; the generic one is fenced with excludes
    # so it only ever captures "IMPORTACIONES TOTALES". All are US$ mln CIF -> unit="level".
    IndicatorRule("imp_bienes_consumo", "externo", "Importaciones de bienes de consumo",
                  ("IMPORTACION", "CONSUMO"), unit="level"),
    IndicatorRule("imp_materias_primas", "externo", "Importaciones de materias primas e insumos",
                  ("IMPORTACION", "PRIMAS"), unit="level"),
    IndicatorRule("imp_bienes_capital", "externo",
                  "Importaciones de bienes de capital y materiales de construcción",
                  ("IMPORTACION", "CAPITAL"), unit="level"),
    IndicatorRule("importaciones", "externo", "Importaciones totales", ("IMPORTACION",),
                  exclude=("CONSUMO", "PRIMAS", "INTERMEDIOS", "CAPITAL", "CONSTRUCCION"),
                  unit="level"),
    IndicatorRule("empleo", "empleo", "Empleo", ("EMPLEO",), unit="index"),
    IndicatorRule("tributos_aduaneros", "fiscal", "Tributos aduaneros", ("TRIBUTOS", "ADUANEROS"), unit="level"),
    IndicatorRule("isc_interno", "fiscal", "Impuesto selectivo al consumo interno",
                  ("SELECTIVO", "CONSUMO"), unit="level"),
    IndicatorRule("manuf_fabril_no_primario", "produccion", "Subsector fabril no primario",
                  ("FABRIL", "NO", "PRIMARIO"), unit="index"),
    IndicatorRule("morosidad_bancaria", "financiero", "Morosidad bancaria (%)", ("MOROSIDAD",), unit="rate"),
    IndicatorRule("reservas_intl_netas", "financiero", "Reservas internacionales netas (US$ mln)",
                  ("RESERVAS", "INTERNACIONALES"), unit="level"),
    IndicatorRule("credito_priv_mn", "financiero", "Crédito al sector privado (MN)",
                  ("CREDITO", "PRIVADO", "NACIONAL"), exclude=("EXTRANJERA", "MULTIPLE"), unit="level"),
    IndicatorRule("credito_priv_me", "financiero", "Crédito al sector privado (ME)",
                  ("CREDITO", "PRIVADO", "EXTRANJERA"), exclude=("MULTIPLE",), unit="level"),
    # Credit by type (leading demand indicators), S/ mln -> unit="level". Exclude the footnote
    # "...reclasificados de créditos de consumo a hipotecarios..." and the narrow Mivivienda line.
    IndicatorRule("credito_hipotecario", "financiero", "Crédito hipotecario para vivienda",
                  ("HIPOTECARI",), exclude=("MIVIVIENDA", "RECLASIFICAD"), unit="level"),
    IndicatorRule("credito_consumo", "financiero", "Crédito de consumo",
                  ("CREDITO", "CONSUMO"), exclude=("RECLASIFICAD", "HIPOTECARI"), unit="level"),
    IndicatorRule("spread_bancario_mn", "financiero", "Spread bancario (MN, pp)",
                  ("SPREAD", "NACIONAL"), unit="rate"),
    IndicatorRule("spread_bancario_me", "financiero", "Spread bancario (ME, pp)",
                  ("SPREAD", "EXTRANJERA"), unit="rate"),
    IndicatorRule("tasas_interes_mn", "financiero", "Tasas de interés bancarias (MN, %)",
                  ("TASAS", "INTERES", "NACIONAL"), exclude=("EXTRANJERA",), unit="rate"),
    IndicatorRule("tasas_interes_me", "financiero", "Tasas de interés bancarias (ME, %)",
                  ("TASAS", "INTERES", "EXTRANJERA"), unit="rate"),
    # Titled "venta de vehículos nuevos" in older bulletins and "venta e inmatriculación de
    # vehículos nuevos" recently (both match); there is no separate inmatriculación table
    # (only a footnote defining the term), so one key spans the whole history.
    IndicatorRule("venta_vehiculos", "otros", "Venta / inmatriculación de vehículos nuevos",
                  ("VENTA", "VEHICULOS"), unit="level"),
    IndicatorRule("turismo_llegadas", "turismo", "Llegada de visitantes internacionales",
                  ("LLEGADA", "VISITANTES"), unit="level"),
    IndicatorRule("turismo_arribos", "turismo", "Arribo de pasajeros en vuelos internacionales",
                  ("ARRIBO", "PASAJEROS"), unit="level"),
)

# key -> unit lookup; unmatched titles (kept under a slug) default to a direct level.
_UNIT_BY_KEY: dict[str, str] = {r.key: r.unit for r in INDICATOR_RULES}

# Plausible band for a published year-on-year % (IPC, tipo de cambio). Peru inflation and
# FX variation stay well inside this over 2001-2026; anything outside is a parse error or a
# bulletin that tabulated the index instead of the change, and is dropped before chaining.
_VARIATION_MIN, _VARIATION_MAX = -50.0, 60.0

# A month is trusted for the clean levels only when independent bulletins corroborate it:
# with >=2 vintages, the max-min spread must be within this fraction of the median. Genuine
# flash-data revisions reach ~25-30%; gross parse errors in the 2003-2010 layouts are
# order-of-magnitude (>100%), so this threshold sits well between the two -- it drops the
# unreliable early history and stray misreads while keeping real revisions (and corroborated
# shocks such as the 2020 COVID collapse, where every vintage agrees). This is separate from
# latest-vintage-wins, which the raw level output uses.
_VINTAGE_TOL = 0.5


def _unit_of(key: str) -> str:
    return _UNIT_BY_KEY.get(key, "level")


def _normalize(text: str) -> str:
    """Upper-case, strip accents, drop punctuation, collapse whitespace."""
    stripped = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )
    stripped = re.sub(r"[^A-Za-z0-9 ]+", " ", stripped)
    return re.sub(r"\s+", " ", stripped).strip().upper()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _normalize(text).lower()).strip("_")[:60]


def classify_title(title: str) -> tuple[str, str, str, bool]:
    """Return ``(indicator_key, section, label, matched)`` for a raw table title.

    Unmatched titles get a slug key and section ``"otros"`` with ``matched=False`` so the
    caller can log them; nothing is dropped.
    """
    norm = _normalize(title)
    for rule in INDICATOR_RULES:
        if all(k in norm for k in rule.all_keywords) and not any(x in norm for x in rule.exclude):
            return rule.key, rule.section, rule.label, True
    return _slug(title), "otros", title.strip(), False


# --------------------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------------------
def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=4, connect=4, read=4, backoff_factor=1.0,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=frozenset({"GET"}), raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"})
    # INEI's mobile host serves an incomplete TLS chain; disable verification for this
    # public, read-only source and silence the warning.
    session.verify = False
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return session


def _resolve_cache(cache_dir: str | Path | None) -> Path:
    path = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------------------
# 1. List scraper
# --------------------------------------------------------------------------------------
_ROW_RE = re.compile(r'id="row_\d+"\s+rel="([^"]+\.pdf)"', re.IGNORECASE)
_DATE_RE = re.compile(r"(\d{2})/(\d{2})/(\d{4})")
_RESULT_RE = re.compile(r"Resultado\s+(\d+)\s+de\s+(\d+)\s+de\s+(\d+)")


def list_reports(cache_dir: str | Path | None = None, refresh: bool = False) -> pd.DataFrame:
    """Scrape the full paginated bulletin index.

    Returns a frame ``[report_id, pub_date, pdf_url, page]`` sorted newest first. Cached to
    ``report_index.parquet``.
    """
    cache = _resolve_cache(cache_dir)
    index_path = cache / "report_index.parquet"
    if index_path.exists() and not refresh:
        return pd.read_parquet(index_path)

    session = _build_session()
    rows: list[dict] = []
    total: int | None = None
    for page in range(1, MAX_LIST_PAGES + 1):
        resp = session.get(INEI_LIST_URL.format(page=page), timeout=DEFAULT_TIMEOUT)
        if resp.status_code >= 400:
            break
        html = resp.text
        rels = _ROW_RE.findall(html)
        if not rels:
            break
        # publication dates appear one per row, in the same order as the rows
        dates = _DATE_RE.findall(html)
        result = _RESULT_RE.search(html)
        if result:
            total = int(result.group(3))
        for i, rel in enumerate(rels):
            pub = None
            if i < len(dates):
                d, m, y = dates[i]
                pub = pd.Timestamp(int(y), int(m), int(d))
            url = rel if rel.startswith("http") else INEI_BASE + rel
            rows.append({"report_id": Path(rel).stem, "pub_date": pub, "pdf_url": url, "page": page})
        if total is not None and len(rows) >= total:
            break

    index = pd.DataFrame(rows).drop_duplicates("pdf_url")
    index = index.sort_values("pub_date", ascending=False, na_position="last").reset_index(drop=True)
    index.to_parquet(index_path)
    LOGGER.info("Scraped %d INEI bulletins (expected %s)", len(index), total)
    return index


# --------------------------------------------------------------------------------------
# 1b. gob.pe discovery (the current publication surface)
#
# INEI publishes the Avance Coyuntural on gob.pe:
#   collection:  /institucion/inei/colecciones/6034-avance-coyuntural   (?sheet=N)
#   publication: /institucion/inei/informes-publicaciones/<id>-avance-coyuntural-...-n-<N>-<mes>-<YYYY>
# The PDF lives on cdn.www.gob.pe under a DYNAMIC upload id with a cache-buster,
# so the PDF URL is always discovered from the publication page, never assumed.
# Every ingested bulletin records: publication date (the page's Spanish date
# line), bulletin URL, PDF URL, retrieval timestamp, sha256 of the PDF bytes,
# and the reference-period range of the parsed rows. Historical as-of runs
# never see bulletins published after their as-of: the slug month prunes
# without a page fetch, and the page date decides at the boundary.
# --------------------------------------------------------------------------------------
GOBPE_BASE = "https://www.gob.pe"
GOBPE_COLLECTION_URL = GOBPE_BASE + "/institucion/inei/colecciones/6034-avance-coyuntural"
_GOBPE_ITEM_RE = re.compile(
    r'href="(/institucion/inei/informes-publicaciones/'
    r'(\d+-avance-coyuntural[^"#?]*?-n-(\d+)-([a-z]+)-(\d{4})))"',
    re.IGNORECASE)
_GOBPE_PDF_RE = re.compile(
    r'href="(https://cdn\.www\.gob\.pe/[^"]+?\.pdf[^"]*)"', re.IGNORECASE)
_SPANISH_MONTHS = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5,
                   "junio": 6, "julio": 7, "agosto": 8, "setiembre": 9,
                   "septiembre": 9, "octubre": 10, "noviembre": 11,
                   "diciembre": 12}
_GOBPE_DATE_RE = re.compile(
    r"(\d{1,2})\s+de\s+(" + "|".join(_SPANISH_MONTHS) + r")\s+de\s+(\d{4})",
    re.IGNORECASE)


def _fetch_text(url: str, session: "requests.Session | None" = None) -> str:
    session = session or _build_session()
    resp = session.get(url, timeout=DEFAULT_TIMEOUT)
    if resp.status_code >= 400:
        raise INEIError(f"gob.pe request failed ({resp.status_code}): {url}")
    return resp.text


def discover_gobpe_bulletins(*, session=None, max_sheets: int = 1,
                             html_pages: list[str] | None = None) -> pd.DataFrame:
    """Avance Coyuntural items from the gob.pe collection, newest first.

    ``html_pages`` injects page texts for tests; otherwise sheets 1..max_sheets
    of the live collection are fetched. Returns ``[report_id, bulletin_no,
    slug_period, bulletin_url]`` where ``slug_period`` is the bulletin's
    nominal month from the slug (first day of month).
    """
    if html_pages is None:
        html_pages = []
        for sheet in range(1, max_sheets + 1):
            url = GOBPE_COLLECTION_URL + (f"?sheet={sheet}" if sheet > 1 else "")
            html_pages.append(_fetch_text(url, session))
    rows, seen = [], set()
    for html in html_pages:
        for path_, slug, no, mes, year in _GOBPE_ITEM_RE.findall(html):
            if slug in seen:
                continue
            seen.add(slug)
            month = _SPANISH_MONTHS.get(mes.lower())
            if month is None:
                continue
            rows.append({"report_id": slug, "bulletin_no": int(no),
                         "slug_period": pd.Timestamp(int(year), month, 1),
                         "bulletin_url": GOBPE_BASE + path_})
    out = pd.DataFrame(rows, columns=["report_id", "bulletin_no",
                                      "slug_period", "bulletin_url"])
    return out.sort_values("slug_period", ascending=False).reset_index(drop=True)


def parse_gobpe_publication(html: str) -> dict:
    """PDF URL and publication date from one gob.pe publication page."""
    pdfs = _GOBPE_PDF_RE.findall(html)
    if not pdfs:
        raise INEIError("gob.pe publication page carries no PDF link; the page "
                        "layout changed or the item has no document")
    m = _GOBPE_DATE_RE.search(html)
    pub_date = (pd.Timestamp(int(m.group(3)), _SPANISH_MONTHS[m.group(2).lower()],
                             int(m.group(1)))
                if m else pd.NaT)
    return {"pdf_url": pdfs[0], "pub_date": pub_date}


def gobpe_index_asof(*, session=None, max_sheets: int = 1,
                     html_pages: list[str] | None = None,
                     page_fetcher=None,
                     as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """The discovery index with FULL metadata, restricted to ``as_of``.

    Slug months later than the as-of month prune WITHOUT a page fetch (a
    historical run must not even request unknowable pages); the page's own
    publication date decides at the boundary. Returns ``[report_id,
    bulletin_no, slug_period, bulletin_url, pdf_url, pub_date]``.
    """
    as_of = (pd.Timestamp.now().normalize() if as_of is None
             else pd.Timestamp(as_of).normalize())
    items = discover_gobpe_bulletins(session=session, max_sheets=max_sheets,
                                     html_pages=html_pages)
    items = items[items.slug_period <= as_of].copy()
    fetch = page_fetcher or (lambda url: _fetch_text(url, session))
    keep = []
    for _, row in items.iterrows():
        meta = parse_gobpe_publication(fetch(row.bulletin_url))
        pub = meta["pub_date"]
        if pd.notna(pub) and pub > as_of:
            continue
        keep.append({**row.to_dict(), **meta})
    cols = ["report_id", "bulletin_no", "slug_period", "bulletin_url",
            "pdf_url", "pub_date"]
    out = pd.DataFrame(keep, columns=cols)
    return out.sort_values("pub_date", ascending=False,
                           na_position="last").reset_index(drop=True)


def _download_gobpe_pdf(pdf_url: str, dest_dir: Path,
                        session=None) -> Path:
    """Download one bulletin PDF (cache-buster stripped from the filename)."""
    session = session or _build_session()
    name = Path(pdf_url.split("?")[0]).name
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / name
    if path.exists() and path.stat().st_size > 0:
        return path
    resp = session.get(pdf_url, timeout=DEFAULT_TIMEOUT)
    if resp.status_code >= 400 or not resp.content.startswith(b"%PDF"):
        raise INEIError(f"Failed to download a valid PDF from {pdf_url} "
                        f"(HTTP {resp.status_code}).")
    path.write_bytes(resp.content)
    return path


def ingest_gobpe_bulletins(cache_dir: str | Path | None = None, *,
                           session=None, max_sheets: int = 1,
                           html_pages: list[str] | None = None,
                           page_fetcher=None, pdf_fetcher=None,
                           as_of: pd.Timestamp | None = None,
                           frontier_pub: pd.Timestamp | None = None,
                           parse=None) -> dict:
    """Ingest every knowable, not-yet-stored gob.pe bulletin into the lake.

    Idempotent: stored report_ids never re-download; each new bulletin writes
    its own immutable vintage file (previous vintages are never touched) and a
    provenance row in ``gobpe_index.parquet`` (publication date, bulletin URL,
    PDF URL, retrieval timestamp, sha256, reference-period range).
    ``page_fetcher``/``pdf_fetcher``/``parse`` are injectable for tests.
    """
    cache = _resolve_cache(cache_dir)
    idx = gobpe_index_asof(session=session, max_sheets=max_sheets,
                           html_pages=html_pages, page_fetcher=page_fetcher,
                           as_of=as_of)
    ipath = cache / "gobpe_index.parquet"
    prior = pd.read_parquet(ipath) if ipath.exists() else pd.DataFrame()
    # duplicate rule: a bulletin counts as ingested when its VINTAGE is stored
    # OR its provenance row exists (a bulletin whose parse matched zero rows
    # must not be re-downloaded forever; re-parsing is a deliberate act)
    stored = set(_stored_report_ids(cache))
    if len(prior):
        stored |= set(prior["report_id"])
    if frontier_pub is None:
        frontier_pub = _stored_frontier(cache)
    todo = _reports_to_ingest(idx, stored, frontier_pub)
    # id migration: the old portal named bulletins by PDF stem, gob.pe by
    # slug. A publication DATE already represented in the lake is the same
    # monthly bulletin under a different name; never ingest it twice.
    if len(todo):
        todo = todo[~todo.pub_date.isin(_stored_pub_dates(cache))]

    fetch_pdf = pdf_fetcher or (lambda url, dest: _download_gobpe_pdf(
        url, dest, session=session))
    parse_fn = parse or _parse_report_to_vintage
    (cache / "pdf").mkdir(parents=True, exist_ok=True)   # fetcher contract
    added, failures, records = [], [], []
    for _, row in todo.iterrows():
        try:
            pdf = fetch_pdf(row.pdf_url, cache / "pdf")
            data = Path(pdf).read_bytes()
            part, _um = parse_fn(Path(pdf), row.report_id, row.pub_date)
            refs = (pd.to_datetime(part["reference_period"], errors="coerce")
                    if "reference_period" in part
                    else pd.Series(dtype="datetime64[ns]"))
            refs = refs.dropna()
            if not part.empty:
                _write_vintage(cache, row.report_id, row.pub_date, part)
            records.append({
                "report_id": row.report_id, "bulletin_no": int(row.bulletin_no),
                "bulletin_url": row.bulletin_url, "pdf_url": row.pdf_url,
                "pub_date": row.pub_date,
                "retrieved_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                "sha256": hashlib.sha256(data).hexdigest(),
                "n_rows": int(len(part)),
                "ref_min": str(refs.min().date()) if len(refs) else None,
                "ref_max": str(refs.max().date()) if len(refs) else None,
            })
            added.append(row.report_id)
        except Exception as exc:
            failures.append((row.report_id, f"{type(exc).__name__}: {exc}"))
            LOGGER.warning("gob.pe ingest failed for %s: %s", row.report_id, exc)
    if records:
        idx_out = pd.concat([prior, pd.DataFrame(records)], ignore_index=True)
        idx_out = idx_out.drop_duplicates("report_id", keep="first")
        idx_out.to_parquet(ipath)
    if added and parse is None:
        _rebuild_from_store(cache)
    return {"n_new_reports": len(added), "added": added, "failures": failures}


# --------------------------------------------------------------------------------------
# 2. Downloader
# --------------------------------------------------------------------------------------
def download_report(pdf_url: str, cache_dir: str | Path | None = None,
                    refresh: bool = False, session: requests.Session | None = None) -> Path:
    """Download one bulletin PDF into ``<cache>/pdf/`` (cached)."""
    cache = _resolve_cache(cache_dir)
    pdf_dir = cache / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    path = pdf_dir / (Path(pdf_url).name)
    if path.exists() and not refresh and path.stat().st_size > 0:
        return path
    session = session or _build_session()
    resp = session.get(pdf_url, timeout=DEFAULT_TIMEOUT)
    if resp.status_code >= 400 or not resp.content.startswith(b"%PDF"):
        raise INEIError(f"Failed to download a valid PDF from {pdf_url} (HTTP {resp.status_code}).")
    path.write_bytes(resp.content)
    return path


# --------------------------------------------------------------------------------------
# 2b. Report selection (skip redundant bulletins)
# --------------------------------------------------------------------------------------
def select_reports_for_history(index: pd.DataFrame, overlap_months: int = 24,
                               window_months: int = 36) -> pd.DataFrame:
    """Newest-first greedy subset whose ~3-year windows tile the whole history.

    Each bulletin restates ``window_months`` of history, so walking newest to oldest and
    keeping a bulletin only when it reaches ``window_months - overlap_months`` further back
    than the last kept one covers 2003-today with the most-updated values. The default
    ``overlap_months=24`` steps ~12 months between kept bulletins (~23 files), enough that
    the within-bulletin (base-safe) YoY is continuous with no 12-month holes.
    """
    ordered = index.sort_values("pub_date", ascending=False).dropna(subset=["pub_date"])
    step = pd.DateOffset(months=max(1, window_months - overlap_months))
    kept, next_anchor = [], None
    for _, row in ordered.iterrows():
        if next_anchor is None or row["pub_date"] <= next_anchor:
            kept.append(row)
            next_anchor = row["pub_date"] - step
    return pd.DataFrame(kept).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# 3. PDF parsing (poppler bbox word coordinates)
# --------------------------------------------------------------------------------------
def _require_pdftotext() -> str:
    exe = shutil.which("pdftotext")
    if exe is None:
        raise INEIError("pdftotext (poppler) is required. Install with 'brew install poppler' "
                        "or 'apt-get install poppler-utils'.")
    return exe


def _local(tag: str) -> str:
    return tag.split("}")[-1]


def _pdf_pages_words(pdf_path: str | Path) -> list[list[dict]]:
    """Extract words with coordinates per page via ``pdftotext -bbox-layout``."""
    exe = _require_pdftotext()
    out = subprocess.run([exe, "-bbox-layout", str(pdf_path), "-"],
                         capture_output=True, text=True, timeout=120)
    if out.returncode != 0 or not out.stdout.strip():
        raise INEIError(f"pdftotext failed for {pdf_path}: {out.stderr[:200]}")
    root = ET.fromstring(out.stdout)
    pages = []
    for page in (el for el in root.iter() if _local(el.tag) == "page"):
        words = []
        for w in (el for el in page.iter() if _local(el.tag) == "word"):
            if not (w.text and w.text.strip()):
                continue
            words.append({"text": w.text.strip(),
                          "x0": float(w.get("xMin")), "x1": float(w.get("xMax")),
                          "top": float(w.get("yMin")), "bottom": float(w.get("yMax"))})
        pages.append(words)
    return pages


def _to_num(token: str) -> float | None:
    """Parse an INEI numeric token to float, tolerant of both decimal conventions.

    INEI separates thousands with a SPACE (e.g. ``73 327``; these arrive as separate tokens
    that the parser has already joined), so a ``.`` or ``,`` inside a token is the decimal
    mark. Crucially, bulletins are inconsistent: some print ``116,3`` (comma decimal) and
    others ``149.0`` (dot decimal) for the same series -- assuming ``.`` were a thousands
    separator turns ``149.0`` into ``1490`` (a silent 10x error). Rules: if both marks appear,
    the rightmost is the decimal; a lone mark is the decimal unless it has exactly three
    trailing digits (``1.490`` -> a thousands group, since decimals here have 1-2 places);
    repeated marks are thousands.
    """
    t = token.replace(" ", "").replace("−", "-").replace("–", "-")
    if not re.search(r"\d", t) or not re.fullmatch(r"-?[\d.,]+", t):
        return None
    if "," in t and "." in t:
        t = t.replace(".", "").replace(",", ".") if t.rfind(",") > t.rfind(".") else t.replace(",", "")
    elif t.count(",") + t.count(".") == 1:
        sep = "," if "," in t else "."
        t = t.replace(sep, "") if re.search(re.escape(sep) + r"\d{3}$", t) else t.replace(sep, ".")
    else:
        t = t.replace(",", "").replace(".", "")   # repeated marks -> thousands groups
    return float(t) if re.fullmatch(r"-?\d+(\.\d+)?", t) else None


def _group_lines(words: list[dict], ytol: float = 4.0) -> list[dict]:
    """Cluster words into visual rows by vertical centre (tolerant to sub-pixel jitter)."""
    lines: list[dict] = []
    for w in sorted(words, key=lambda w: ((w["top"] + w["bottom"]) / 2, w["x0"])):
        c = (w["top"] + w["bottom"]) / 2
        for ln in lines:
            if abs(ln["c"] - c) <= ytol:
                ln["words"].append(w)
                ln["c"] = (ln["c"] * ln["n"] + c) / (ln["n"] + 1)
                ln["n"] += 1
                break
        else:
            lines.append({"c": c, "n": 1, "words": [w]})
    for ln in lines:
        ln["words"].sort(key=lambda w: w["x0"])
        ln["text"] = " ".join(w["text"] for w in ln["words"])
    return sorted(lines, key=lambda ln: ln["c"])


def _month_of(word: str) -> int | None:
    return SPANISH_MONTHS.get(_normalize(word).lower()[:3])


def parse_report(pdf_path: str | Path) -> pd.DataFrame:
    """Parse one bulletin into tidy rows.

    Columns: ``[reference_period, indicator_key, section, label, raw_title, base_year,
    value_level, var_pct_printed, matched]``. ``reference_period`` is a monthly
    ``pd.Period``. The frame carries ``attrs['reference_month']`` (the report's headline
    month, derived from the data) and ``attrs['unmatched_titles']``.
    """
    pages = _pdf_pages_words(pdf_path)
    records: list[dict] = []
    unmatched: set[str] = set()

    for words in pages:
        if not words:
            continue
        lines = _group_lines(words)
        for i, ln in enumerate(lines):
            norm_line = _normalize(ln["text"])
            year_words = [w for w in ln["words"] if re.fullmatch(r"(19|20)\d{2}", w["text"])]
            if "MES" not in norm_line.split() or len(year_words) < 2:
                continue  # a real table header carries the "MES" label and >=2 year columns
            # keep only the consecutive year block (drop stray tokens like a base year
            # "1994" or a value that happens to look like a year "1950")
            max_year = max(int(w["text"]) for w in year_words)
            cols = sorted(((w["x0"] + w["x1"]) / 2, int(w["text"]))
                          for w in year_words if 0 <= max_year - int(w["text"]) <= 5)
            if len(cols) < 2:
                continue
            xs = [c[0] for c in cols]
            lo, hi = xs[0] - 40, xs[-1] + 95  # table column span; excludes the chart

            rows_out: dict[int, dict] = {}
            for ln2 in lines[i + 1:i + 24]:
                head = ln2["words"][0]["text"]
                low = _normalize(head).lower()
                if low.startswith("prom"):
                    break
                if "MES" in _normalize(ln2["text"]).split() and \
                   sum(bool(re.fullmatch(r"(19|20)\d{2}", w["text"])) for w in ln2["words"]) >= 2:
                    break  # next table
                month = _month_of(head)
                if month is None:
                    continue
                # A value like "73 327" is emitted as two words ("73", "327"). Bin every
                # token to its nearest year column (or the VAR% column to the right), then
                # join the tokens that share a column so split thousands are rejoined, not
                # truncated to one fragment. The half-spacing gate keeps a wide two-token
                # number in its own column without bleeding into the neighbour.
                half = min(b - a for a, b in zip(xs, xs[1:])) / 2.0 if len(xs) > 1 else 30.0
                bucket: dict = {}
                for w in ln2["words"][1:]:
                    xc = (w["x0"] + w["x1"]) / 2
                    if not (lo <= xc <= hi):
                        continue
                    if not (re.search(r"\d", w["text"]) or w["text"] in {"-", "−", "–"}):
                        continue
                    near = min(cols, key=lambda c: abs(c[0] - xc))
                    if abs(near[0] - xc) <= half:
                        bucket.setdefault(near[1], []).append((w["x0"], w["text"]))
                    elif xc > xs[-1] + half:
                        bucket.setdefault("var", []).append((w["x0"], w["text"]))
                vals: dict = {}
                for col_key, toks in bucket.items():
                    num = _to_num("".join(t for _, t in sorted(toks)))
                    if num is not None:
                        vals[col_key] = num
                if vals:
                    rows_out[month] = vals
            if len(rows_out) < 6:
                continue  # not a genuine data table (e.g. a chart legend)

            # Text just below the table (bounded to before the next table/section), used as
            # a fallback identity when the table carries no usable title of its own.
            below_parts = []
            for ln3 in lines[i + 1:i + 30]:
                nt = _normalize(ln3["text"])
                is_hdr = "MES" in nt.split() and sum(
                    bool(re.fullmatch(r"(19|20)\d{2}", w["text"])) for w in ln3["words"]) >= 2
                if is_hdr or re.match(r"[IVXL]+ INDICADOR", nt):
                    break
                below_parts.append(nt)
            below_text = " ".join(below_parts)

            base_year, title_parts = None, []
            for prev in lines[max(0, i - 7):i]:
                ntext = _normalize(prev["text"])
                mb = re.search(r"A.O\s+BASE\s+(\d{4})", ntext)
                if mb:
                    base_year = int(mb.group(1))
                letters = [c for c in prev["text"] if c.isalpha()]
                boiler = ("AVANCE COYUNTURAL" in ntext or ntext.startswith("INDICADORES")
                          or "INFORME TECNICO" in ntext or "ACTIVIDAD ECONOMICA" in ntext)
                if len(prev["text"].strip()) > 6 and letters and \
                   sum(c.isupper() for c in letters) / len(letters) > 0.75 and \
                   "BASE" not in ntext and not boiler:
                    title_parts.append(prev["text"].strip())
            # titles often wrap over two lines; join the (deduped) caps lines just above
            # the header. The chart repeats the indicator title, so drop exact duplicates.
            seen_tp, uniq_tp = set(), []
            for t in title_parts:
                if t not in seen_tp:
                    seen_tp.add(t)
                    uniq_tp.append(t)
            raw_title = " ".join(uniq_tp[-3:]) if uniq_tp else "UNKNOWN"
            # drop a leading section header such as "I. INDICADORES DE PRODUCCION"
            raw_title = re.sub(r"^\s*[IVX]+\.\s*INDICADOR(?:ES)?\s+(?:DE[L]?\s+)?"
                               r"(PRODUCCION|PRECIOS|GESTION\s+FISCAL|SECTOR\s+EXTERNO|"
                               r"FINANCIEROS?|EMPLEO|TURISMO|CONSUMO)\b\s*", "",
                               raw_title, flags=re.IGNORECASE)
            raw_title = re.sub(r"\s+", " ", raw_title).strip() or "UNKNOWN"
            key, section, label, matched = classify_title(raw_title)
            # Recent bulletins drop the "venta e inmatriculación de vehículos nuevos" title and
            # leave the table under the bare "VII. INDICADOR DE CONSUMO" section header; it is
            # then identifiable only by the SUNARP inmatriculación footnote beneath it.
            if not matched and "INMATRICULACION" in below_text and "SUNARP" in below_text:
                key, section, label, matched = (
                    "venta_vehiculos", "otros",
                    "Venta / inmatriculación de vehículos nuevos", True)
            if not matched:
                unmatched.add(raw_title)

            for _, yr in cols:
                for month, vals in rows_out.items():
                    if yr not in vals:
                        continue
                    records.append({
                        "reference_period": pd.Period(f"{yr}-{month:02d}", freq="M"),
                        "indicator_key": key, "section": section, "label": label,
                        "raw_title": raw_title, "base_year": base_year,
                        "value_level": vals[yr],
                        "var_pct_printed": vals.get("var") if yr == cols[-1][1] else None,
                        "matched": matched,
                    })

    frame = pd.DataFrame.from_records(records)
    if not frame.empty:
        frame = frame.drop_duplicates(["indicator_key", "reference_period"], keep="first")
        # headline month = latest production-block month (avoids WTI/financial leaking a later month)
        prod = frame[frame["section"] == "produccion"]
        frame.attrs["reference_month"] = (prod if not prod.empty else frame)["reference_period"].max()
    frame.attrs["unmatched_titles"] = sorted(unmatched)
    return frame


# --------------------------------------------------------------------------------------
# 4-5. Vintage store, unifier, consistency
# --------------------------------------------------------------------------------------
VINTAGE_COLUMNS = ["report_id", "pub_date", "report_ref_month", "indicator_key", "section",
                   "label", "raw_title", "reference_period", "base_year", "value_level",
                   "var_pct_printed"]


def _parse_report_to_vintage(pdf_path: Path, report_id: str, pub_date) -> tuple[pd.DataFrame, list[str]]:
    """Parse a PDF once; return its vintage rows and its unmatched titles."""
    parsed = parse_report(pdf_path)
    unmatched = list(parsed.attrs.get("unmatched_titles", []))
    if parsed.empty:
        return pd.DataFrame(columns=VINTAGE_COLUMNS), unmatched
    ref_month = parsed.attrs.get("reference_month")
    parsed = parsed.assign(
        report_id=report_id,
        pub_date=pd.Timestamp(pub_date) if pd.notna(pub_date) else pd.NaT,
        report_ref_month=str(ref_month) if ref_month is not None else None,
    )
    parsed["reference_period"] = parsed["reference_period"].astype(str)  # parquet-safe
    return parsed[VINTAGE_COLUMNS], unmatched


def _load_overrides(override_dir: str | Path | None = None) -> dict[str, pd.Series]:
    """Authoritative external series that override parsed values where they overlap.

    Each committed CSV in ``utils/inei_overrides/`` is named ``<indicator_key>.csv`` with
    columns ``reference_period`` (``YYYY-MM``) and ``value_level``. These are curated ground
    truth (e.g. the AAP/SUNARP new-vehicle registrations for ``venta_vehiculos``, whose recent
    table the bulletins print without a title) that win over the PDF-parsed series in their
    date range; the parsed layer still fills months outside it, so new bulletins keep
    extending the recent end. Returns ``{indicator_key: Series indexed by monthly Period}``.
    """
    directory = Path(override_dir) if override_dir is not None else DEFAULT_OVERRIDE_DIR
    out: dict[str, pd.Series] = {}
    for csv in (sorted(directory.glob("*.csv")) if directory.exists() else []):
        df = pd.read_csv(csv)
        s = pd.Series(pd.to_numeric(df["value_level"], errors="coerce").to_numpy(),
                      index=pd.PeriodIndex(df["reference_period"].astype(str), freq="M"))
        out[csv.stem] = s.dropna().sort_index()
    return out


def _within_report_yoy(vintages: pd.DataFrame) -> pd.DataFrame:
    """Base-safe YoY: compute it inside each bulletin (one base year), then latest wins.

    A single bulletin shares one base year, so ``level_m / level_{m-12} - 1`` from that
    bulletin is never contaminated by a rebasing. Latest publication wins per month.
    """
    parts = []
    for (_rid, _ind), g in vintages.groupby(["report_id", "indicator_key"], sort=False):
        s = pd.to_numeric(g.set_index("reference_period")["value_level"], errors="coerce").sort_index()
        denom = s.shift(12).where(lambda d: d != 0)      # avoid division by zero
        yy = (s / denom - 1.0) * 100.0
        part = g[["pub_date", "indicator_key", "reference_period"]].copy()
        part["yoy"] = part["reference_period"].map(yy)
        parts.append(part)
    long = pd.concat(parts, ignore_index=True).dropna(subset=["yoy"])
    long = long.sort_values("pub_date").drop_duplicates(
        ["indicator_key", "reference_period"], keep="last")
    return long.pivot(index="reference_period", columns="indicator_key", values="yoy").sort_index()


def _chain_levels(yoy_col: pd.Series, level_col: pd.Series, seed_months: int = 24) -> pd.Series:
    """Reconstruct a continuous single-base level index from growth rates.

    Anchor on the most recent ``seed_months`` of actual levels (which are on the current
    base) and back-cast earlier months with ``level_{m} = level_{m+12} / (1 + yoy_{m+12})``.
    This yields one coherent level path on the latest base, with seasonality preserved and
    no rebasing jumps. New bulletins extend the recent (anchor) end going forward.
    """
    recon = pd.Series(index=level_col.index, dtype=float)
    actual = level_col.dropna()
    if actual.empty:
        return recon
    last = actual.index.max()
    seed_lo = last - (seed_months - 1)
    for p in level_col.index:
        if seed_lo <= p <= last and pd.notna(level_col[p]):
            recon[p] = level_col[p]
    growth = yoy_col / 100.0
    for p in sorted((p for p in level_col.index if p < seed_lo), reverse=True):
        nxt = p + 12
        g = growth.get(nxt)
        if nxt in recon.index and pd.notna(recon.get(nxt)) and pd.notna(g) and (1.0 + g) != 0:
            recon[p] = recon[nxt] / (1.0 + g)
    return recon


def _index_from_yoy(yoy_pct: pd.Series, base: float = 100.0) -> pd.Series:
    """Build a monthly index from a published year-on-year % series.

    Some indicators (IPC, tipo de cambio) are only published as a year-on-year variation,
    with no level in the bulletin. Seed the first available 12 months to ``base`` and chain
    forward month-on-same-month (``level_m = level_{m-12} * (1 + yoy_m / 100)``) to recover a
    continuous index whose base is its first year = 100.
    """
    g = pd.to_numeric(yoy_pct, errors="coerce")
    idx = pd.Series(index=g.index, dtype=float)
    first = g.first_valid_index()
    if first is None:
        return idx
    start = g.index.get_loc(first)
    seed = g.index[start:start + 12]
    idx.loc[seed] = base
    for p in g.index[start + 12:]:
        prev, r = p - 12, g.get(p)
        if prev in idx.index and pd.notna(idx.get(prev)) and pd.notna(r):
            idx[p] = idx[prev] * (1.0 + r / 100.0)
    return idx


def _longest_contiguous_run(series: pd.Series) -> pd.Series:
    """Keep only the longest gap-free run; blank everything outside it.

    Drops disconnected fragments (a lone 2001-2004 tail from the oldest bulletin, or a
    stray recent month after a gap) so the delivered series has no internal holes.
    """
    valid = series.notna().to_numpy()
    if not valid.any():
        return series
    best_start, best_end, best_len, cur_start = 0, -1, 0, None
    for i, v in enumerate(valid):
        if v:
            cur_start = i if cur_start is None else cur_start
            if i - cur_start + 1 > best_len:
                best_len, best_start, best_end = i - cur_start + 1, cur_start, i
        else:
            cur_start = None
    out = series.copy()
    keep = series.index[best_start:best_end + 1]
    return out.where(out.index.isin(keep))


def build_unified_series(vintages: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Unify vintages into the most-updated series per indicator.

    Returns ``{"yoy", "level_latest", "level_chained"}`` (all wide, monthly ``PeriodIndex``):

    - ``yoy``: year-on-year growth. Base-safe (computed within each bulletin, latest vintage
      wins) for index/level/rate indicators; for ``"variation"`` indicators (IPC, FX) it is
      the published annual variation itself. The primary growth series.
    - ``level_latest``: the raw latest-vintage value. Contiguous, but index indicators still
      carry INEI's base-year breaks (1994 -> 2007 -> 2012), and variation indicators here are
      the published % change, not a level.
    - ``level_chained``: the clean level series, built per indicator ``unit`` (see
      ``IndicatorRule``): rebased indices are reconstructed onto a single base from base-safe
      growth; money/price/rate levels are used directly (they never rebase); variation
      indicators are turned into a base-100 index by chaining their growth. No rebasing jumps.
    """
    empty = {"yoy": pd.DataFrame(), "level_latest": pd.DataFrame(), "level_chained": pd.DataFrame()}
    if vintages.empty:
        return empty
    v = vintages.copy()
    v["reference_period"] = pd.PeriodIndex(v["reference_period"].astype(str), freq="M")
    v["value_level"] = pd.to_numeric(v["value_level"], errors="coerce")
    latest = v.sort_values("pub_date").drop_duplicates(
        ["indicator_key", "reference_period"], keep="last")
    lvl_wide = latest.pivot(index="reference_period", columns="indicator_key",
                            values="value_level").sort_index()
    yoy_wide = _within_report_yoy(v)

    lo = min(lvl_wide.index.min(), yoy_wide.index.min())
    hi = max(lvl_wide.index.max(), yoy_wide.index.max())
    full = pd.period_range(lo, hi, freq="M")
    lvl_wide = lvl_wide.reindex(full)

    # Robust cross-vintage consensus, used to build the clean levels (kept separate from
    # lvl_wide, whose latest-vintage-wins semantics the raw `level_latest` output needs). Per
    # month, drop vintages far from the median as parse outliers (e.g. a stray reservas figure
    # landing in imports), then keep the month only if the surviving inliers are a majority,
    # valued at their median. Corroborated months (incl. real shocks) survive; a lone poison
    # vintage no longer masks the month, and single-vintage months are taken as-is.
    vv = v[["reference_period", "indicator_key", "value_level"]].dropna(subset=["value_level"])
    key = ["reference_period", "indicator_key"]
    med_s = vv.groupby(key)["value_level"].transform("median")
    vv = vv.assign(_in=((vv["value_level"] - med_s).abs()
                        / med_s.abs().replace(0, pd.NA)) <= _VINTAGE_TOL)
    gk = vv.groupby(key)
    keep = gk["_in"].sum() > gk["value_level"].count() / 2.0
    val = vv[vv["_in"]].groupby(key)["value_level"].median()
    _wide = lambda s: s.unstack("indicator_key").reindex(index=full, columns=lvl_wide.columns)
    consensus = _wide(val.where(keep))
    # Base-safe YoY has sparse gaps (a bulletin cannot form the ratio for its oldest year);
    # the latest-vintage level series is contiguous, so its implied YoY fills those gaps.
    # The base-safe value still wins where present, so rebasings are handled correctly and
    # the fill only bridges same-base gaps -> a continuous, base-correct YoY.
    yoy_within = yoy_wide.reindex(index=full, columns=lvl_wide.columns)
    level_implied = (lvl_wide / lvl_wide.shift(12) - 1.0) * 100.0
    yoy_final = yoy_within.fillna(level_implied)

    # "Variation" indicators (IPC, tipo de cambio) are published as a year-on-year %. A few
    # bulletins misprint them or tabulate the index instead of the change, which would win
    # under latest-vintage and then compound in the chained index. Take the median across
    # vintages per month (robust to a minority of bad bulletins) and drop residual extremes.
    variation = [c for c in lvl_wide.columns if _unit_of(c) == "variation"]
    if variation:
        vr = v[v["indicator_key"].isin(variation)]
        # Some whole bulletins tabulate the index instead of the % change; their values are
        # implausibly large for a variation. Drop such (report, indicator) blocks wholesale
        # (by their median) before aggregating, so they neither win nor punch holes.
        rep_med = vr.groupby(["report_id", "indicator_key"])["value_level"].transform("median")
        vr = vr[(rep_med >= _VARIATION_MIN) & (rep_med <= _VARIATION_MAX)]
        med = (vr.groupby(["reference_period", "indicator_key"])["value_level"].median()
               .unstack("indicator_key").reindex(index=full, columns=variation))
        med = med.where((med >= _VARIATION_MIN) & (med <= _VARIATION_MAX))
        for c in variation:
            yoy_final[c] = med[c]

    # Clean level per indicator, routed by unit (see IndicatorRule), built on the consensus.
    # Keep the untruncated column (`level_raw`) as well, so an override can bridge parsed
    # segments before the longest-gap-free run is taken (otherwise pre-override truncation
    # discards history the override would reconnect).
    level_cols, level_raw = {}, {}
    for c in lvl_wide.columns:
        unit = _unit_of(c)
        if unit == "index":                         # rebased -> reconstruct single base
            col = _chain_levels(yoy_final[c], consensus[c])
        elif unit == "variation":                   # only a % is published -> base-100 index
            col = _index_from_yoy(yoy_final[c])
        else:                                        # money / price / rate: already a level
            col = consensus[c]
        level_raw[c] = col
        level_cols[c] = _longest_contiguous_run(col)
    chained = pd.DataFrame(level_cols)
    yoy_final = yoy_final.apply(_longest_contiguous_run)

    # Authoritative external overrides win over the parsed level in their range; the parsed
    # layer (untruncated) still fills months outside it -- older history and future bulletins.
    for key, ext_raw in _load_overrides().items():
        ext = ext_raw.reindex(full)
        base = ext.combine_first(level_raw[key]) if key in level_raw else ext
        chained[key] = _longest_contiguous_run(base)
        denom = chained[key].shift(12).where(lambda d: d != 0)
        yoy_final[key] = _longest_contiguous_run((chained[key] / denom - 1.0) * 100.0)
    return {"yoy": yoy_final, "level_latest": lvl_wide, "level_chained": chained}


def clean_indicators(level_chained: pd.DataFrame, min_months: int = 108,
                     recent_within: int = 18) -> list[str]:
    """The most coherent and consistent indicators.

    Selects columns of the (gap-free-per-column) reconstructed level frame whose run is
    long (``>= min_months``, i.e. at least ~9 years) and reaches near the present (last
    observation within ``recent_within`` months of the newest data). This drops indicators
    with isolated source gaps (e.g. pesca / hidrocarburos, which have scattered missing
    months in the bulletins), discontinued ones (empleo, fabril no primario), series whose
    recent vintages are unreliable (they end too early), and the untitled ``unknown`` block.
    """
    if level_chained is None or level_chained.empty:
        return []
    last = level_chained.index.max()
    keep = []
    for column in level_chained.columns:
        if column == "unknown":
            continue
        series = level_chained[column].dropna()
        if len(series) >= min_months and (last - series.index.max()).n <= recent_within:
            keep.append(column)
    return keep


def rank_indicator_consistency(vintages: pd.DataFrame) -> pd.DataFrame:
    """Rank indicators by publication consistency, tracking title variants over time."""
    if vintages.empty:
        return pd.DataFrame()
    v = vintages.copy()
    v["reference_period"] = pd.PeriodIndex(v["reference_period"].astype(str), freq="M")
    rows = []
    for key, g in v.groupby("indicator_key"):
        periods = g["reference_period"]
        span = pd.period_range(periods.min(), periods.max(), freq="M")
        covered = periods.nunique()
        rows.append({
            "indicator_key": key,
            "section": g["section"].iloc[0],
            "label": g["label"].iloc[0],
            "first_period": str(periods.min()),
            "last_period": str(periods.max()),
            "n_reports": g["report_id"].nunique(),
            "months_covered": int(covered),
            "months_in_span": len(span),
            "pct_months_covered": round(covered / len(span), 3) if len(span) else 0.0,
            "name_variants": " | ".join(sorted(g["raw_title"].dropna().unique())),
            "latest_base_year": g.sort_values("pub_date")["base_year"].dropna().iloc[-1]
            if g["base_year"].notna().any() else None,
        })
    return (pd.DataFrame(rows)
            .sort_values(["months_covered", "pct_months_covered"], ascending=False)
            .reset_index(drop=True))


# --------------------------------------------------------------------------------------
# 6. Public API
# --------------------------------------------------------------------------------------
def _store_paths(cache: Path) -> dict[str, Path]:
    return {"vintages": cache / "inei_vintages.parquet",
            "yoy": cache / "inei_yoy_latest.parquet",
            "level": cache / "inei_level_latest.parquet",
            "level_chained": cache / "inei_level_chained.parquet",
            "levels_clean": cache / "inei_levels_clean.parquet",
            "yoy_clean": cache / "inei_yoy_clean.parquet",
            "consistency": cache / "inei_indicator_consistency.csv",
            "unmapped": cache / "unmapped_titles.csv"}


# --------------------------------------------------------------------------------------
# Vintage lake: one immutable parquet per bulletin (the source of truth). Every merged /
# unified / clean output is *derived* from these files and can be rebuilt at any time, so a
# re-run never clobbers a captured vintage. Each new bulletin restates the prior ~3 years, so
# its file also carries that bulletin's *revisions* of earlier months, which the unifier then
# resolves (latest-vintage-wins for the raw level; cross-vintage consensus for the clean one).
# --------------------------------------------------------------------------------------
_VINTAGE_NAME_DATE = "%Y-%m-%d"          # fixed-width (10 chars) publication-date filename prefix
_VINTAGE_NAME_PREFIX_LEN = 12            # len("YYYY-MM-DD__")


def _vintage_dir(cache: Path) -> Path:
    path = cache / "vintages"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _vintage_name(report_id: str, pub_date) -> str:
    """``<pub_date>__<report_id>.parquet`` (date prefix so the folder sorts chronologically)."""
    day = pd.Timestamp(pub_date).strftime(_VINTAGE_NAME_DATE) if pd.notna(pub_date) else "0000-00-00"
    return f"{day}__{report_id}.parquet"


def _report_id_from_name(name: str) -> str:
    return name[_VINTAGE_NAME_PREFIX_LEN:-len(".parquet")]


def _write_vintage(cache: Path, report_id: str, pub_date, df: pd.DataFrame) -> Path:
    path = _vintage_dir(cache) / _vintage_name(report_id, pub_date)
    out = df.copy()
    out.attrs = {}          # parse_report leaves a Period in attrs; parquet can't JSON it
    out.to_parquet(path, index=False)
    return path


def _stored_report_ids(cache: Path) -> set[str]:
    directory = cache / "vintages"
    return {_report_id_from_name(p.name) for p in directory.glob("*.parquet")} \
        if directory.exists() else set()


def _stored_pub_dates(cache: Path) -> set[pd.Timestamp]:
    """Publication dates already in the lake (read from filenames, no file I/O)."""
    directory = cache / "vintages"
    dates: set[pd.Timestamp] = set()
    for p in (directory.glob("*.parquet") if directory.exists() else []):
        try:
            dates.add(pd.Timestamp(p.name[:10]))
        except ValueError:
            continue
    return dates


def _stored_frontier(cache: Path) -> pd.Timestamp:
    """Newest publication date already in the lake."""
    dates = _stored_pub_dates(cache)
    return max(dates) if dates else pd.NaT


def _load_all_vintages(cache: Path) -> pd.DataFrame:
    files = sorted((cache / "vintages").glob("*.parquet")) if (cache / "vintages").exists() else []
    if not files:
        return pd.DataFrame(columns=VINTAGE_COLUMNS)
    frame = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    return frame.drop_duplicates(["report_id", "indicator_key", "reference_period"], keep="last")


def _read_vintages(cache: Path) -> pd.DataFrame:
    """Merged vintages: the per-bulletin lake is authoritative; fall back to the derived
    merged file only when the lake is absent (e.g. a store written before the lake existed)."""
    if _stored_report_ids(cache):
        return _load_all_vintages(cache)
    merged = _store_paths(cache)["vintages"]
    return pd.read_parquet(merged) if merged.exists() else pd.DataFrame(columns=VINTAGE_COLUMNS)


def _reports_to_ingest(index: pd.DataFrame, stored_ids, frontier_pub=None) -> pd.DataFrame:
    """Bulletins to ingest on an update: those whose vintage is not yet stored (by
    ``report_id``) and, when a frontier publication date is given, published on or after it.

    Selecting by "what we do not already have" (not a single date cursor) means a skipped
    month is caught up on the next run and bulletins sharing a publication date are never
    missed; the frontier keeps the update from re-pulling the older bulletins that the sparse
    history build deliberately skipped.
    """
    fresh = index[~index["report_id"].isin(set(stored_ids))]
    if frontier_pub is not None and pd.notna(frontier_pub):
        fresh = fresh[fresh["pub_date"] >= frontier_pub]
    return fresh.sort_values("pub_date").reset_index(drop=True)


def _rebuild_outputs(vintages: pd.DataFrame, cache: Path) -> None:
    paths = _store_paths(cache)
    vintages.to_parquet(paths["vintages"], index=False)   # derived merged convenience copy
    unified = build_unified_series(vintages)
    unified["yoy"].to_parquet(paths["yoy"])
    unified["level_latest"].to_parquet(paths["level"])
    unified["level_chained"].to_parquet(paths["level_chained"])
    rank_indicator_consistency(vintages).to_csv(paths["consistency"], index=False)
    # the clean deliverable: reconstructed levels (and matching YoY) for the consistent set
    keep = clean_indicators(unified["level_chained"])
    unified["level_chained"][keep].to_parquet(paths["levels_clean"])
    unified["yoy"].reindex(columns=keep).to_parquet(paths["yoy_clean"])


def _rebuild_from_store(cache: Path) -> pd.DataFrame:
    """Rebuild every derived output from the vintage lake; return the merged frame."""
    vintages = _load_all_vintages(cache)
    _rebuild_outputs(vintages, cache)
    return vintages


def build_inei_history(cache_dir: str | Path | None = None, density: str = "history",
                       refresh: bool = False, overlap_months: int = 30) -> dict:
    """One-off build: scrape, select the bulletin subset, parse, write the vintage lake.

    Writes one immutable parquet per bulletin under ``<cache>/vintages/`` and rebuilds the
    derived merged / unified / clean outputs from the whole lake. ``density="history"`` uses
    the newest-first tiling subset (``overlap_months=30`` -> ~29 bulletins); ``density="all"``
    processes every bulletin. A denser subset gives more vintages per month, which the
    consensus needs to keep the less-frequently-tabulated series (e.g. import sub-categories)
    gap-free and to outvote stray misreads; raise ``overlap_months`` for more. Re-running
    re-parses the selected bulletins (so a parser fix propagates) and leaves any other lake
    files untouched; delete ``<cache>/vintages/`` for a clean slate.
    """
    cache = _resolve_cache(cache_dir)
    index = list_reports(cache_dir=cache, refresh=refresh)
    selected = index if density == "all" else select_reports_for_history(index, overlap_months)
    session = _build_session()
    unmatched, added, failures = set(), [], []
    for _, row in selected.iterrows():
        try:
            pdf = download_report(row["pdf_url"], cache_dir=cache, session=session, refresh=refresh)
            part, um = _parse_report_to_vintage(pdf, row["report_id"], row["pub_date"])
            if not part.empty:
                _write_vintage(cache, row["report_id"], row["pub_date"], part)
                added.append(row["report_id"])
            unmatched |= set(um)
        except Exception as exc:  # never fail the whole build on one bad PDF
            failures.append((row["report_id"], f"{type(exc).__name__}: {exc}"))
            LOGGER.warning("parse failed for %s: %s", row["report_id"], exc)
    vintages = _rebuild_from_store(cache)
    pd.DataFrame(sorted(unmatched), columns=["unmatched_title"]).to_csv(
        _store_paths(cache)["unmapped"], index=False)
    return {"n_reports_selected": len(selected), "n_reports_parsed": len(added),
            "n_vintage_files": len(_stored_report_ids(cache)), "n_rows": len(vintages),
            "n_indicators": vintages["indicator_key"].nunique() if not vintages.empty else 0,
            "n_unmatched_titles": len(unmatched), "failures": failures}


def update_inei_latest(cache_dir: str | Path | None = None, *,
                       as_of=None) -> dict:
    """Idempotent monthly update: ingest every bulletin the lake does not already hold.

    Discovery walks the gob.pe collection (the current publication surface;
    the old m.inei library remains only for historical rebuilds). Writes each
    new bulletin as its own vintage file (existing files are never rewritten)
    plus a provenance row in ``gobpe_index.parquet`` (publication date,
    bulletin URL, PDF URL, retrieval timestamp, sha256, reference-period
    range), then rebuilds the derived outputs. ``as_of`` bounds discovery for
    historical replays: bulletins published after it are invisible.
    """
    cache = _resolve_cache(cache_dir)
    stored = _stored_report_ids(cache)
    if not stored and not _store_paths(cache)["vintages"].exists():
        raise INEIError("No vintage store yet; run build_inei_history() first, then update.")
    return ingest_gobpe_bulletins(cache_dir=cache, as_of=as_of)


def available_inei_indicators(cache_dir: str | Path | None = None) -> pd.DataFrame:
    """The consistency ranking of indicators found in the vintage store."""
    cache = _resolve_cache(cache_dir)
    vintages = _read_vintages(cache)
    if vintages.empty:
        raise INEIError("No vintage store yet; run build_inei_history() first.")
    return rank_indicator_consistency(vintages)


def get_inei_series(indicators: list[str] | None = None, kind: str = "clean",
                    as_of: str | pd.Timestamp | None = None,
                    cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Return a wide monthly frame of unified series.

    ``kind``:
      - ``"clean"`` (default): reconstructed continuous levels for the consistent indicators.
      - ``"level_chained"``: reconstructed continuous levels, all indicators.
      - ``"level"``: raw latest-vintage index level (carries base-year breaks).
      - ``"yoy"``: base-safe year-on-year growth.
    ``as_of`` reconstructs the vintage known at a past date (real-time backtesting); else
    the latest stored series.
    """
    cache = _resolve_cache(cache_dir)
    paths = _store_paths(cache)
    file_for = {"clean": paths["levels_clean"], "level_chained": paths["level_chained"],
                "level": paths["level"], "yoy": paths["yoy"]}
    if kind not in file_for:
        raise INEIError(f"Unknown kind '{kind}'. Use clean, level_chained, level or yoy.")
    if as_of is None:
        path = file_for[kind]
        if not path.exists():
            raise INEIError("No unified series yet; run build_inei_history() first.")
        wide = pd.read_parquet(path)
    else:
        vintages = _read_vintages(cache)
        vintages = vintages[vintages["pub_date"] <= pd.Timestamp(as_of)]
        unified = build_unified_series(vintages)
        table = {"clean": unified["level_chained"], "level_chained": unified["level_chained"],
                 "level": unified["level_latest"], "yoy": unified["yoy"]}[kind]
        if kind == "clean":
            table = table.reindex(columns=clean_indicators(unified["level_chained"]))
        wide = table
    if indicators is not None:
        wide = wide[[c for c in indicators if c in wide.columns]]
    wide.attrs["inei_metadata"] = {"kind": kind, "as_of": None if as_of is None else str(as_of)}
    return wide


# --------------------------------------------------------------------------- #
# Uniform provider interface (see sources/base.py)
# --------------------------------------------------------------------------- #
PROVIDER = "inei"
FREQUENCIES = ("M",)


def available(frequency: str | None = None, *, cache_dir=None) -> "pd.DataFrame":
    """Catalogue of INEI indicators in the standard schema."""
    from .base import as_catalog

    return as_catalog(available_inei_indicators(cache_dir=cache_dir),
                      provider=PROVIDER, frequency="M")


def fetch(series=None, *, frequency=None, start=None, end=None, refresh=False,
          kind: str = "clean", as_of=None, cache_dir=None) -> "pd.DataFrame":
    """Wide monthly frame of INEI series (standard shape)."""
    from .base import to_wide

    idx = list(series) if series is not None else None
    return to_wide(get_inei_series(indicators=idx, kind=kind, as_of=as_of, cache_dir=cache_dir),
                   start=start, end=end)
