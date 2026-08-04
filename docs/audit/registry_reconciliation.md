# Registry reconciliation

Publication-lag conflicts between `sources/catalog.csv` and the
target modules' `DELAYS` dictionaries, and how each was resolved.
Rule: where both declare a value and disagree, the LARGER lag wins,
because claiming a release earlier than it happens is the direction
that leaks information. Hand overrides live in
`pipeline/config/build_registry.py:LAG_OVERRIDES`.

| series | catalog | model code | chosen | rule |
|---|---|---|---|---|
| manufacturing_pmi | None | 0 | 0 | catalog lag missing; model code value adopted |
| non_manufacturing_pmi | None | 0 | 0 | catalog lag missing; model code value adopted |
| real_estate_investment_cum_yoy | None | 15 | 15 | catalog lag missing; model code value adopted |

3 rows reconciled.

## Hand-resolved cross-code conflict

The official terms-of-trade index appears under three internal codes
(`tdi` in the catalog, `g_tdi` in the Peru panel, `g_pe_tot` in the commodity
block) with lags of 40, 35 and 35. Resolved to 40 days everywhere (catalog
value; the larger lag is the safe direction): `targets/commodities.py`
(DELAYS and the TARGETS table), `notebooks/peru/forecast/common.py`, and the
curated registry entries. `pipeline/blocks/peru.py` already used 40 for its
released-quarter gate.

## Follow-up audit additions (2026-08-04)

The declared-inputs contract test (`tests/test_input_contract.py`) extracts
every series production consumes from the model specifications themselves and
requires one monitored registry entry each. It surfaced three gaps, all fixed:

- `us_fedfunds` (NEW, required, monitored on `input/us/us_monthly.parquet`):
  fixed condition of the China Cond-BVAR system and the Peru S1 system. ALIAS
  reconciliation: the Peru spec3 panel ingests the same upstream FRED series
  as `fed` (1 day lag); both entries now cross-reference each other. Two
  monitored columns, one source; neither is optional.
- `us_cpi_yoy` (NEW, required, monitored, 12 day lag from targets/usa.py):
  fixed condition of the China Cond-BVAR and a steady-state anchor of
  Cond-BVAR+SS.
- `g_pbim` (NEW, required, monitored, 51 day lag): the RAW X13-adjusted Peru
  monthly proxy column the refresh due-gate watches; companion entry of
  `g_pbim_yoy` (same upstream series, different processed column).

Registry: 116 series, 32 required for publication. No variable was made
optional to satisfy the preflight.
