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
