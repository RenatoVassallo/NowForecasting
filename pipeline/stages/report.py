"""Stage 5: the weekly report - a Beamer PDF plus a markdown log.

Fills ``pipeline/report/template.tex`` with this run's numbers, copies the
vector figures next to it, and compiles with latexmk. The template is the
design surface: edit it once, and every weekly run inherits the change.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.stages import fanchart as F

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "pipeline" / "report" / "template.tex"


def _esc(s: str) -> str:
    return str(s).replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")


def _tokens(ctx) -> dict:
    from sources import imf

    peru = ctx["peru"]
    node = peru.iloc[0]
    y = ctx["peru_hist"]
    d = peru.set_index("period")
    yr = node["period"].year

    ann1 = list(y[y.index.year == yr].values) + \
           [v for p, v in d["mode"].items() if p.year == yr and p not in y.index]
    ann2 = [v for p, v in d["mode"].items() if p.year == yr + 1]
    try:
        weo, _ = imf.path("PER", pd.Timestamp.now())
        weo1, weo2 = f"{float(weo.get(yr, np.nan)):.1f}", f"{float(weo.get(yr + 1, np.nan)):.1f}"
    except Exception:
        weo1 = weo2 = "-"

    state = ("the first node is well informed" if node.get("information_index", 0) > 0.75
             else "the run is early in the release cycle: the first node is no better "
                  "informed than the second")
    info = (f"As of {node.get('as_of', '')}: {abs(int(node.get('days_to_publication', 0)))} days to the "
            f"{node['quarter']} GDP release, information index "
            f"{node.get('information_index', float('nan')):.2f} - {state}. "
            "Conditioning: US from the SPF and the IMF WEO live round; China from the "
            "published China profile; terms of trade from the monthly commodity BVAR; "
            "business expectations held at their last value.")

    fanrows = "\n".join(
        f"        {r['quarter']} & {r['mode']:.1f} & "
        f"[{r['lo60']:.1f}, {r['hi60']:.1f}] & [{r['lo90']:.1f}, {r['hi90']:.1f}] \\\\"
        for _, r in peru.iterrows())

    # conditioning table
    b1, b0 = ctx["bridge"]
    grid = list(ctx["usa"]["period"])
    heads = [str(grid[0] - 4 + i) for i in range(4)] + [str(p) for p in grid]
    condhead = " & " + " & ".join(_esc(h) for h in heads)

    def hist_vals(series):
        base = grid[0]
        return [f"{float(series.get(base - 4 + i, np.nan)):.1f}"
                if (base - 4 + i) in series.index else "-" for i in range(4)]

    from targets import usa as usmod
    um, _, _ = usmod.load_panel()
    us_h = um["us_gdp_yoy_m"].dropna(); us_h.index = pd.PeriodIndex(us_h.index, freq="Q")
    us_h = us_h[~us_h.index.duplicated(keep="last")]
    mm_p, _, _ = ctx["peru_spec"].load_panel()
    ex = mm_p["exp_eco3m"].dropna()
    exq = ex.groupby(pd.PeriodIndex(ex.index, freq="Q")).mean()
    ip_fc = b0 + b1 * ctx["china"]["centre"]
    ip_obs = ctx["ip_hist"]
    rows = {
        "US GDP": hist_vals(us_h) + [f"{v:.1f}" for v in ctx["usa"]["centre"]],
        "China GDP": hist_vals(ctx["china_hist"]) + [f"{v:.1f}" for v in ctx["china"]["centre"]],
        "China IP": hist_vals(ip_obs) + [f"{float(ip_obs[p]):.1f}" if p in ip_obs.index
                                         else f"{v:.1f}" for p, v in zip(grid, ip_fc)],
        "Terms of trade": hist_vals(ctx["tot_hist"]) + [f"{v:.1f}" for v in ctx["tot"]["centre"]],
        "Expectations": hist_vals(exq) + [f"{float(ex.iloc[-1]):.1f}"] * len(grid),
        "Peru GDP": hist_vals(y) + [f"{float(d['mode'].get(p, np.nan)):.1f}"
                                    if p in d.index else "-" for p in grid],
    }
    condrows = "\n".join(f"    {_esc(k)} & " + " & ".join(v) + r" \\" for k, v in rows.items())

    # data snapshot
    from sources import commodities as csrc
    px = csrc.load(); px.index = pd.DatetimeIndex(px.index)

    def last4(s, fmt="{:.1f}", freq="M"):
        s = s.dropna().iloc[-4:]
        labels = [str(pd.Period(ix, freq=freq)) if not isinstance(ix, pd.Period) else str(ix)
                  for ix in s.index]
        return labels, [fmt.format(v) for v in s.values]

    m_series = {"Peru monthly GDP": mm_p["g_pbim"], "Expectations 3m": mm_p["exp_eco3m"],
                "Copper (\\$/t)": px["p_copper"], "Gold (\\$/oz)": px["p_gold"],
                "WTI (\\$/bbl)": px["p_wti"], "Terms of trade": px["pe_tot"]}
    fmts = {"Copper (\\$/t)": "{:,.0f}", "Gold (\\$/oz)": "{:,.0f}", "WTI (\\$/bbl)": "{:,.0f}"}
    mlabels = last4(mm_p["g_pbim"])[0]
    mhead = " & " + " & ".join(_esc(l) for l in mlabels)
    mrows = []
    for k, s in m_series.items():
        _, vals = last4(s, fmts.get(k, "{:.1f}"))
        mrows.append(f"        {k} & " + " & ".join(vals) + r" \\")
    q_series = {"Peru GDP": y, "China GDP": ctx["china_hist"],
                "US GDP": us_h, "Terms of trade": ctx["tot_hist"]}
    qlabels = last4(y, freq="Q")[0]
    qhead = " & " + " & ".join(_esc(l) for l in qlabels)
    qrows = []
    for k, s in q_series.items():
        _, vals = last4(s, freq="Q")
        qrows.append(f"        {_esc(k)} & " + " & ".join(vals) + r" \\")

    return {
        "<<DATE>>": str(pd.Timestamp.now().date()),
        "<<QUARTER>>": str(node["quarter"]),
        "<<NOWCAST>>": f"{node['mode']:.1f}",
        "<<NOWCAST90>>": f"[{node['lo90']:.1f}, {node['hi90']:.1f}]",
        "<<ANNUAL1>>": f"{np.mean(ann1[:4]):.1f}", "<<ANNUAL1Y>>": str(yr),
        "<<ANNUAL2>>": f"{np.mean(ann2):.1f}", "<<ANNUAL2Y>>": str(yr + 1),
        "<<WEO1>>": weo1, "<<WEO2>>": weo2,
        "<<TOTNOW>>": f"{ctx['tot'].iloc[0]['mode']:.0f}",
        "<<TOTSRC>>": _esc(ctx["tot"].iloc[0]["source"]),
        "<<INFOSTATE>>": _esc(info),
        "<<FANROWS>>": fanrows,
        "<<CONDHEAD>>": condhead, "<<CONDROWS>>": condrows,
        "<<MHEAD>>": mhead, "<<MROWS>>": "\n".join(mrows),
        "<<QHEAD>>": qhead, "<<QROWS>>": "\n".join(qrows),
    }


def run(store, params, whatsnew: str, lines: list[str], timings: dict) -> None:
    ctx = getattr(store, "fig_ctx", None) or F.load_context()
    root = Path(store.root)
    figdir = root / "figures"
    figdir.mkdir(exist_ok=True)
    for f in (REPO / "products" / "figures").glob("*.pdf"):
        shutil.copy2(f, figdir / f.name)

    tex = TEMPLATE.read_text()
    for k, v in _tokens(ctx).items():
        tex = tex.replace(k, v)
    (root / "report.tex").write_text(tex)

    compiled = False
    for cmd in (["latexmk", "-pdf", "-silent", "-halt-on-error", "report.tex"],
                ["tectonic", "report.tex"]):
        if shutil.which(cmd[0]) is None:
            continue
        try:
            subprocess.run(cmd, cwd=root, check=True, capture_output=True, timeout=300)
            if cmd[0] == "latexmk":
                subprocess.run(["latexmk", "-c"], cwd=root, capture_output=True)
            shutil.copy2(root / "report.pdf", REPO / "products" / "report.pdf")
            print(f"  [report] {root / 'report.pdf'} (via {cmd[0]})")
            compiled = True
            break
        except Exception as exc:
            out = (getattr(exc, "stdout", b"") or b"").decode(errors="ignore")
            print(f"  [report] {cmd[0]} failed; tail:\n{out[-600:]}")
    if not compiled:
        print("  [report] no LaTeX engine found - report.tex is ready to compile.\n"
              "           install one with: apt-get install -y latexmk "
              "texlive-latex-recommended texlive-fonts-recommended")

    run_id = getattr(store, "run_id", root.name)
    md = [f"# NowForecasting run {run_id}", ""] + lines + \
         ["", "## What's new in the data", "", whatsnew or "-", "", "## Timings", ""] + \
         [f"- {k}: {v:.0f}s" for k, v in timings.items()] + ["", "Full report: report.pdf"]
    (root / "report.md").write_text("\n".join(md))
