"""Generate a TradingView indicator with the desk's CURRENT levels baked in.

    python research/sheet/tv_export.py              # golden + watchlist, with insider
    python research/sheet/tv_export.py --all        # every covered name
    python research/sheet/tv_export.py --no-insider # skip the SEC pass (faster)

Writes tradingview/shaheen_desk.pine. Paste it into TradingView once; then on any
covered symbol it auto-draws that name's support, resistance, entry, stop, target
and buy-zone band, and flags an INSIDER WARNING when the desk saw heavy Form 4
selling - no typing per stock.

WHY BAKE IT IN
Pine Script cannot make an HTTP request, so an indicator cannot pull from our
server. The only way desk numbers reach a chart is to compile them into the
script as lookup arrays. That means the file is a SNAPSHOT: regenerate and
re-paste when you want fresh levels (run.py --daily does the regenerate for you).
For symbols not in the snapshot the indicator falls back to auto pivot S/R, so it
is still useful everywhere.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import signal_engine as se          # noqa: E402
import kuwait as kw                 # noqa: E402
import insider_watch as iw          # noqa: E402

OUT = os.path.join(HERE, "..", "..", "tradingview", "shaheen_desk.pine")
MAX_SYMS = 120


def _f(v) -> str:
    """A float literal for Pine, or na."""
    try:
        x = float(v)
        return "na" if (x != x) else f"{x:.4f}"
    except (TypeError, ValueError):
        return "na"


def _s(v) -> str:
    """A quoted, Pine-safe string literal."""
    t = str(v or "").replace("\\", " ").replace('"', "'").replace("\n", " ")
    return '"' + t[:80] + '"'


def collect(all_names: bool, do_insider: bool) -> pd.DataFrame:
    us = se.load_signals()
    us["market"] = "US"
    frames = [us]
    k = kw.load()
    if k is not None and not k.empty:
        k["market"] = "KW"
        frames.append(k)
    cols = sorted(set().union(*[set(f.columns) for f in frames]))
    df = pd.concat([f.reindex(columns=cols) for f in frames], ignore_index=True)

    if not all_names:
        keep = set(df.loc[df["golden"] == True, "symbol"])          # noqa: E712
        keep |= set(se.picked(us))
        df = df[df["symbol"].isin(keep)]
    df = df.sort_values("fund_100", ascending=False).head(MAX_SYMS).reset_index(drop=True)

    df["ins_warn"] = False
    df["ins_note"] = ""
    if do_insider:
        # only the US names, and only where a warning would matter - bounded work
        us_syms = df.loc[df["market"] == "US", "symbol"].tolist()
        for s in us_syms:
            try:
                r = iw.insider_summary(s, 30)
            except Exception:
                continue
            lab = r.get("label", "")
            if lab.startswith("WARNING") or "heavy selling" in lab:
                df.loc[df["symbol"] == s, ["ins_warn", "ins_note"]] = [True, lab]
            elif lab.startswith("BUY CLUSTER"):
                df.loc[df["symbol"] == s, ["ins_warn", "ins_note"]] = [True, lab]
    return df


def emit(df: pd.DataFrame) -> str:
    def arr(col, conv):
        return "array.from(" + ", ".join(conv(v) for v in df[col]) + ")"

    body = f'''//@version=5
// ============================================================
// SHAHEEN DESK LEVELS  -  generated {dt.datetime.now():%Y-%m-%d %H:%M}
// {len(df)} symbols baked in. Regenerate with research/sheet/tv_export.py
// and re-paste to refresh. Uncovered symbols fall back to auto pivot S/R.
// ============================================================
indicator("Shaheen Desk Levels", overlay = true, max_lines_count = 200,
     max_labels_count = 50)

var string[] SYMS = {arr("symbol", _s)}
var float[]  SUP  = {arr("support", _f)}
var float[]  RES  = {arr("resistance", _f)}
var float[]  ENT  = {arr("entry", _f)}
var float[]  STP  = {arr("stop", _f)}
var float[]  TGT  = {arr("target", _f)}
var float[]  FUND = {arr("fund_100", _f)}
var float[]  TECH = {arr("tech_100", _f)}
var bool[]   INSW = {arr("ins_warn", lambda v: "true" if bool(v) else "false")}
var string[] INSN = {arr("ins_note", _s)}

showAuto = input.bool(true, "Auto pivot S/R when symbol not covered")
pivLen   = input.int(15, "Auto swing strength", minval = 3)

t   = syminfo.ticker
idx = array.indexof(SYMS, t)
covered = idx >= 0

sup = covered ? array.get(SUP, idx) : na
res = covered ? array.get(RES, idx) : na
ent = covered ? array.get(ENT, idx) : na
stp = covered ? array.get(STP, idx) : na
tgt = covered ? array.get(TGT, idx) : na

// desk lines (flat, extend right)
plot(sup, "Support",    color.new(color.green, 0), 2, plot.style_linebr)
plot(res, "Resistance", color.new(color.red, 0),   2, plot.style_linebr)
plot(ent, "Entry",      color.new(color.blue, 0),  2, plot.style_linebr)
plot(stp, "Stop",       color.new(color.maroon, 0),2, plot.style_linebr)
plot(tgt, "Target",     color.new(color.teal, 0),  2, plot.style_linebr)

// shaded golden buy zone between entry and stop
pE = plot(ent, display = display.none)
pS = plot(stp, display = display.none)
fill(pE, pS, color = color.new(color.green, 85), title = "Buy zone")

// ratings + insider table
var table tb = table.new(position.top_right, 2, 4, border_width = 1)
if barstate.islast and covered
    table.cell(tb, 0, 0, "Fundamental", text_color = color.gray, text_size = size.small)
    table.cell(tb, 1, 0, str.tostring(array.get(FUND, idx), "#") + "/100", text_size = size.small)
    table.cell(tb, 0, 1, "Technical", text_color = color.gray, text_size = size.small)
    table.cell(tb, 1, 1, str.tostring(array.get(TECH, idx), "#") + "/100", text_size = size.small)
    table.cell(tb, 0, 2, "Reward:Risk", text_color = color.gray, text_size = size.small)
    rr = (not na(tgt) and not na(ent) and not na(stp) and (ent - stp) != 0) ? (tgt - ent) / (ent - stp) : na
    table.cell(tb, 1, 2, na(rr) ? "-" : str.tostring(rr, "#.##") + ":1", text_size = size.small)
    table.cell(tb, 0, 3, "Insider", text_color = color.gray, text_size = size.small)
    table.cell(tb, 1, 3, array.get(INSW, idx) ? "WARNING" : "ok",
         text_color = array.get(INSW, idx) ? color.orange : color.gray, text_size = size.small)

// insider warning label
if barstate.islast and covered and array.get(INSW, idx)
    label.new(bar_index, high, "INSIDER: " + array.get(INSN, idx),
         style = label.style_label_down, color = color.new(color.orange, 0),
         textcolor = color.white, size = size.small)

// auto pivot S/R fallback for uncovered symbols
ph = ta.pivothigh(high, pivLen, pivLen)
pl = ta.pivotlow(low, pivLen, pivLen)
if showAuto and not covered and not na(ph)
    line.new(bar_index - pivLen, ph, bar_index, ph, extend = extend.right,
         color = color.new(color.red, 40), style = line.style_dotted)
if showAuto and not covered and not na(pl)
    line.new(bar_index - pivLen, pl, bar_index, pl, extend = extend.right,
         color = color.new(color.green, 40), style = line.style_dotted)

// alert when price trades into the desk buy zone
inZone = covered and not na(ent) and not na(stp) and low <= ent and high >= stp
alertcondition(inZone, "Entered buy zone", "Price entered the Shaheen buy zone")
plotshape(inZone and not inZone[1], title = "Entered zone",
     style = shape.triangleup, location = location.belowbar,
     color = color.new(color.green, 0), size = size.tiny)
'''
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-insider", action="store_true")
    a = ap.parse_args()
    df = collect(all_names=a.all, do_insider=not a.no_insider)
    txt = emit(df)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(txt)
    warned = int(df["ins_warn"].sum())
    print("wrote %s" % os.path.normpath(OUT))
    print("  %d symbols baked, %d insider warnings flagged" % (len(df), warned))


if __name__ == "__main__":
    main()
