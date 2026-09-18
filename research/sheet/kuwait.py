"""Boursa Kuwait names on the same desk - ratings /100, buy zone, risk:reward.

    python research/sheet/kuwait.py            # fetch + build cache/kuwait.csv
    python research/sheet/kuwait.py --show     # print the table from cache

Kuwait sits apart from the US pipeline for honest reasons, stated so they are
not forgotten:
  * No SEC Form 4 in Kuwait, so there is NO insider layer for these names.
  * yfinance news for .KW tickers is thin to empty.
  * Fundamerals here are the few fields Boursa/analyst sheets publish (PE, PB,
    dividend yield, a fair-value note), not the SEC XBRL panel the US names get.
So a Kuwait rating is price-and-valuation based; it is not the same instrument as
a US RATE and is labelled that way.

Prices come from yfinance with the .KW suffix (verified: NBK.KW, KFH.KW, ...).
Technical levels (200-day, RSI, ATR, support/resistance) are computed here from
one year of daily closes, and the buy-zone / risk-reward reuse signal_engine so
the numbers mean the same thing across both markets.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import signal_engine as se          # noqa: E402

CACHE = os.path.join(HERE, "cache")
UNIV = os.path.join(HERE, "kuwait_universe.json")


def _rsi(close: pd.Series, n: int = 14) -> float:
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def _atr(h: pd.DataFrame, n: int = 14) -> float:
    hi, lo, cl = h["High"], h["Low"], h["Close"]
    tr = pd.concat([(hi - lo), (hi - cl.shift()).abs(), (lo - cl.shift()).abs()],
                   axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1])


def _pct(v):
    try:
        return float(str(v).replace("%", "").strip())
    except (TypeError, ValueError):
        return np.nan


def _fund_100(df: pd.DataFrame) -> pd.Series:
    """Valuation-based rating: cheaper P/E and P/B and fatter yield rank higher,
    scored as within-Kuwait percentiles. Not a US RATE - a relative-value rank."""
    pe = pd.to_numeric(df["pe"], errors="coerce")
    pb = pd.to_numeric(df["pb"], errors="coerce")
    dy = df["dy"].map(_pct)
    r_pe = (1 - pe.rank(pct=True)) * 100          # low PE -> high score
    r_pb = (1 - pb.rank(pct=True)) * 100
    r_dy = dy.rank(pct=True) * 100                 # high yield -> high score
    out = pd.concat([r_pe, r_pb, r_dy], axis=1).mean(axis=1)
    return out.fillna(50).round(0)


def build() -> pd.DataFrame:
    u = json.load(open(UNIV, encoding="utf-8"))
    rows = []
    for x in u:
        sym, yfs = x["t"], x["yf"]
        try:
            h = yf.Ticker(yfs).history(period="1y", auto_adjust=True)
        except Exception:
            h = pd.DataFrame()
        if len(h) < 60:
            print(f"  {sym}: thin/no history ({len(h)})", flush=True)
            continue
        cl = h["Close"]
        price = float(cl.iloc[-1])
        sma200 = float(cl.rolling(200).mean().iloc[-1]) if len(cl) >= 200 else float(cl.mean())
        lo60 = float(cl.tail(60).min())
        hi60 = float(cl.tail(120).max())
        atr = _atr(h)
        rows.append({
            "symbol": sym, "yf": yfs, "shortName": x.get("en"), "sector": x.get("sec"),
            "price": round(price, 3), "sma200": round(sma200, 3),
            "vs_200sma": "ABOVE" if price >= sma200 else "BELOW",
            "pct_vs_200sma": round((price / sma200 - 1) * 100, 1),
            "rsi14": round(_rsi(cl), 1), "atr14": round(atr, 4),
            "atr_pct": round(atr / price * 100, 2),
            "support": round(lo60, 3), "resistance": round(hi60, 3),
            "pct_to_support": round((lo60 / price - 1) * 100, 1),
            "pct_to_resistance": round((hi60 / price - 1) * 100, 1),
            "pe": x.get("pe"), "pb": x.get("pb"), "dy": x.get("dy"),
            "fv": x.get("fv"), "halal_auto": x.get("halal"),
            "MOMENTUM_SCORE": np.nan,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # momentum proxy = 6-month price change percentile (no US momentum panel here)
    df["MOMENTUM_SCORE"] = ((df["price"] / df["sma200"]).rank(pct=True) * 100).round(0)
    df["fund_100"] = _fund_100(df)
    df["RATE"] = df["fund_100"]
    df["tech_100"] = se._tech_100(df)

    # buy zone: anchors below price = support, 200-day, price-1.5*ATR; perfect_buy
    # is their median, floored at -15% (Kuwait moves less than US small caps)
    anch = pd.concat([df["support"], df["sma200"],
                      df["price"] - 1.5 * df["atr14"]], axis=1)
    below = anch.where(anch.lt(df["price"], axis=0))
    df["n_anchors"] = below.notna().sum(axis=1)
    df["perfect_buy"] = below.median(axis=1).clip(lower=df["price"] * 0.85).round(3)
    df["buy_zone_low"] = below.min(axis=1).round(3)
    df["buy_zone_high"] = below.median(axis=1).round(3)
    df["discount_to_buy_pct"] = ((df["perfect_buy"] / df["price"] - 1) * 100).round(2)
    df["zone_entry_price"] = df["perfect_buy"]
    df["zone_status"] = np.select(
        [df["n_anchors"] >= 3, df["n_anchors"] == 2, df["n_anchors"] == 1],
        ["AT ZONE", "IN ZONE", "NEAR ZONE"], default="ABOVE ZONE")

    df = se.add_rr(df)
    df["golden"] = se.is_golden(df)
    os.makedirs(CACHE, exist_ok=True)
    df.to_csv(os.path.join(CACHE, "kuwait.csv"), index=False)
    return df


def load() -> pd.DataFrame:
    p = os.path.join(CACHE, "kuwait.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    df = load() if a.show else build()
    if df.empty:
        print("no Kuwait data")
    else:
        cols = ["symbol", "price", "fund_100", "tech_100", "zone_status",
                "perfect_buy", "rr", "shares", "gain_dollars", "halal_auto"]
        print("Kuwait: %d names | golden %d" % (len(df), int(df["golden"].sum())))
        print(df[cols].to_string(index=False))
