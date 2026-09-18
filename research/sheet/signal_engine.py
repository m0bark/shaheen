"""Deterministic signal layer for the desk: ratings /100, risk:reward, dollar
risk/gain, and the GOLDEN buy-zone flag. No AI, no network - pure functions of
the CSV cache, so the AI layer and the Telegram alerts sit on numbers that were
computed one way, once.

    from signal_engine import load_signals
    df = load_signals()          # every symbol, with the columns below added

Columns added
  fund_100     fundamental rating 0-100  (= RATE, the value/quality/safety/growth
               composite already in the cache; surfaced under a clearer name)
  tech_100     technical rating 0-100    (momentum + trend + RSI position +
               proximity to support, blended and clipped)
  entry        planned entry  = perfect_buy, else zone_entry_price, else price
  stop         protective stop = just under support, floored at a max % loss
  target       first target   = resistance, else a measured move
  rr           reward-to-risk  = (target-entry) / (entry-stop)
  risk_dollars account_size * risk_per_trade_pct/100   (what you put at risk)
  shares       floor(risk_dollars / (entry-stop))       (position sized to the stop)
  position_dollars  shares * entry
  gain_dollars shares * (target-entry)                  (reward if target hits)
  golden       True when a name is in the golden buy zone (see is_golden)

WHY fund_100 == RATE, not a new formula
The repo already validated that quality-alone beats the market and that adding
screener filters SUBTRACTS. Inventing a second fundamental score here would just
be an unvalidated re-weighting. RATE is the number the rest of the shop trusts;
this module renames it, it does not second-guess it.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
CONFIG = os.path.join(HERE, "config.json")


def cfg() -> dict:
    return json.load(open(CONFIG, encoding="utf-8")) if os.path.exists(CONFIG) else {}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _tech_100(d: pd.DataFrame) -> pd.Series:
    """Blend the technical picture into 0-100. Each leg is already 0-100 or is
    mapped into it; weights favour trend and momentum, which the repo's own
    momentum work found to be the only technical factors that carried anything.
    """
    mom = _num(d.get("MOMENTUM_SCORE")).fillna(50)           # 0-100 already
    trend = np.where(d.get("vs_200sma").astype(str) == "ABOVE", 70.0, 30.0)
    trend = pd.Series(trend, index=d.index)
    # add the distance above/below the 200d, capped, so a name well above trend
    # scores higher than one barely above it
    trend = (trend + _num(d.get("pct_vs_200sma")).fillna(0).clip(-20, 20)).clip(0, 100)
    # RSI: reward the constructive 40-65 band, penalise >75 (extended) and the
    # deep <30 (falling knife) - a technical BUY score, not a mean-reversion bet
    rsi = _num(d.get("rsi14")).fillna(50)
    rsi_score = (100 - (rsi - 55).abs() * 2.2).clip(0, 100)
    # proximity to support: closer to support = better entry technically
    near_sup = (100 - _num(d.get("pct_to_support")).abs().fillna(25) * 4).clip(0, 100)
    out = 0.40 * mom + 0.30 * trend + 0.15 * rsi_score + 0.15 * near_sup
    return out.round(0)


def is_golden(d: pd.DataFrame, c: dict | None = None) -> pd.Series:
    """Golden buy zone: price has fallen into the zone AND the business rates
    well enough to buy the dip rather than catch a broken one."""
    c = c or cfg()
    rate_min = float(c.get("golden_rate_min", 55))
    within = float(c.get("golden_within_pct", 3.0))
    require_halal = bool(c.get("golden_require_halal", False))
    z = d["zone_status"].astype(str)
    disc = _num(d.get("discount_to_buy_pct"))
    g = (z.isin(["AT ZONE", "IN ZONE"])
         & (_num(d.get("fund_100")) >= rate_min)
         & (disc.notna()) & (disc >= -within)
         & (_num(d.get("perfect_buy")).notna()))
    if require_halal and "halal_auto" in d.columns:
        g &= d["halal_auto"].astype(str).eq("pass")
    return g


def add_rr(d: pd.DataFrame, c: dict | None = None) -> pd.DataFrame:
    c = c or cfg()
    acct = float(c.get("account_size", 25000))
    risk_pct = float(c.get("risk_per_trade_pct", 1.5))

    price = _num(d.get("price"))
    entry = _num(d.get("perfect_buy"))
    entry = entry.fillna(_num(d.get("zone_entry_price"))).fillna(price)
    sup = _num(d.get("support"))
    res = _num(d.get("resistance"))
    atr = _num(d.get("atr14"))

    # stop: just under support when support sits below entry, else a measured
    # 2-ATR stop, and never a loss deeper than 15% (the repo's tested trailing
    # band; tighter stops near lows get wicked out 52% of the time)
    stop_support = np.where((sup.notna()) & (sup < entry), sup * 0.98, np.nan)
    stop_atr = entry - 2.0 * atr.fillna(entry * 0.04)
    stop = pd.Series(stop_support, index=d.index).fillna(stop_atr)
    floor = entry * 0.85
    stop = pd.Series(np.maximum(stop, floor), index=d.index)

    # A support sitting a hair under spot produces a 0.5%-risk stop, which then
    # inflates R:R past 15 and sizes an absurd share count. Floor the per-share
    # risk at the larger of ~0.8 ATR or 3% of entry, and pull the stop down to
    # match, so the numbers stay tradeable.
    min_risk = pd.concat([0.8 * atr.fillna(entry * 0.04), 0.03 * entry],
                         axis=1).max(axis=1)
    risk_ps = (entry - stop)
    risk_ps = pd.concat([risk_ps, min_risk], axis=1).max(axis=1).clip(lower=0.01)
    stop = (entry - risk_ps).round(2)

    # target: first resistance above entry, else a measured move of ~2x the risk
    tgt_res = np.where((res.notna()) & (res > entry), res, np.nan)
    tgt_move = entry + 2.0 * risk_ps
    target = pd.Series(tgt_res, index=d.index).fillna(tgt_move)

    reward_ps = (target - entry).clip(lower=0)
    rr = (reward_ps / risk_ps).round(2)

    risk_dollars = acct * risk_pct / 100.0
    # size to the stop, but never buy more than the whole account can hold
    shares = np.minimum(np.floor(risk_dollars / risk_ps),
                        np.floor(acct / entry)).clip(lower=0)
    d["entry"] = entry.round(2)
    d["stop"] = stop.round(2)
    d["target"] = target.round(2)
    d["rr"] = rr
    d["risk_dollars"] = round(risk_dollars, 2)
    d["shares"] = shares.astype("Int64")
    d["position_dollars"] = (shares * entry).round(0)
    d["gain_dollars"] = (shares * reward_ps).round(0)
    return d


def load_signals() -> pd.DataFrame:
    s = pd.read_csv(os.path.join(CACHE, "summary.csv"))
    c = cfg()
    s["fund_100"] = _num(s.get("RATE")).round(0)
    s["tech_100"] = _tech_100(s)
    s = add_rr(s, c)
    s["golden"] = is_golden(s, c)
    return s


def picked(df: pd.DataFrame | None = None) -> list[str]:
    """The names we actually watch: everything currently golden, plus the
    user's own watchlist. This is the set the AI, news and insider layers act
    on - never the whole index."""
    df = load_signals() if df is None else df
    names = set(df.loc[df["golden"], "symbol"].tolist())
    wl = os.path.join(HERE, "watchlist.csv")
    if os.path.exists(wl):
        w = pd.read_csv(wl)
        if "symbol" in w.columns:
            names |= {x for x in w["symbol"].astype(str) if x and x != "ANY"}
    return sorted(names)


if __name__ == "__main__":
    d = load_signals()
    g = d[d["golden"]].sort_values("fund_100", ascending=False)
    print("universe: %d | golden: %d" % (len(d), len(g)))
    cols = ["symbol", "price", "fund_100", "tech_100", "entry", "stop", "target",
            "rr", "shares", "risk_dollars", "gain_dollars", "zone_status"]
    print(g[cols].head(20).to_string(index=False))
    print("\npicked set (golden + watchlist):", ", ".join(picked(d)) or "(none)")
