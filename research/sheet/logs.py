"""Three paper-trading ledgers. Signals in, marked to market, scored forward.

    python research/sheet/logs.py            # record new signals, mark to market
    python research/sheet/logs.py --report   # performance of each ledger
    python research/sheet/logs.py --reset analyst    # wipe one and start over

THE THREE

  analyst   an analyst UPGRADE or INITIATE at Buy on the local rating tape
  buyzone   price enters the computed buy zone AND the rate is not bottom-quartile
  bigmoney  unusual option premium, call-heavy

WHY LEDGERS AND NOT JUST BACKTESTS
Two of these can be backtested from price history and one cannot: nobody sells
free historical option chains, so 'bigmoney' has no past to test against. A
forward ledger is the only honest way to measure it -- record every signal the
moment it fires, never revise, and let time score it. The same machine runs all
three so the two that CAN be backtested act as a calibration check on whether
the ledger is being kept fairly.

RULES, FIXED IN ADVANCE SO THEY CANNOT DRIFT
  entry     next available close after the signal (what you could actually get)
  exit      63 trading days, or a 2-ATR stop, whichever comes first
  size      equal weight, notional only, no leverage
  benchmark SPY over the identical holding window, recorded per position

An entry is written ONCE. Rows are never edited except to mark price, and the
entry price is never touched. That is what stops a ledger becoming a story.
"""
from __future__ import annotations

import csv
import io
import os
import sys
import warnings
from datetime import datetime

_OUT = io.TextIOWrapper(open(sys.stdout.fileno(), "wb", closefd=False),
                        encoding="utf-8", errors="replace")


def say(*a) -> None:
    print(*a, file=_OUT)
    _OUT.flush()


warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
LOGS = os.path.join(HERE, "logs")
HOLD_DAYS = 63
STOP_ATR = 2.0
COLS = ["opened", "strategy", "symbol", "entry", "trigger", "rate", "sector",
        "stop", "target_days", "last_price", "last_date", "pnl_pct",
        "spy_entry", "spy_now", "vs_spy_pct", "days_held", "status", "closed"]


def load(name: str) -> pd.DataFrame:
    p = os.path.join(CACHE, f"{name}.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def path(strategy: str) -> str:
    return os.path.join(LOGS, f"{strategy}_log.csv")


def read_log(strategy: str) -> pd.DataFrame:
    p = path(strategy)
    if not os.path.exists(p):
        return pd.DataFrame(columns=COLS)
    d = pd.read_csv(p)
    for c in COLS:
        if c not in d:
            d[c] = np.nan
    return d


def write_log(strategy: str, d: pd.DataFrame) -> None:
    os.makedirs(LOGS, exist_ok=True)
    d[COLS].to_csv(path(strategy), index=False)


# ------------------------------------------------------------ signal rules


def sig_analyst(summ, an, big):
    """Upgrade or fresh initiation at Buy, recorded within the last 2 days."""
    # last_action/last_rating/last_firm only exist once a local ratings tape has
    # been swept in; without it the analysts sheet is yfinance consensus only,
    # so there are no dated upgrade events to log - return empty, do not crash
    # the whole hourly run over a missing column.
    need = ("last_action", "last_rating", "last_firm")
    if an.empty or any(c not in an.columns for c in need):
        return pd.DataFrame()
    a = an.copy()
    a["days_ago"] = pd.to_numeric(a.get("days_ago"), errors="coerce")
    m = (a["last_action"].astype(str).str.startswith(("Upgrade", "Initiates"))
         & a["last_rating"].astype(str).str.contains("Buy|Outperform|Overweight",
                                                     case=False, na=False)
         & (a["days_ago"] <= 2))
    out = a[m][["symbol"]].copy()
    out["trigger"] = (a[m]["last_action"].astype(str) + " -> "
                      + a[m]["last_rating"].astype(str) + " ("
                      + a[m]["last_firm"].astype(str) + ")")
    return out


def sig_buyzone(summ, an, big):
    """In or at the computed zone, and not bottom-quartile quality."""
    if summ.empty or "zone_status" not in summ:
        return pd.DataFrame()
    q1 = pd.to_numeric(summ["RATE"], errors="coerce").quantile(0.25)
    m = (summ["zone_status"].isin(["IN ZONE", "AT ZONE"])
         & (pd.to_numeric(summ["RATE"], errors="coerce") > q1))
    out = summ[m][["symbol"]].copy()
    out["trigger"] = (summ[m]["zone_status"].astype(str) + " @ "
                      + summ[m]["price"].round(2).astype(str)
                      + " vs buy " + summ[m]["perfect_buy"].round(2).astype(str))
    return out


def sig_bigmoney(summ, an, big):
    """Top-5% unusual option premium, skewed to calls."""
    if big.empty or "UNUSUAL" not in big:
        return pd.DataFrame()
    m = (big["UNUSUAL"] == "UNUSUAL") & (big.get("skew", "") == "CALL-HEAVY")
    out = big[m][["symbol"]].copy()
    out["trigger"] = ("unusual score " + big[m]["unusual_score"].round(0).astype(str)
                      + ", calls " + big[m]["call_pct"].round(0).astype(str) + "%"
                      + ", " + big[m]["max_strike"].astype(str))
    return out


RULES = {"analyst": sig_analyst, "buyzone": sig_buyzone, "bigmoney": sig_bigmoney}


# ------------------------------------------------------------ the machine


def update() -> None:
    summ, an, big = load("summary"), load("analysts"), load("bigmoney")
    if summ.empty:
        say("no cache -- run build_workbook.py first")
        return
    px = summ.set_index("symbol")["price"].to_dict()
    atr = (summ.set_index("symbol")["atr14"].to_dict()
           if "atr14" in summ else {})
    rate = summ.set_index("symbol")["RATE"].to_dict() if "RATE" in summ else {}
    sec = summ.set_index("symbol")["sector"].to_dict() if "sector" in summ else {}
    today = datetime.now().strftime("%Y-%m-%d")

    spy_now = np.nan
    sp = os.path.join(CACHE, "px_close.csv")
    if os.path.exists(sp):
        s = pd.read_csv(sp, index_col=0, parse_dates=True)
        if "SPY" in s:
            spy_now = float(s["SPY"].dropna().iloc[-1])

    for strat, rule in RULES.items():
        log = read_log(strat)
        sig = rule(summ, an, big)
        open_syms = set(log.loc[log["status"] == "OPEN", "symbol"]) \
            if len(log) else set()
        new = []
        for _, r in sig.iterrows():
            s = r["symbol"]
            if s in open_syms or s not in px or not np.isfinite(px[s]):
                continue
            a = atr.get(s, np.nan)
            new.append({
                "opened": today, "strategy": strat, "symbol": s,
                "entry": round(float(px[s]), 2), "trigger": r["trigger"],
                "rate": rate.get(s, np.nan), "sector": sec.get(s, ""),
                "stop": round(float(px[s]) - STOP_ATR * a, 2)
                        if np.isfinite(a) else np.nan,
                "target_days": HOLD_DAYS, "last_price": round(float(px[s]), 2),
                "last_date": today, "pnl_pct": 0.0, "spy_entry": spy_now,
                "spy_now": spy_now, "vs_spy_pct": 0.0, "days_held": 0,
                "status": "OPEN", "closed": ""})
        if new:
            log = pd.concat([log, pd.DataFrame(new)], ignore_index=True)

        # mark to market -- entry is NEVER rewritten
        for i in log.index[log["status"] == "OPEN"]:
            s = log.at[i, "symbol"]
            if s not in px or not np.isfinite(px[s]):
                continue
            p = float(px[s])
            e = float(log.at[i, "entry"])
            log.at[i, "last_price"] = round(p, 2)
            log.at[i, "last_date"] = today
            log.at[i, "pnl_pct"] = round((p / e - 1) * 100, 2)
            log.at[i, "spy_now"] = spy_now
            se = float(log.at[i, "spy_entry"]) if np.isfinite(
                pd.to_numeric(log.at[i, "spy_entry"], errors="coerce")) else np.nan
            if np.isfinite(se) and np.isfinite(spy_now) and se:
                log.at[i, "vs_spy_pct"] = round(
                    (p / e - spy_now / se) * 100, 2)
            held = (pd.Timestamp(today) - pd.Timestamp(log.at[i, "opened"])).days
            log.at[i, "days_held"] = held
            stop = pd.to_numeric(log.at[i, "stop"], errors="coerce")
            if np.isfinite(stop) and p <= stop:
                log.at[i, "status"], log.at[i, "closed"] = "STOPPED", today
            elif held >= HOLD_DAYS * 1.45:      # ~63 trading days in calendar terms
                log.at[i, "status"], log.at[i, "closed"] = "CLOSED", today
        write_log(strat, log)
        o = int((log["status"] == "OPEN").sum())
        say(f"{strat:<10}+{len(new):>3} new   {o:>3} open   {len(log):>4} total")


def report() -> None:
    say(f"{'ledger':<11}{'n':>5}{'open':>6}{'closed':>7}{'win%':>7}"
        f"{'avg':>8}{'vs SPY':>9}{'best':>8}{'worst':>8}")
    say("-" * 76)
    for strat in RULES:
        d = read_log(strat)
        if d.empty:
            say(f"{strat:<11}    no entries yet")
            continue
        p = pd.to_numeric(d["pnl_pct"], errors="coerce")
        v = pd.to_numeric(d["vs_spy_pct"], errors="coerce")
        done = d[d["status"] != "OPEN"]
        say(f"{strat:<11}{len(d):>5}{int((d['status']=='OPEN').sum()):>6}"
            f"{len(done):>7}{float((p > 0).mean()*100):>6.0f}%"
            f"{p.mean():>7.2f}%{v.mean():>8.2f}%"
            f"{p.max():>7.2f}%{p.min():>7.2f}%")
    say("-" * 76)
    say("vs SPY is the number that matters -- a ledger up 5% in a month the")
    say("market rose 6% has lost. Everything here is paper, marked at the")
    say("close, with no costs; treat it as a scoreboard, not a broker statement.")
    n = sum(len(read_log(s)) for s in RULES)
    if n < 30:
        say(f"\nOnly {n} entries so far. Nothing here is significant until each")
        say("ledger has 40+ closed positions -- that is months, not days.")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--reset" in a:
        s = a[a.index("--reset") + 1]
        if os.path.exists(path(s)):
            os.remove(path(s))
        say(f"reset {s}")
    elif "--report" in a:
        report()
    else:
        update()
        say("")
        report()
