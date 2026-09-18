"""GOLDEN alert engine - the piece that ties the desk together and pushes it to
your phone.

    python research/sheet/golden_alert.py --scan     # detect entries, alert new ones
    python research/sheet/golden_alert.py --dry       # same, but print, do not send
    python research/sheet/golden_alert.py --test SYM  # force one full alert for a name
    python research/sheet/golden_alert.py --macro     # send the world thesis now

WHAT IT DOES
For every name that ENTERS the golden buy zone (signal_engine.is_golden) it
sends one rich Telegram alert, once, and attaches a PDF:
  fundamental + technical rating /100, entry/stop/target, reward:risk, the exact
  dollars risked vs dollars gained on a stop-sized position, recent news, an SEC
  Form 4 insider read, and an AI thesis written on your Claude subscription.
Plus a periodic global-markets thesis (POSTURE line) for context.

FIRE-ONCE
State lives in golden_state.json keyed by symbol. A name alerts when it first
becomes golden and not again while it stays golden; it clears when it leaves, so
a genuine re-entry later alerts again. First ever run SEEDS the current golden
list and sends a single summary instead of a burst of full alerts.

AI/insider/news run ONLY for newly-triggered names, so a normal scan is cheap.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import sys

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import signal_engine as se          # noqa: E402
import news_feed as nf              # noqa: E402
import insider_watch as iw          # noqa: E402
import ai_layer as ai               # noqa: E402
import pdf_report as pr             # noqa: E402
import kuwait as kw                 # noqa: E402


def load_all():
    """US names plus any cached Kuwait names, tagged by market so the evidence
    builder knows Kuwait has no SEC insider layer."""
    us = se.load_signals()
    us["market"] = "US"
    k = kw.load()
    if k is not None and not k.empty:
        k["market"] = "KW"
        cols = sorted(set(us.columns) | set(k.columns))
        return pd.concat([us.reindex(columns=cols), k.reindex(columns=cols)],
                         ignore_index=True)
    return us

CONFIG = os.path.join(HERE, "config.json")
STATE = os.path.join(HERE, "golden_state.json")
MACRO_STATE = os.path.join(HERE, "golden_macro_state.json")

_OUT = io.TextIOWrapper(open(sys.stdout.fileno(), "wb", closefd=False),
                        encoding="utf-8", errors="replace")


def say(*a):
    print(*a, file=_OUT)
    _OUT.flush()


def cfg() -> dict:
    return json.load(open(CONFIG, encoding="utf-8")) if os.path.exists(CONFIG) else {}


def recipients(c: dict) -> list[str]:
    owner = str(c.get("telegram_chat_id", "")).strip()
    guests = [str(x).strip() for x in c.get("allowed_chats", []) if str(x).strip()]
    return [x for x in ([owner] + guests) if x and "PUT_YOUR" not in x]


# --------------------------------------------------------------- telegram
def _tok(c):
    return c.get("telegram_token", "")


def tg_message(text: str, chat: str, c: dict) -> bool:
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{_tok(c)}/sendMessage",
            data={"chat_id": chat, "text": text, "parse_mode": "HTML",
                  "disable_web_page_preview": "true"}, timeout=20).json()
        return bool(r.get("ok"))
    except Exception as e:
        say("send msg failed:", type(e).__name__, e)
        return False


def tg_document(path: str, caption: str, chat: str, c: dict) -> bool:
    try:
        with open(path, "rb") as f:
            r = requests.post(
                f"https://api.telegram.org/bot{_tok(c)}/sendDocument",
                data={"chat_id": chat, "caption": caption[:1000]},
                files={"document": (os.path.basename(path), f, "application/pdf")},
                timeout=60).json()
        return bool(r.get("ok"))
    except Exception as e:
        say("send doc failed:", type(e).__name__, e)
        return False


def broadcast(text: str, c: dict, pdf: str | None = None):
    for chat in recipients(c):
        tg_message(text, chat, c)
        if pdf:
            tg_document(pdf, f"{text.splitlines()[0][:80]}", chat, c)


# --------------------------------------------------------------- evidence
def build_evidence(row, c: dict, with_ai=True) -> dict:
    sym = row["symbol"]
    ev = {k: row.get(k) for k in (
        "symbol", "price", "sector", "zone_status", "fund_100", "tech_100",
        "entry", "stop", "target", "rr", "risk_dollars", "gain_dollars",
        "shares", "position_dollars", "support", "resistance",
        "trailingPE", "forwardPE", "profitMargins", "revenueGrowth",
        "debtToEquity")}
    ev["name"] = row.get("shortName") or sym
    ev["halal"] = row.get("halal_auto", "unknown")
    market = row.get("market", "US")
    ev["market"] = market
    news_sym = row.get("yf") or sym          # Kuwait news key is the .KW symbol
    ev["news"] = nf.headlines(news_sym, 6)
    if market == "KW":
        # No SEC in Kuwait - do not fake an insider read.
        ev["insider"] = "n/a - no insider filings in Kuwait"
    else:
        try:
            ev["insider"] = iw.insider_summary(sym, 30)["label"]
        except Exception:
            ev["insider"] = "insider check failed"
    ev["thesis"] = ai.stock_thesis(ev) if with_ai else "(AI skipped)"
    return ev


def format_alert(ev: dict) -> str:
    rr = ev.get("rr")
    lines = [
        f"<b>GOLDEN BUY ZONE - {ev['symbol']}</b>  {ev.get('name','')}",
        f"{ev.get('sector','')} | {ev.get('zone_status','')} | price ${ev.get('price')}",
        "",
        f"Fundamental <b>{ev.get('fund_100')}/100</b>   Technical <b>{ev.get('tech_100')}/100</b>   Halal: {ev.get('halal')}",
        f"Entry <b>${ev.get('entry')}</b>  Stop <b>${ev.get('stop')}</b>  Target <b>${ev.get('target')}</b>",
        f"Reward:Risk <b>{rr}:1</b>  -  risk <b>${ev.get('risk_dollars')}</b> to make <b>${ev.get('gain_dollars')}</b> on {ev.get('shares')} shares",
        f"Insider (30d): {ev.get('insider')}",
        "",
        f"<b>Thesis</b>\n{ev.get('thesis','')}",
    ]
    news = ev.get("news") or []
    if news:
        lines.append("")
        lines.append("<b>News</b>")
        lines += [f"- {h}" for h in news[:4]]
    return "\n".join(lines)


# --------------------------------------------------------------- state
def load_state(path):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}


def save_state(path, obj):
    json.dump(obj, open(path, "w", encoding="utf-8"), indent=2)


# --------------------------------------------------------------- macro
def macro_context(df, c):
    above = (df["vs_200sma"].astype(str) == "ABOVE").mean() * 100
    by_sec = df.groupby("sector")["chg_1d_pct"].mean().sort_values()
    movers = df.reindex(df["chg_1d_pct"].abs().sort_values(ascending=False).index)
    mv = ", ".join(f"{r.symbol} {r.chg_1d_pct:+.1f}%" for r in movers.head(5).itertuples())
    return {"date": dt.date.today().isoformat(),
            "pct_above_200sma": round(above),
            "n_golden": int(df["golden"].sum()),
            "sector_leaders": ", ".join(by_sec.tail(3).index[::-1]),
            "sector_laggards": ", ".join(by_sec.head(3).index),
            "movers": mv}


def maybe_macro(df, c, force=False):
    hrs = float(c.get("ai_macro_hours", 12))
    st = load_state(MACRO_STATE)
    last = st.get("last")
    now = dt.datetime.now()
    if not force and last:
        try:
            if (now - dt.datetime.fromisoformat(last)).total_seconds() < hrs * 3600:
                return None
        except ValueError:
            pass
    thesis = ai.macro_thesis(macro_context(df, c))
    save_state(MACRO_STATE, {"last": now.isoformat()})
    return thesis


# --------------------------------------------------------------- run
def run(scan=True, dry=False, force_symbol=None, do_macro=False):
    c = cfg()
    if not recipients(c) and not dry:
        say("no telegram recipients in config.json - use --dry, or set the token/chat id")
    df = load_all()
    gdf = df[df["golden"]].copy()
    golden_now = set(gdf["symbol"])
    say(f"universe {len(df)} | golden now {len(golden_now)}")

    if force_symbol:
        row = df[df["symbol"] == force_symbol.upper()]
        if row.empty:
            say("no such symbol:", force_symbol)
            return
        ev = build_evidence(row.iloc[0], c)
        txt = format_alert(ev)
        pdf = pr.build_pdf(ev)
        if dry:
            say(txt); say("PDF:", pdf)
        else:
            broadcast(txt, c, pdf)
            say("sent test alert for", force_symbol)
        return

    if do_macro:
        m = maybe_macro(df, c, force=True)
        head = f"<b>WORLD THESIS - {dt.date.today()}</b>\n"
        if dry:
            say(head + (m or ""))
        else:
            broadcast(head + (m or ""), c)
            say("sent macro thesis")
        return

    state = load_state(STATE)
    # first ever run: seed, don't flood
    if not state:
        for s in golden_now:
            state[s] = {"since": dt.date.today().isoformat()}
        save_state(STATE, state)
        names = ", ".join(sorted(golden_now)) or "(none)"
        msg = (f"<b>Golden watch armed</b>\n{len(golden_now)} names in the zone "
               f"now: {names}\nYou'll get a full alert when a NEW name enters.")
        say(msg) if dry else broadcast(msg, c)
        return

    prev = set(state.keys())
    entered = sorted(golden_now - prev)
    left = sorted(prev - golden_now)
    say(f"entered: {entered or '-'} | left: {left or '-'}")

    for s in left:
        state.pop(s, None)

    for s in entered:
        row = df[df["symbol"] == s].iloc[0]
        ev = build_evidence(row, c)
        txt = format_alert(ev)
        pdf = pr.build_pdf(ev)
        if dry:
            say("\n" + txt + f"\n[PDF {pdf}]")
        else:
            broadcast(txt, c, pdf)
            say("alerted", s)
        state[s] = {"since": dt.date.today().isoformat()}

    # periodic macro after the entries
    m = maybe_macro(df, c)
    if m:
        head = f"<b>WORLD THESIS - {dt.date.today()}</b>\n"
        say(head + m) if dry else broadcast(head + m, c)

    if not dry:
        save_state(STATE, state)
    say("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--macro", action="store_true")
    ap.add_argument("--test", metavar="SYMBOL")
    a = ap.parse_args()
    run(scan=a.scan or True, dry=a.dry, force_symbol=a.test, do_macro=a.macro)
