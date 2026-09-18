"""AI analysis on your Claude subscription, via the local `claude` CLI.

    from ai_layer import stock_thesis, macro_thesis, ai_status

The desk computes every NUMBER deterministically (signal_engine.py). This layer
only asks Claude to REASON OVER those numbers plus the news and insider facts we
hand it - it is never asked to invent a price, a level or a rating. That is the
"evidence pack in, judgement out" pattern; it keeps the AI from hallucinating
figures because the figures are supplied, not requested.

Billing: `claude -p` runs one-shot on whatever subscription the CLI is logged
into. NOTHING here works until `claude` is logged in on this machine.

The logged-OUT trap (learned the hard way): the CLI answers "Not logged in ·
Please run /login" AS the reply and still exits 0. So a naive caller pastes that
string into the thesis and a scheduled job records it as analysis. Every call
here is screened for that and returns a clean "AI offline" instead.
"""
from __future__ import annotations

import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "config.json")

_LOGIN_MARKERS = ("not logged in", "please run /login", "/login",
                  "invalid api key", "authentication_error", "credit balance")


def cfg() -> dict:
    return json.load(open(CONFIG, encoding="utf-8")) if os.path.exists(CONFIG) else {}


def _run_claude(prompt: str, model: str = "sonnet", timeout: int = 150) -> str | None:
    """One-shot claude call. Returns clean text, or None when the CLI is not
    usable (logged out, empty, timed out, missing)."""
    try:
        r = subprocess.run(
            ["claude", "-p", prompt, "--model", model],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, shell=(os.name == "nt"))
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    out = (r.stdout or "").strip()
    if not out:
        return None
    low = out.lower()
    if any(m in low for m in _LOGIN_MARKERS) and len(out) < 200:
        return None                     # the logged-out reply, not analysis
    return out


def ai_status() -> tuple[bool, str]:
    """Quick liveness probe for the AI layer. Cheap: one tiny prompt."""
    if not cfg().get("ai_enabled", True):
        return False, "AI disabled in config"
    out = _run_claude("Reply with exactly: OK", timeout=60)
    if out is None:
        return False, "AI offline - run `claude` and /login on this machine"
    return True, "AI online"


def stock_thesis(ev: dict, model: str | None = None) -> str:
    """Concise buy-side thesis for ONE name, grounded in the evidence pack."""
    model = model or cfg().get("ai_model", "sonnet")
    news = ev.get("news") or []
    news_txt = "\n".join(f"  - {h}" for h in news[:6]) or "  (none fetched)"
    prompt = f"""You are a disciplined buy-side analyst for a HALAL, long-only,
cash-account retail book (no options, shorts or margin). Benchmark is SPUS.
Write a TIGHT thesis (max ~150 words) for {ev.get('symbol')} using ONLY the
facts below. Do not invent numbers. End with one line: VERDICT: BUY / WATCH / AVOID
and one line: KEY RISK: <one risk>.

FACTS
 name/sector : {ev.get('name')} / {ev.get('sector')}
 price       : {ev.get('price')}
 fundamental rating /100 : {ev.get('fund_100')}
 technical rating /100   : {ev.get('tech_100')}
 zone        : {ev.get('zone_status')}  (entry {ev.get('entry')}, stop {ev.get('stop')}, target {ev.get('target')})
 reward:risk : {ev.get('rr')}   risking ${ev.get('risk_dollars')} to make ${ev.get('gain_dollars')} on {ev.get('shares')} shares
 valuation   : trailingPE {ev.get('trailingPE')}, forwardPE {ev.get('forwardPE')}, profitMargin {ev.get('profitMargins')}
 growth/debt : revenueGrowth {ev.get('revenueGrowth')}, debtToEquity {ev.get('debtToEquity')}
 halal       : {ev.get('halal')}
 insider     : {ev.get('insider')}
 recent news headlines:
{news_txt}
"""
    out = _run_claude(prompt, model=model)
    return out or "AI offline - run `claude` and /login on this machine to enable analysis."


def macro_thesis(ctx: dict, model: str | None = None) -> str:
    """Overall world/market thesis, grounded in the breadth/sector facts we feed.
    Claude has no live feed, so it is told to reason from the supplied readings
    and its own macro knowledge, and to flag what it cannot see."""
    model = model or cfg().get("ai_model", "sonnet")
    prompt = f"""You are the strategist for a halal, long-only retail book
(benchmark SPUS). Write a SHORT global markets thesis (max ~180 words) as of
{ctx.get('date')}. Use the readings below; where you rely on general knowledge
rather than a supplied number, say so. Cover: overall risk posture (risk-on /
neutral / risk-off), what the breadth and sector spread imply, the main macro
watch-items into the next few weeks, and what it means for a dip-buying cash
account. End with one line: POSTURE: RISK-ON / NEUTRAL / RISK-OFF.

READINGS (from our own desk)
 index breadth   : {ctx.get('pct_above_200sma')}% of names above their 200-day
 golden setups   : {ctx.get('n_golden')} names in the golden buy zone now
 leading sectors : {ctx.get('sector_leaders')}
 lagging sectors : {ctx.get('sector_laggards')}
 biggest movers  : {ctx.get('movers')}
"""
    out = _run_claude(prompt, model=model)
    return out or "AI offline - run `claude` and /login on this machine to enable the macro thesis."


if __name__ == "__main__":
    ok, msg = ai_status()
    print("AI status:", msg)
