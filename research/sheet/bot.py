"""Telegram trigger bot: ask the desk questions, arm alerts, from your phone.

    python research/sheet/bot.py              # long-poll forever
    python research/sheet/bot.py --once       # drain pending updates and exit

COMMANDS
    /top                 momentum leaders right now
    /zone                names sitting in the buy zone
    /movers              biggest moves today, with the reason
    /why NVDA            full forensic breakdown for one stock
    /confirm NVDA        should I buy this NOW? the timing evidence
    /q NVDA              quick card: price, rate, momentum, zone, levels
    /upgrades            analyst upgrades in the last few days
    /unusual             unusual option premium today
    /watch MSFT 360      arm a price-below alert
    /watch MSFT 360 above     ... or price-above
    /unwatch MSFT        drop every alert on a symbol
    /alerts              what is armed, and what has already fired
    /follow Quinn Bolton      follow an analyst
    /follow Needham firm      follow a firm
    /following           who you follow
    /status              data freshness and job state
    /help                this list

WHY A BOT AND NOT JUST PUSH ALERTS
Push tells you when something you already thought of happens. A bot lets you
ask a question you had not pre-registered -- standing in a queue, away from
the desk. Same CSV cache underneath, so it can never disagree with the sheet
or the web UI.

SECURITY: only the chat id in config.json is answered. Anyone else who finds
the bot gets silence. The token lives in config.json, never in this file.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request

_OUT = io.TextIOWrapper(open(sys.stdout.fileno(), "wb", closefd=False),
                        encoding="utf-8", errors="replace")


def say(*a) -> None:
    print(*a, file=_OUT)
    _OUT.flush()


import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
CONFIG = os.path.join(HERE, "config.json")
OFFSET = os.path.join(HERE, "bot_offset.txt")


def _mod(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


al = _mod("al", os.path.join(HERE, "alerts.py"))
se = _mod("se", os.path.join(HERE, "signal_engine.py"))
ai = _mod("ai", os.path.join(HERE, "ai_layer.py"))
ga = _mod("ga", os.path.join(HERE, "golden_alert.py"))
tw = None


def cfg() -> dict:
    return json.load(open(CONFIG, encoding="utf-8")) if os.path.exists(CONFIG) else {}


def api(method: str, **params):
    c = cfg()
    tok = c.get("telegram_token", "")
    if not tok or tok.startswith("PASTE"):
        return {"ok": False, "error": "no token"}
    url = f"https://api.telegram.org/bot{tok}/{method}"
    data = urllib.parse.urlencode(params).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data),
                                    timeout=70) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def load(name: str) -> pd.DataFrame:
    p = os.path.join(CACHE, f"{name}.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def code(s: str) -> str:
    return f"<pre>{s}</pre>"


def cmd_top(_):
    d = load("summary")
    if d.empty:
        return "no data yet"
    d = d.sort_values("MOMENTUM_SCORE", ascending=False).head(12)
    out = [f"{'sym':<7}{'score':>6}{'price':>10}{'1d':>7}  hot"]
    for _, r in d.iterrows():
        out.append(f"{r.symbol:<7}{r.MOMENTUM_SCORE:>6.0f}{r.price:>10.2f}"
                   f"{r.chg_1d_pct:>+7.1f}  {r.get('hot','') or ''}")
    return "<b>MOMENTUM LEADERS</b>\n" + code("\n".join(out))


def cmd_zone(_):
    d = load("summary")
    d = d[d.zone_status.isin(["IN ZONE", "AT ZONE"])].sort_values(
        "RATE", ascending=False).head(12)
    if d.empty:
        return "nothing in the buy zone right now"
    out = [f"{'sym':<7}{'price':>9}{'buy':>9}{'rate':>6}  status"]
    for _, r in d.iterrows():
        out.append(f"{r.symbol:<7}{r.price:>9.2f}{(r.perfect_buy or 0):>9.2f}"
                   f"{(r.RATE or 0):>6.0f}  {r.zone_status}")
    return ("<b>IN THE BUY ZONE</b>\n" + code("\n".join(out))
            + "\nNote: buying at support/zone measured -2.0% vs random. "
              "Use it to avoid overpaying, not to pick.")


def cmd_movers(_):
    d = load("whyitmoved")
    if d.empty:
        return "no data"
    d = d.reindex(d.move_pct.abs().sort_values(ascending=False).index).head(6)
    parts = []
    for _, r in d.iterrows():
        parts.append(f"<b>{r.symbol}  {r.move_pct:+.2f}%</b>\n{r.VERDICT}")
    return "<b>BIGGEST MOVES TODAY</b>\n\n" + "\n\n".join(parts)


def cmd_why(arg):
    if not arg:
        return "usage: /why NVDA"
    d = load("whyitmoved")
    r = d[d.symbol == arg.upper()]
    if r.empty:
        return f"{arg.upper()} not found"
    r = r.iloc[0]
    lines = [f"<b>{r.symbol}  {r.move_pct:+.2f}%</b>", r.VERDICT, ""]
    lines.append(code(
        f"where     {r.move_where}\n"
        f"gap       {r.gap_pct:+.2f}%   intraday {r.intraday_pct:+.2f}%\n"
        f"size      {r.move_in_sigma:+.1f} sigma, {r.move_in_atr:+.1f} ATR\n"
        f"vs year   bigger than {r.pctile_vs_own_year:.0f}% of its own days\n"
        f"volume    {r.volume_x_normal:.2f}x normal\n"
        f"sector    peers {r.sector_move_pct:+.2f}%, this {r.vs_sector_pct:+.2f}% vs them\n"
        f"peers     {r.pct_peers_same_way:.0f}% moved the same way\n"
        f"explained {r.explained_pp:+.2f}pp   company {r.company_specific_pp:+.2f}pp\n"
        f"model R2  {r.r2_of_model:.2f}"))
    if isinstance(r.get("analyst_action"), str) and r["analyst_action"]:
        lines.append(f"catalyst: {r['analyst_action']}")
    return "\n".join(lines)


def cmd_q(arg):
    if not arg:
        return "usage: /q NVDA"
    d = load("summary")
    r = d[d.symbol == arg.upper()]
    if r.empty:
        return f"{arg.upper()} not found"
    r = r.iloc[0]
    return (f"<b>{r.symbol}  {r.get('shortName','')}</b>\n" + code(
        f"price     ${r.price:,.2f}  ({r.chg_1d_pct:+.2f}% today)\n"
        f"rate      {r.get('RATE',float('nan')):.0f}/100  ({r.get('rate_grade','')})\n"
        f"momentum  {r.get('MOMENTUM_SCORE',float('nan')):.0f}/100 {r.get('hot','') or ''}\n"
        f"trend     {r.vs_200sma} 200d ({r.pct_vs_200sma:+.1f}%)\n"
        f"rsi       {r.rsi14:.0f}\n"
        f"support   ${r.support:,.2f} ({r.pct_to_support:+.1f}%)\n"
        f"resist    ${r.resistance:,.2f} ({r.pct_to_resistance:+.1f}%)\n"
        f"zone      {r.zone_status}   buy ~${r.get('perfect_buy',0) or 0:,.2f}\n"
        f"halal     {r.get('halal_auto','')}\n"
        f"size      {r.get('shares_for_$250_risk',0) or 0:.0f} shares = $250 risk"))


def cmd_upgrades(_):
    d = load("analysts")
    # last_action only exists when a local ratings tape has been swept in
    # (research/analysts/ratings_tape.csv). On a machine without it the
    # analysts sheet carries only the yfinance consensus, so say so plainly
    # instead of throwing on a missing column.
    if "last_action" not in d.columns:
        return ("no local rating tape on this machine yet -- run "
                "research/analysts/copy_desk.py to build one, then /upgrades "
                "will list dated upgrades and initiations")
    d = d[d.last_action.astype(str).str.startswith(("Upgrade", "Initiates"))]
    d = d.sort_values("days_ago").head(12)
    if d.empty:
        return "no recent upgrades on the tape"
    out = [f"{'sym':<7}{'action':<11}{'rating':<7}{'upside':>8}  firm"]
    for _, r in d.iterrows():
        u = r.get("last_target_upside_pct")
        out.append(f"{r.symbol:<7}{str(r.last_action)[:10]:<11}"
                   f"{str(r.last_rating)[:6]:<7}"
                   f"{(f'{u:+.0f}%' if pd.notna(u) else '-'):>8}  "
                   f"{str(r.last_firm)[:18]}")
    return ("<b>RECENT UPGRADES / INITIATIONS</b>\n" + code("\n".join(out))
            + "\nMeasured: entering on the public rating date underperformed "
              "a nearby day by 0.7-1.6%. Context, not a buy.")


def cmd_unusual(_):
    d = load("bigmoney")
    d = d[d.UNUSUAL == "UNUSUAL"].head(10)
    if d.empty:
        return "nothing unusual in the option tape today"
    out = [f"{'sym':<7}{'score':>6}{'call%':>7}  skew / biggest strike"]
    for _, r in d.iterrows():
        out.append(f"{r.symbol:<7}{r.unusual_score:>6.0f}{r.call_pct:>7.0f}  "
                   f"{r.get('skew','') or '-'} {str(r.get('max_strike',''))[:22]}")
    return "<b>UNUSUAL OPTION ACTIVITY</b>\n" + code("\n".join(out))


def cmd_watch(arg):
    p = (arg or "").split()
    if len(p) < 2:
        return "usage: /watch MSFT 360   (add 'above' for price-above)"
    sym, lvl = p[0].upper(), p[1]
    typ = "price_above" if len(p) > 2 and p[2].lower().startswith("ab") else "price_below"
    w = al.load_watchlist()
    w = w[~((w.symbol.astype(str).str.upper() == sym) & (w.type == typ))]
    new = pd.DataFrame([{"symbol": sym, "type": typ, "level": lvl,
                         "note": "via telegram", "fired": ""}])
    pd.concat([w, new], ignore_index=True).to_csv(al.WATCH, index=False)
    d = load("summary")
    r = d[d.symbol == sym]
    now = f" (now ${r.iloc[0].price:,.2f})" if not r.empty else ""
    return f"armed: {sym} {typ.replace('_',' ')} {lvl}{now}"


def cmd_unwatch(arg):
    if not arg:
        return "usage: /unwatch MSFT"
    sym = arg.split()[0].upper()
    w = al.load_watchlist()
    n0 = len(w)
    w = w[w.symbol.astype(str).str.upper() != sym]
    w.to_csv(al.WATCH, index=False)
    return f"removed {n0 - len(w)} alert(s) on {sym}"


def cmd_alerts(_):
    w = al.load_watchlist()
    if w.empty:
        return "no alerts armed"
    out = []
    for _, r in w.iterrows():
        f = str(r.get("fired") or "").strip()
        state = "FIRED " + f[:16] if f and f != "nan" else "armed"
        out.append(f"{str(r.symbol):<7}{str(r['type']):<19}"
                   f"{str(r.get('level') or ''):>9}  {state}")
    return "<b>ALERTS</b>\n" + code("\n".join(out))


def cmd_follow(arg):
    global tw
    if tw is None:
        tw = _mod("tw", os.path.join(HERE, "tape_watch.py"))
    if not arg:
        return "usage: /follow Quinn Bolton   |   /follow Needham firm"
    p = arg.split()
    kind = "firm" if p[-1].lower() == "firm" else "analyst"
    val = " ".join(p[:-1]) if kind == "firm" and len(p) > 1 else arg
    tw.add_follow(val.strip(), kind)
    return f"following {kind}: {val.strip()}\nNext sweep picks it up, no restart."


def cmd_following(_):
    global tw
    if tw is None:
        tw = _mod("tw", os.path.join(HERE, "tape_watch.py"))
    f = tw.follow_list()
    if not f:
        return "not following anyone yet"
    return "<b>FOLLOWING</b>\n" + code(
        "\n".join(f"{r.get('kind','analyst'):<9}{r['value']}" for r in f))


def cmd_status(_):
    p = os.path.join(CACHE, "summary.csv")
    if not os.path.exists(p):
        return "no data cache yet"
    age = (time.time() - os.path.getmtime(p)) / 60
    tape = os.path.join(HERE, "..", "analysts", "ratings_tape.csv")
    n_tape = sum(1 for _ in open(tape, encoding="utf-8")) - 1 \
        if os.path.exists(tape) else 0
    w = al.load_watchlist()
    return "<b>STATUS</b>\n" + code(
        f"prices      {age:.0f} min old\n"
        f"stocks      {len(load('summary'))}\n"
        f"rating tape {n_tape:,} events\n"
        f"alerts      {len(w)} armed\n"
        f"desk UI     http://127.0.0.1:8777")


def cmd_confirm(arg):
    """The question that actually gets asked: I like this one -- buy now?"""
    if not arg:
        return "usage: /confirm NVDA"
    cf = _mod("cf", os.path.join(HERE, "confirm.py"))
    d = cf.load()
    r = d[d.symbol == arg.split()[0].upper()]
    if r.empty:
        return arg.upper() + " not in the sheet"
    r = r.iloc[0]
    con, warn, veto, score = cf.check(r)
    v = ("DO NOT BUY YET" if veto else "CONFIRMED" if score >= 4
         else "PARTIAL" if score >= 2 else "NO CONFIRMATION")
    out = ["<b>" + str(r["symbol"]) + "  $" + format(cf.num(r, "price"), ",.2f") + "</b>",
           "<b>" + v + "</b>  (score " + str(score) + "/8)", ""]
    for x in veto:
        out.append("STOP: " + x)
    for x in con:
        out.append("+ " + x)
    for x in warn:
        out.append("! " + x)
    rows = [
        "from 52w high   " + format(cf.num(r, "pct_from_52w_high"), ">7.1f") + "%",
        "off 60d low     " + format(cf.num(r, "pct_off_recent_low"), ">7.1f") + "%",
        "R:R             " + format(cf.num(r, "RR"), ">7.2f") + ":1",
        "needs to work   " + format(cf.num(r, "breakeven_hit_rate_pct"), ">7.1f") + "%",
        "momentum        " + format(cf.num(r, "MOMENTUM_SCORE"), ">7.0f") + "/100",
    ]
    out.append(code(chr(10).join(rows)))
    p3 = cf.num(r, "price_for_3R")
    px = cf.num(r, "price")
    if p3 == p3 and px == px and px > p3:
        out.append("WAIT FOR $" + format(p3, ",.2f") + " for a 3:1 setup ("
                   + format((p3 / px - 1) * 100, "+.1f") + "%)")
    return chr(10).join(out)


def _i(v, default="-"):
    try:
        import math as _m
        f = float(v)
        return str(int(f)) if not _m.isnan(f) else default
    except (TypeError, ValueError):
        return default


def cmd_rate(arg):
    """Fundamental + technical rating /100, plan and dollar risk for one name."""
    sym = (arg or "").strip().upper()
    if not sym:
        return "usage: /rate SYMBOL"
    d = se.load_signals()
    r = d[d.symbol == sym]
    if r.empty:
        return f"no data for {sym}"
    r = r.iloc[0]
    rows = [
        f"fundamental    {_i(r.fund_100)}/100",
        f"technical      {_i(r.tech_100)}/100",
        f"zone           {r.zone_status}",
        f"entry/stop/tgt {r.entry} / {r.stop} / {r.target}",
        f"reward:risk    {r.rr}:1",
        f"risk ${_i(r.risk_dollars)} -> gain ${_i(r.gain_dollars)} on {_i(r.shares)} sh",
        f"halal          {r.get('halal_auto', '?')}",
    ]
    return f"<b>{sym} RATING</b>\n" + code(chr(10).join(rows))


def cmd_thesis(arg):
    """Full AI thesis for one name (news + insider + rating -> Claude)."""
    sym = (arg or "").strip().upper()
    if not sym:
        return "usage: /thesis SYMBOL"
    d = se.load_signals()
    r = d[d.symbol == sym]
    if r.empty:
        return f"no data for {sym}"
    ev = ga.build_evidence(r.iloc[0], ga.cfg())
    return (f"<b>{sym} THESIS</b>  F {_i(ev.get('fund_100'))} / T {_i(ev.get('tech_100'))} "
            f"| R:R {ev.get('rr')}:1\nInsider: {ev.get('insider')}\n\n{ev.get('thesis','')}")


def cmd_macro(_):
    """Overall world/markets thesis on your Claude subscription."""
    d = se.load_signals()
    m = ai.macro_thesis(ga.macro_context(d, ga.cfg()))
    return f"<b>WORLD THESIS</b>\n{m}"


def cmd_kw(_):
    """Kuwaiti names in the golden / buy zone (valuation + technical only)."""
    d = load("kuwait")
    if d.empty:
        return "no Kuwait data yet - run research/sheet/kuwait.py to build it"
    is_g = d["golden"].astype(str).isin(["True", "true", "1"]) if "golden" in d.columns else None
    g = d[is_g] if is_g is not None and is_g.any() else d[d.zone_status.isin(["AT ZONE", "IN ZONE"])]
    g = g.sort_values("fund_100", ascending=False).head(20)
    out = [f"{'sym':<10}{'F':>4}{'T':>4}{'R:R':>6}  zone"]
    for r in g.itertuples():
        rr = format(r.rr, ">6") if r.rr == r.rr else "   -"
        out.append(f"{r.symbol:<10}{_i(r.fund_100):>4}{_i(r.tech_100):>4}{rr:>6}  {r.zone_status}")
    return ("<b>KUWAIT buy zone</b>\n" + code(chr(10).join(out))
            + "\nValuation + technical only - no insider/US-news layer in Kuwait.")


def cmd_golden(_):
    """Names in the golden buy zone right now."""
    d = se.load_signals()
    g = d[d.golden].sort_values("fund_100", ascending=False)
    if g.empty:
        return "nothing in the golden buy zone right now"
    out = [f"{'sym':<6}{'F':>4}{'T':>4}{'R:R':>6}  zone"]
    for r in g.head(20).itertuples():
        out.append(f"{r.symbol:<6}{_i(r.fund_100):>4}{_i(r.tech_100):>4}"
                   f"{format(r.rr, '>6') if r.rr == r.rr else '   -':>6}  {r.zone_status}")
    return ("<b>GOLDEN BUY ZONE</b>\n" + code(chr(10).join(out))
            + "\nA full alert with thesis + PDF fires automatically on new entries.")


CMDS = {"confirm": cmd_confirm, "top": cmd_top, "zone": cmd_zone, "movers": cmd_movers, "why": cmd_why,
        "q": cmd_q, "upgrades": cmd_upgrades, "unusual": cmd_unusual,
        "rate": cmd_rate, "thesis": cmd_thesis, "macro": cmd_macro, "golden": cmd_golden, "kw": cmd_kw,
        "watch": cmd_watch, "unwatch": cmd_unwatch, "alerts": cmd_alerts,
        "follow": cmd_follow, "following": cmd_following, "status": cmd_status}


def cmd_help(_):
    return ("<b>DESK BOT</b>\n" + code(
        "/top          momentum leaders\n"
        "/zone         names in the buy zone\n"
        "/movers       biggest moves + why\n"
        "/why NVDA     full forensics\n"
        "/q NVDA       quick card\n"
        "/rate NVDA    fundamental + technical /100 + plan\n"
        "/thesis NVDA  AI thesis (news+insider+rating)\n"
        "/golden       names in the golden buy zone now\n"
        "/macro        world/markets thesis\n"
        "/upgrades     recent analyst upgrades\n"
        "/unusual      unusual option premium\n"
        "/watch MSFT 360    arm an alert\n"
        "/unwatch MSFT      drop alerts\n"
        "/alerts       what is armed\n"
        "/follow NAME  follow an analyst\n"
        "/following    who you follow\n"
        "/status       freshness + counts"))


CMDS["help"] = cmd_help
CMDS["start"] = cmd_help


def handle(text: str) -> str:
    text = (text or "").strip()
    if not text.startswith("/"):
        return ("send /help for commands, or /q SYMBOL for a quick card")
    parts = text[1:].split(maxsplit=1)
    name = parts[0].split("@")[0].lower()
    arg = parts[1] if len(parts) > 1 else ""
    fn = CMDS.get(name)
    if not fn:
        return f"unknown command /{name} -- try /help"
    try:
        return fn(arg)
    except Exception as e:
        return f"error in /{name}: {type(e).__name__}: {e}"


def offset() -> int:
    try:
        return int(open(OFFSET).read().strip())
    except Exception:
        return 0


def poll(once: bool = False) -> None:
    c = cfg()
    # The owner's chat drives alerts; `allowed_chats` is an optional guest list
    # so a friend can query the same desk without being able to arm alerts on
    # the owner's phone. Everyone else still gets silence.
    owner = str(c.get("telegram_chat_id", "")).strip()
    guests = {str(x).strip() for x in c.get("allowed_chats", []) if str(x).strip()}
    allowed = {owner} | guests if owner else guests
    if not allowed:
        say("no telegram_chat_id in config.json -- run alerts.py --setup first")
        return
    say(f"bot listening. {len(allowed)} chat(s) answered: {sorted(allowed)}. "
        "Ctrl+C to stop.")
    while True:
        r = api("getUpdates", offset=offset() + 1, timeout=50)
        if not r.get("ok"):
            say(f"getUpdates failed: {r.get('error') or r.get('description')}")
            if once:
                return
            time.sleep(10)
            continue
        ups = r.get("result", [])
        for u in ups:
            open(OFFSET, "w").write(str(u["update_id"]))
            m = u.get("message") or u.get("edited_message") or {}
            chat = str((m.get("chat") or {}).get("id", ""))
            txt = m.get("text", "")
            if chat not in allowed:
                say(f"ignored message from chat {chat}")
                continue
            say(f"<- {txt}")
            reply = handle(txt)
            api("sendMessage", chat_id=chat, text=reply, parse_mode="HTML",
                disable_web_page_preview="true")
            say(f"-> {reply.splitlines()[0][:70]}")
        if once:
            say(f"drained {len(ups)} update(s)")
            return


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--test" in a:
        # exercise every command locally, no network
        for name in ["help", "top", "zone", "movers", "upgrades", "unusual",
                     "alerts", "status", "following"]:
            out = handle("/" + name)
            say(f"/{name:<11}{'OK' if out and 'error' not in out[:30] else 'FAIL'}"
                f"  {len(out)} chars")
        for t in ["/q NVDA", "/why NVDA", "/q ZZZZ"]:
            out = handle(t)
            say(f"{t:<12}{'OK' if out else 'FAIL'}  {out.splitlines()[0][:52]}")
    else:
        poll(once="--once" in a)
