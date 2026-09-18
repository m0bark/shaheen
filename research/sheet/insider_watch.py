"""Insider warning from SEC Form 4 filings - free, direct from EDGAR.

    from insider_watch import insider_summary
    insider_summary("NVDA")   # -> {"label": "...", "n_buyers": .., "buy_usd": ..}

Form 4 reports an insider's trade within 2 business days. This reads the last
`days` of a name's Form 4s, classifies each transaction by its SEC code
(P = open-market purchase, S = sale) and sums the dollar value, then returns a
one-line label for the alert. Purchases are the informative side: the repo's one
validated edge is insider-cluster BUYING at <=1-session latency, so a cluster of
buyers is a genuine tailwind and heavy selling is the warning.

Only run on the picked set (a few dozen names): each ticker costs the submissions
call plus a few filing fetches, and SEC asks for a real User-Agent and <=10 req/s.
"""
from __future__ import annotations

import datetime as dt
import os
import re
import time
import xml.etree.ElementTree as ET

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
_UA = {"User-Agent": "shaheen desk research contact@example.com"}
_S = requests.Session()
_S.headers.update(_UA)
_CIK: dict[str, str] = {}


def _load_cik() -> None:
    """ticker -> zero-padded 10-digit CIK, from SEC's own map (cached in-proc)."""
    global _CIK
    if _CIK:
        return
    try:
        j = _S.get("https://www.sec.gov/files/company_tickers.json", timeout=25).json()
        for row in j.values():
            _CIK[row["ticker"].upper()] = str(row["cik_str"]).zfill(10)
    except Exception:
        _CIK = {}


def _form4_xml_urls(cik: str, days: int) -> list[str]:
    """URLs of recent Form 4 primary XML docs within `days`."""
    try:
        j = _S.get(f"https://data.sec.gov/submissions/CIK{cik}.json", timeout=25).json()
    except Exception:
        return []
    rec = j.get("filings", {}).get("recent", {})
    forms = rec.get("form", [])
    accs = rec.get("accessionNumber", [])
    docs = rec.get("primaryDocument", [])
    dates = rec.get("filingDate", [])
    cutoff = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    urls = []
    for f, a, doc, d in zip(forms, accs, docs, dates):
        if f != "4" or d < cutoff:
            continue
        acc = a.replace("-", "")
        cik_int = str(int(cik))
        # primaryDocument points at the XSL-STYLED view (e.g. "xslF345X06/
        # wk-form4_123.xml"); the machine-readable XML sits at the same path
        # without that leading xsl folder. Strip it or we parse HTML and get 0.
        doc = re.sub(r"^xsl[^/]*/", "", doc)
        urls.append(f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc}/{doc}")
        if len(urls) >= 25:
            break
    return urls


def _parse_form4(url: str) -> list[dict]:
    """Return [{owner, code, shares, price}] for non-derivative transactions."""
    try:
        txt = _S.get(url, timeout=25).text
        if "<ownershipDocument" not in txt:
            return []
        root = ET.fromstring(txt.encode("utf-8", "replace"))
    except Exception:
        return []
    owner = None
    ro = root.find(".//reportingOwner/reportingOwnerId/rptOwnerName")
    if ro is not None and ro.text:
        owner = ro.text.strip()
    out = []
    for tr in root.findall(".//nonDerivativeTransaction"):
        code = tr.findtext(".//transactionCoding/transactionCode") or ""
        sh = tr.findtext(".//transactionAmounts/transactionShares/value")
        px = tr.findtext(".//transactionAmounts/transactionPricePerShare/value")
        try:
            shares = float(sh) if sh else 0.0
            price = float(px) if px else 0.0
        except ValueError:
            shares, price = 0.0, 0.0
        out.append({"owner": owner or "insider", "code": code.strip(),
                    "shares": shares, "price": price})
    return out


def insider_summary(ticker: str, days: int = 30, polite: float = 0.12) -> dict:
    """One-line insider read for the last `days`."""
    _load_cik()
    cik = _CIK.get(ticker.upper())
    base = {"ticker": ticker, "n_buyers": 0, "n_sellers": 0,
            "buy_usd": 0.0, "sell_usd": 0.0, "label": "no data"}
    if not cik:
        base["label"] = "not on SEC (foreign/OTC?)"
        return base
    urls = _form4_xml_urls(cik, days)
    if not urls:
        base["label"] = "quiet - no Form 4s in %dd" % days
        return base
    buyers, sellers = set(), set()
    buy_usd = sell_usd = 0.0
    for u in urls:
        for t in _parse_form4(u):
            val = t["shares"] * t["price"]
            if t["code"] == "P":                 # open-market purchase
                buyers.add(t["owner"]); buy_usd += val
            elif t["code"] == "S":               # sale
                sellers.add(t["owner"]); sell_usd += val
        time.sleep(polite)
    base.update(n_buyers=len(buyers), n_sellers=len(sellers),
                buy_usd=round(buy_usd), sell_usd=round(sell_usd))
    # dollar floors so a token sale is not dressed up as a warning. A real
    # insider signal is either a cluster of open-market buyers or genuinely
    # large selling; everything smaller is context, not an alert.
    BIG = 2_000_000        # "heavy" selling floor
    MEANINGFUL = 250_000   # below this, treat as quiet noise
    if len(buyers) >= 2 and buy_usd > sell_usd:
        base["label"] = "BUY CLUSTER: %d insiders bought $%s (%dd)" % (
            len(buyers), _m(buy_usd), days)
    elif len(buyers) >= 1 and buy_usd >= max(sell_usd, MEANINGFUL):
        base["label"] = "%d insider buy(s) $%s (%dd)" % (len(buyers), _m(buy_usd), days)
    elif sell_usd >= BIG and sell_usd > buy_usd * 2:
        base["label"] = "WARNING heavy selling: $%s by %d (%dd)" % (
            _m(sell_usd), len(sellers), days)
    elif sell_usd >= MEANINGFUL:
        base["label"] = "some selling $%s (%dd)" % (_m(sell_usd), days)
    else:
        base["label"] = "quiet (%dd)" % days
    return base


def _m(x: float) -> str:
    x = float(x)
    if x >= 1e6:
        return "%.1fM" % (x / 1e6)
    if x >= 1e3:
        return "%.0fK" % (x / 1e3)
    return "%.0f" % x


if __name__ == "__main__":
    import sys
    for t in (sys.argv[1:] or ["NVDA", "MU"]):
        print(t, "->", insider_summary(t)["label"])
