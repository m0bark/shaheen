"""Free per-ticker news for the names we actually pick, via yfinance.

    from news_feed import headlines
    headlines("NVDA", limit=6)   # -> ["title (provider, date)", ...]

Only ever called on the picked set (golden + watchlist), never the whole index -
the point is context on the handful we might buy, not a firehose.

yfinance has shipped two news shapes; this reads both: the older flat dict with
top-level 'title'/'link', and the newer nested {'content': {'title', 'pubDate',
'provider': {'displayName'}, 'canonicalUrl': {'url'}}}.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")
import yfinance as yf


def _one(item: dict) -> dict | None:
    c = item.get("content") if isinstance(item.get("content"), dict) else None
    if c:
        title = c.get("title")
        prov = (c.get("provider") or {}).get("displayName")
        date = (c.get("pubDate") or c.get("displayTime") or "")[:10]
        url = (c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url")
    else:
        title = item.get("title")
        prov = item.get("publisher")
        date = ""
        url = item.get("link")
    if not title:
        return None
    return {"title": title.strip(), "provider": prov or "", "date": date or "",
            "url": url or ""}


def news(ticker: str, limit: int = 6) -> list[dict]:
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []
    out = []
    for it in raw:
        one = _one(it)
        if one:
            out.append(one)
        if len(out) >= limit:
            break
    return out


def headlines(ticker: str, limit: int = 6) -> list[str]:
    """One-line strings for a Telegram/PDF block."""
    rows = news(ticker, limit)
    lines = []
    for r in rows:
        tag = " ".join(x for x in (r["provider"], r["date"]) if x)
        lines.append(f"{r['title']}" + (f"  ({tag})" if tag else ""))
    return lines


if __name__ == "__main__":
    for t in ("NVDA", "AAPL"):
        print(f"=== {t} ===")
        for h in headlines(t, 4):
            print("  -", h)
