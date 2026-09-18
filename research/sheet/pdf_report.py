"""One-page PDF wrapper for a single name's alert - the "PDF wrapper" the desk
sends alongside the Telegram message.

    from pdf_report import build_pdf
    path = build_pdf(evidence_dict)   # -> absolute path to the .pdf

Pure layout of an evidence dict that signal_engine / news / insider / ai already
produced. It computes nothing itself, so the PDF can never disagree with the
alert text.
"""
from __future__ import annotations

import datetime as dt
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "reports")


def _fmt(v, money=False, pct=False):
    if v is None or v == "":
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if money:
        return f"${f:,.2f}"
    if pct:
        return f"{f:+.1f}%"
    return f"{f:,.2f}"


def build_pdf(ev: dict, out_dir: str = OUT) -> str:
    os.makedirs(out_dir, exist_ok=True)
    sym = ev.get("symbol", "STOCK")
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    path = os.path.join(out_dir, f"{sym}_{stamp}.pdf")

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=18, spaceAfter=2)
    sub = ParagraphStyle("sub", parent=styles["Normal"], fontSize=9,
                         textColor=colors.grey, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12,
                        spaceBefore=8, spaceAfter=4)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9.5, leading=13)

    doc = SimpleDocTemplate(path, pagesize=A4, topMargin=16 * mm,
                            bottomMargin=16 * mm, leftMargin=16 * mm,
                            rightMargin=16 * mm, title=f"{sym} - shaheen desk")
    el = []
    el.append(Paragraph(f"{sym} &nbsp; {ev.get('name','')}", h1))
    el.append(Paragraph(
        f"{ev.get('sector','')} &nbsp;|&nbsp; price {_fmt(ev.get('price'), money=True)} "
        f"&nbsp;|&nbsp; {ev.get('zone_status','')} "
        f"&nbsp;|&nbsp; {dt.datetime.now():%Y-%m-%d %H:%M}", sub))

    # ratings + plan table
    rr = ev.get("rr")
    grid = [
        ["Fundamental /100", str(ev.get("fund_100", "-")),
         "Entry", _fmt(ev.get("entry"), money=True)],
        ["Technical /100", str(ev.get("tech_100", "-")),
         "Stop", _fmt(ev.get("stop"), money=True)],
        ["Halal", str(ev.get("halal", "-")),
         "Target", _fmt(ev.get("target"), money=True)],
        ["Reward : Risk", (f"{rr} : 1" if rr not in (None, "") else "-"),
         "Support / Resist", f"{_fmt(ev.get('support'))} / {_fmt(ev.get('resistance'))}"],
    ]
    t = Table(grid, colWidths=[38 * mm, 30 * mm, 38 * mm, 42 * mm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#555555")),
        ("TEXTCOLOR", (2, 0), (2, -1), colors.HexColor("#555555")),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("FONTNAME", (3, 0), (3, -1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f4f6f8")]),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#eeeeee")),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    el.append(t)

    el.append(Paragraph("Position (from your account size)", h2))
    el.append(Paragraph(
        f"Risking <b>${_fmt(ev.get('risk_dollars'))[1:]}</b> to make "
        f"<b>${_fmt(ev.get('gain_dollars'))[1:]}</b> on "
        f"<b>{ev.get('shares','-')}</b> shares "
        f"(position ~${_fmt(ev.get('position_dollars'))[1:]}).", body))

    el.append(Paragraph("Insider (SEC Form 4, last 30d)", h2))
    el.append(Paragraph(str(ev.get("insider", "-")), body))

    el.append(Paragraph("AI thesis", h2))
    thesis = str(ev.get("thesis", "AI offline")).replace("\n", "<br/>")
    el.append(Paragraph(thesis, body))

    news = ev.get("news") or []
    if news:
        el.append(Paragraph("Recent news", h2))
        for h in news[:6]:
            el.append(Paragraph("&bull; " + str(h), body))

    el.append(Spacer(1, 8))
    el.append(Paragraph(
        "Research tool for a halal, long-only cash account. Not investment "
        "advice. Verify halal status in Zoya/Musaffa before buying.", sub))
    doc.build(el)
    return path


if __name__ == "__main__":
    demo = {"symbol": "DEMO", "name": "Demo Corp", "sector": "Tech",
            "price": 100, "zone_status": "IN ZONE", "fund_100": 72,
            "tech_100": 65, "halal": "pass", "entry": 98, "stop": 92,
            "target": 110, "support": 96, "resistance": 111, "rr": 2.0,
            "risk_dollars": 375, "gain_dollars": 750, "shares": 62,
            "position_dollars": 6076, "insider": "quiet (30d)",
            "thesis": "Demo thesis.\nVERDICT: WATCH", "news": ["Demo headline"]}
    print(build_pdf(demo))
