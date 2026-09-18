"""Orchestrator. One entry point for Task Scheduler.

    python research/sheet/run.py --hourly    prices -> technicals -> workbook -> alerts
    python research/sheet/run.py --daily     fundamentals + options, then --hourly
    python research/sheet/run.py --alerts    watchlist only, no fetching
    python research/sheet/run.py --gsheet    push to Google Sheets only
    python research/sheet/run.py --once      full cold start

The split exists because fundamentals cost one request per ticker (~12 min for
503) and change four times a year, while prices come back for the whole index
in one batched call. Running the slow path hourly would be ~12,000 requests a
day to restate numbers that did not move.
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import time

_OUT = io.TextIOWrapper(open(sys.stdout.fileno(), "wb", closefd=False),
                        encoding="utf-8", errors="replace")


def say(*a) -> None:
    print(*a, file=_OUT)
    _OUT.flush()


HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def step(script: str, *args) -> bool:
    t0 = time.time()
    say(f"--- {script} {' '.join(args)}")
    r = subprocess.run([PY, os.path.join(HERE, script), *args],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    tail = [l for l in (r.stdout or "").splitlines() if l.strip()][-4:]
    for l in tail:
        say(f"    {l}")
    if r.returncode != 0:
        say(f"    FAILED rc={r.returncode}")
        for l in (r.stderr or "").splitlines()[-6:]:
            say(f"    ! {l}")
        return False
    say(f"    ok ({time.time() - t0:.0f}s)")
    return True


def main() -> None:
    a = sys.argv[1:] or ["--hourly"]
    say(f"=== {time.strftime('%Y-%m-%d %H:%M')} {' '.join(a)}")
    if "--gsheet" in a:
        step("to_gsheet.py")
        return
    if "--daily" in a or "--once" in a:
        step("fetch.py", "--fundamentals")
        step("fetch.py", "--options")
        step("kuwait.py")        # Boursa Kuwait prices + ratings (changes slowly)
    if "--alerts" not in a:
        if not step("fetch.py", "--prices"):
            say("price fetch failed; keeping the previous cache")
        step("build_workbook.py")
        step("logs.py")          # the three headline ledgers
        step("ladders.py")       # 18 probation rungs + sanity + excel
        step("golden_alert.py", "--scan")   # golden-zone entries -> Telegram + PDF
        # Google Sheets push is best-effort: no credentials means no push, and
        # that must never take the local workbook down with it
        if os.path.exists(os.path.join(HERE, "gcreds.json")):
            step("to_gsheet.py", *([] if "--daily" in a or "--once" in a
                                   else ["--fast"]))
        else:
            say("--- to_gsheet.py skipped (no gcreds.json; see SETUP.md)")
    step("alerts.py", "--check")
    say("done")


if __name__ == "__main__":
    main()
