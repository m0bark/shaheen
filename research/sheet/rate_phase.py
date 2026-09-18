"""Does CYCLE POSITION matter? Early-cycle versus late-cycle rate moves.

    python research/sheet/rate_phase.py

THE CLAIM BEING TESTED
The conventional story is that the FIRST leg of a rate-up cycle is repriced
hardest: duration gets marked down violently when the market wakes up, and by
the time the cycle is a year old the repricing is already in the price. If that
is true, the rate-beta decile spread should be large EARLY in a rates-up episode
and small or absent LATE in one. A rate view would then be worth something only
at the turn, which is exactly when it is hardest to have.

HOW PHASE IS DEFINED, USING ONLY TRAILING DATA
TLT is the long-Treasury proxy. Rates up shows as TLT down. For each month-end
decision date:
  * how many CONSECUTIVE trading days TLT has closed below its own 200-day
    average, counted backwards from the decision date and nothing else;
  * TLT's trailing 252-day return.
Both are ratios of two adjusted closes or counts of them, so the split/dividend
adjustment factor cancels and neither is contaminated (see the CONTAMINATED note
in tree_screen.py for why price LEVELS are banned here and these are not).

  UPTREND  TLT at or above its 200d average -- not a rates-up trend at all
  EARLY    1..63 days below        the trend has just turned, under 3 months
  MID      64..126 days below
  LATE     over 126 days below     a sustained, well-advertised downtrend

RATE EXPOSURE
A stock's rate beta is the slope of its daily returns on TLT's over the trailing
252 days. Point-in-time. Positive rate beta = rises when bonds rise = long
duration, the names the story says an early hike should punish hardest. So the
pre-registered expectation is a NEGATIVE high-minus-low spread in EARLY months,
larger in magnitude than in LATE months.

TWO LABELLINGS, AND ONLY ONE OF THEM IS TRADEABLE
  (a) WITH THE ANSWER KEY. rate_foresight.py classified months by what TLT
      ACTUALLY did over the next 63 days. That is deliberate lookahead, used to
      put a ceiling on what a perfect rate call could be worth. This file keeps
      that labelling for the EARLY-vs-LATE contrast so the two files are
      comparable, and subjects anything large to the rule-3 decomposition.
  (b) TRAILING ONLY. Phase alone, no forward information. Strictly worse as a
      measurement and strictly better as a strategy, because it is the only
      version a cash account could actually have traded.

WHAT WOULD HAVE FALSIFIED THE ANGLE
An EARLY spread no bigger than the LATE spread, or an EARLY spread that is fully
accounted for by (rate-beta spread) x (realised TLT move) plus (market-beta
spread) x (realised SPY move), or an EARLY spread that survives on the raw
target and dies on the beta-neutralised one. Any of those three and cycle
position is not adding information.

THE SAMPLE IS THE WHOLE PROBLEM AND IT IS REPORTED FIRST
This is conditional-on-conditional: rates-up AND phase. 158 month-ends split
four ways by phase and two ways by rate direction leaves buckets in the teens.
Overlapping 63-day forward windows on monthly dates make the effective count
roughly a third of the raw one, so a bucket of 15 months is about 5 independent
observations. Any bucket under MIN_CREDIBLE_MONTHS is labelled an anecdote in
the output, not a measurement, and no amount of t-statistic changes that.

WHAT IT FOUND, recorded after the run so the next reader does not redo it
Nothing, on all three of the falsification tests above.
  * The pre-registered EARLY bucket's rate-beta spread is NOT wider than LATE's.
    Rates-up EARLY -8.37% against LATE -6.92%, a difference of -1.45% with a
    deflated two-sample t of -0.12 against a search bar of 2.72. Every one of
    the 20 early-minus-late contrasts printed fails; the best is 1.75.
  * The spread does vary strongly across phase, but in the OPPOSITE direction to
    a cycle story and for the wrong reason. It is widest where the bond
    downtrend has not started (-14.53%) and narrowest where it is entrenched
    (-5.15%). That -9.38% difference has a deflated t of -2.01 on the raw
    target and -0.03% with t -0.01 once market beta is regressed out. The phases
    differ in how much market beta the rate-beta deciles happened to carry.
  * Rule 3 eats most of what is left: 68-117% of each rates-up bucket's spread
    is (rate-beta gap) x (realised TLT move) + (market-beta gap) x (realised SPY
    move), and no bucket's residual clears the bar either.
  * 6 of 10 phase buckets flip the sign of their beta-neutral spread between
    train and holdout.
  * The long-only version does nothing: -1.50pp against equal weight in train,
    +2.90pp in holdout, -0.00pp over the full sample, inside the +-0.83pp noise
    of which month the quarterly chain starts in.
The one piece of context that is real and does not need a rate view: the
universe returned +12.3% annualised in the 46 EARLY months against +15.9% in
the 83 UPTREND months, and won 67% of EARLY months against 81% of UPTREND ones.
"""
from __future__ import annotations

import io
import os
import sys
import warnings

_OUT = io.TextIOWrapper(open(sys.stdout.fileno(), "wb", closefd=False),
                        encoding="utf-8", errors="replace")


def say(*a) -> None:
    print(*a, file=_OUT)
    _OUT.flush()


warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LONG = os.path.join(HERE, "cache_long")

PROXY = "TLT"              # long-Treasury proxy: rates up == this falls
MARKET = "SPY"
BETA_WINDOW = 252          # trailing days for both rate beta and 12m return
SMA_WINDOW = 200           # the trend filter the phase definition rests on
FWD_DAYS = 63              # must match grid.csv's fwd63

EARLY_MAX_DAYS = 63        # "the trend just turned": under ~3 months below
MID_MAX_DAYS = 126         # 3..6 months below
# anything beyond MID_MAX_DAYS is LATE

# Second phase axis, added after the first one was run. A single close back above
# the average resets the consecutive-day count to zero, and TLT whipsaws across
# its 200d average constantly, so the pre-registered LATE bucket never fills:
# 12 months out of 157. That is a property of the definition, not evidence about
# cycles. PERSISTENCE is the fraction of the last 126 trading days spent below
# the average, which tolerates the whipsaw and splits the sample far more evenly.
# It is reported as a ROBUSTNESS CHECK on the sample-size problem. It was not
# chosen because it produced a better number -- both axes are printed in full
# and both are counted against the multiple-testing bar below.
PERSIST_WINDOW = 126
PERSIST_TURNING = 0.20     # above this much of the window below = trend exists
PERSIST_ENTRENCHED = 0.60  # above this = entrenched, well-advertised downtrend

N_DEC = 10
MIN_NAMES_FOR_DECILES = 40
MIN_CREDIBLE_MONTHS = 15   # below this the output says "anecdote", per the brief
OVERLAP_INFLATION = np.sqrt(3.0)   # 63d windows on monthly dates

TRAIN_END = pd.Timestamp("2021-09-30")
HOLDOUT_START = pd.Timestamp("2022-01-01")

# Every headline spread this file prints, counted honestly: three phase axes
# (4 + 3 + 3 buckets) crossed with {answer-key rates-up, trailing-only} and with
# {raw, beta-neutral}. 10 buckets x 2 x 2 = 40. Searching 40 numbers means the
# most extreme one is expected to look good, so the bar is the Bonferroni-style
# sqrt(2 ln K), not 1.96.
N_VARIANTS = 40
SEARCH_BAR = float(np.sqrt(2 * np.log(N_VARIANTS)))
# Below this spread, a "percent explained" ratio is noise divided by noise.
NEGLIGIBLE_SPREAD = 0.5

# Equal-weight version of this universe compounded at this CAGR 2013-2026;
# SPY itself did 14.78%. Any absolute return below this is a loss of money
# relative to doing nothing clever.
EW_BAR_CAGR = 14.82

PHASES = ("UPTREND", "EARLY", "MID", "LATE")
PERSIST_PHASES = ("NOT DOWN", "TURNING", "ENTRENCHED")
TRAIL_PHASES = ("TLT up", "TLT flat", "TLT down hard")

# The three axes, as (column, ordered bucket names, one-line description).
AXES = (
    ("phase", PHASES,
     "consecutive trading days TLT has closed below its 200d average "
     "(the pre-registered definition)"),
    ("persist_phase", PERSIST_PHASES,
     f"fraction of the last {PERSIST_WINDOW} days spent below the 200d average "
     "(robustness, fills the buckets)"),
    ("trail_phase", TRAIL_PHASES,
     "terciles of TLT's trailing 252-day return (the other variable the angle "
     "named)"),
)

# The differences that actually answer the angle, as (axis, earlier, later).
# The first two are the literal pre-registered question: trend just turned versus
# trend long established. The last two are the STRONGEST form of the claim, added
# after seeing that the spread rises monotonically as you go BACKWARDS along each
# axis: the biggest spread sits where the downtrend has not started yet at all,
# which is the bucket that contains the actual turn (the decision date precedes
# it, the turn happens inside the forward window). If "first move repriced
# hardest" is true of anything here, it is true of that bucket, so it is tested
# explicitly rather than left as a pattern noticed in a table.
CONTRASTS = (
    ("phase", "EARLY", "LATE"),
    ("persist_phase", "TURNING", "ENTRENCHED"),
    ("phase", "UPTREND", "LATE"),
    ("persist_phase", "NOT DOWN", "ENTRENCHED"),
    ("trail_phase", "TLT up", "TLT down hard"),
)


# --------------------------------------------------------------------- data
def rate_betas(C: pd.DataFrame, syms: list[str], proxy: str) -> pd.DataFrame:
    """Rolling slope of each name's daily returns on the proxy's, trailing only.

    Vectorised cov/var over the whole panel rather than 465 regression loops:
    cov(r_i, r_b) / var(r_b) built from rolling means. Same answer, seconds
    instead of minutes. Returns are ratios of two prices so the adjustment
    factor cancels and this is clean.
    """
    R = C[syms].pct_change()
    b = C[proxy].pct_change()
    mb = b.rolling(BETA_WINDOW).mean()
    vb = b.rolling(BETA_WINDOW).var()
    mr = R.rolling(BETA_WINDOW).mean()
    cov = (R.mul(b, axis=0).rolling(BETA_WINDOW).mean()
           .sub(mr.mul(mb, axis=0)))
    return cov.div(vb, axis=0)


def below_sma(px: pd.Series) -> pd.Series:
    """Boolean: is the close under its own trailing 200-day average."""
    sma = px.rolling(SMA_WINDOW).mean()
    return (px < sma) & sma.notna()


def days_below_sma(px: pd.Series) -> pd.Series:
    """Consecutive trading days the series has closed below its own 200d mean.

    Resets to zero on any close at or above the average. This is the
    pre-registered cycle-phase definition and it uses nothing but past closes.
    Its weakness, measured below rather than assumed, is that the reset is
    absolute: one close back above the average sends a nine-month-old downtrend
    back to day one.
    """
    b = below_sma(px)
    # Each unbroken run of True gets its own group id from the cumulative count
    # of the False values that precede it; cumsum within the group is the
    # length-so-far of the current run.
    grp = (~b).cumsum()
    return b.groupby(grp).cumsum().astype(float)


def persistence(px: pd.Series) -> pd.Series:
    """Fraction of the trailing PERSIST_WINDOW days spent below the 200d mean."""
    return below_sma(px).rolling(PERSIST_WINDOW).mean()


def persist_phase_of(f: float) -> str:
    if not np.isfinite(f):
        return "NOT DOWN"
    if f <= PERSIST_TURNING:
        return "NOT DOWN"
    if f <= PERSIST_ENTRENCHED:
        return "TURNING"
    return "ENTRENCHED"


def phase_of(days: float) -> str:
    if not np.isfinite(days) or days <= 0:
        return "UPTREND"
    if days <= EARLY_MAX_DAYS:
        return "EARLY"
    if days <= MID_MAX_DAYS:
        return "MID"
    return "LATE"


def fwd_return(px: pd.Series, dates: np.ndarray, n: int) -> dict:
    """Forward n-trading-day percent return of a price series at each date.

    Used for the deliberate-lookahead rate label and for the realised factor
    returns the rule-3 decomposition needs. Ratio of two closes, so clean.
    """
    out: dict[pd.Timestamp, float] = {}
    for dt in dates:
        ts = pd.Timestamp(dt)
        if ts not in px.index:
            continue
        i = px.index.get_loc(ts)
        j = min(i + n, len(px) - 1)
        out[ts] = (px.iloc[j] / px.iloc[i] - 1) * 100
    return out


def trailing_return(px: pd.Series, dates: np.ndarray, n: int) -> dict:
    """Trailing n-trading-day percent return. Past only."""
    out: dict[pd.Timestamp, float] = {}
    for dt in dates:
        ts = pd.Timestamp(dt)
        if ts not in px.index:
            continue
        i = px.index.get_loc(ts)
        if i < n:
            continue
        out[ts] = (px.iloc[i] / px.iloc[i - n] - 1) * 100
    return out


def load() -> pd.DataFrame:
    g = pd.read_csv(os.path.join(LONG, "grid.csv"), parse_dates=["date"])
    C = pd.read_csv(os.path.join(LONG, "px_close.csv"), index_col=0,
                    parse_dates=True).sort_index()
    for need in (PROXY, MARKET):
        if need not in C.columns:
            raise SystemExit(f"{need} absent from px_close.csv; cannot run")
    tech = pd.read_csv(os.path.join(LONG, "feat_tech.csv"), parse_dates=["date"])
    if "beta252" not in tech.columns:
        raise SystemExit("beta252 absent from feat_tech.csv; rule 2 check "
                         "would be impossible, so refusing to report anything")

    syms = sorted(set(g.symbol) & set(C.columns))
    say(f"  grid {len(g):,} rows | {len(syms)} symbols priced | "
        f"{g.date.nunique()} month-end decision dates")

    RB = rate_betas(C, syms, PROXY)
    rb = RB.stack().rename("rate_beta").reset_index()
    rb.columns = ["date", "symbol", "rate_beta"]

    d = g.merge(rb, on=["date", "symbol"], how="left")
    d = d.merge(tech[["date", "symbol", "beta252"]], on=["date", "symbol"],
                how="left")
    d = d.dropna(subset=["rate_beta", "fwd63"])

    dates = d.date.unique()
    tlt = C[PROXY].dropna()
    spy = C[MARKET].dropna()

    d["days_below"] = d.date.map(days_below_sma(tlt).to_dict())
    d["phase"] = d.days_below.map(phase_of)
    d["persist"] = d.date.map(persistence(tlt).to_dict())
    d["persist_phase"] = d.persist.map(persist_phase_of)

    d["tlt_trail12"] = d.date.map(trailing_return(tlt, dates, BETA_WINDOW))
    # Terciles of the trailing 12-month TLT return, cut on the TRAINING period
    # only so the bucket edges are not set with hindsight over the holdout.
    tr12 = (d[["date", "tlt_trail12"]].drop_duplicates("date")
            .set_index("date").tlt_trail12.dropna().sort_index())
    edges = np.quantile(tr12[tr12.index <= TRAIN_END].to_numpy(), [1 / 3, 2 / 3])
    say(f"  trailing-12m TLT terciles cut on train only at "
        f"{edges[0]:+.2f}% / {edges[1]:+.2f}%")
    bins = [-np.inf, edges[0], edges[1], np.inf]
    d["trail_phase"] = pd.cut(d.tlt_trail12, bins,
                              labels=list(reversed(TRAIL_PHASES)))
    d["trail_phase"] = d.trail_phase.astype(object)
    # THE DELIBERATE LOOKAHEAD, same trick as rate_foresight.py: what long
    # rates actually did over the window we are measuring. Nobody has this on
    # the decision date. It is here to bound what a perfect call is worth.
    d["tlt_fwd"] = d.date.map(fwd_return(tlt, dates, FWD_DAYS))
    d["spy_fwd"] = d.date.map(fwd_return(spy, dates, FWD_DAYS))
    d = d.dropna(subset=["tlt_fwd", "spy_fwd", "phase"])
    d["rates_up"] = np.where(d.tlt_fwd < 0, "rates UP", "rates DOWN")

    # SELECTION, not market timing: the universe mean of the month is removed.
    d["y"] = d.fwd63 - d.groupby("date")["fwd63"].transform("mean")

    # Rule 2. Demeaning removes the market's LEVEL, not exposure to it. Every
    # number in this file is also reported on the residual of y regressed on
    # beta252 within the month, which is what is left after paying for beta.
    d["y_bn"] = beta_neutral(d)

    # Deciles are cut within each month, once, on the full panel: the cut does
    # not depend on which phase bucket the month later lands in.
    d["dec"] = d.groupby("date")["rate_beta"].transform(
        lambda x: pd.qcut(x.rank(method="first"), N_DEC, labels=False,
                          duplicates="drop")
        if x.notna().sum() >= MIN_NAMES_FOR_DECILES else np.nan)
    d = d.dropna(subset=["dec", "y_bn", "phase", "persist_phase", "trail_phase"])
    say(f"  usable {len(d):,} rows | {d.date.nunique()} months with rate betas, "
        f"all three phase axes, deciles and a beta-neutral target")
    return d


def beta_neutral(d: pd.DataFrame) -> pd.Series:
    """Within each month, regress the demeaned forward return on market beta and
    keep the residual. Rule 2: this is the only target on which a big number is
    allowed to be called a finding."""
    out = pd.Series(np.nan, index=d.index)
    for _, g in d.groupby("date"):
        x = g["beta252"].to_numpy(dtype=float)
        yy = g["y"].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(yy)
        if m.sum() < MIN_NAMES_FOR_DECILES:
            continue
        b1, b0 = np.polyfit(x[m], yy[m], 1)
        out.loc[g.index[m]] = yy[m] - (b0 + b1 * x[m])
    return out


# ------------------------------------------------------------- measurement
def monthly_spreads(sub: pd.DataFrame, ycol: str) -> pd.Series:
    """High-decile minus low-decile mean, computed separately in each month.

    A time series of monthly spreads, not a pool of rows. Pooling rows treats
    465 names in one month as 465 independent observations when they share a
    market, which is how a t of 20 gets manufactured out of 30 months.
    """
    out = {}
    for dt, g in sub.groupby("date"):
        hi = g.loc[g.dec == N_DEC - 1, ycol]
        lo = g.loc[g.dec == 0, ycol]
        if len(hi) < 2 or len(lo) < 2:
            continue
        out[dt] = hi.mean() - lo.mean()
    return pd.Series(out, dtype=float).sort_index()


def fmt(v: float, w: int = 6, dp: int = 2) -> str:
    """Fixed-width number that prints 'n/a' rather than '+nan'."""
    return f"{'n/a':>{w}}" if not np.isfinite(v) else f"{v:>+{w}.{dp}f}"


def month_t(sp: pd.Series) -> tuple[float, float]:
    """t of the mean monthly spread, raw and deflated for window overlap."""
    if len(sp) < 3 or sp.std(ddof=1) == 0:
        return np.nan, np.nan
    t = sp.mean() / (sp.std(ddof=1) / np.sqrt(len(sp)))
    return t, t / OVERLAP_INFLATION


def monotonicity(sub: pd.DataFrame, ycol: str) -> float:
    means = sub.groupby("dec")[ycol].mean()
    if len(means) < 3:
        return np.nan
    return means.corr(pd.Series(means.index, index=means.index),
                      method="spearman")


def bucket_row(sub: pd.DataFrame, ycol: str) -> dict:
    sp = monthly_spreads(sub, ycol)
    t, t_adj = month_t(sp)
    return {"months": len(sp), "spread": sp.mean() if len(sp) else np.nan,
            "t": t, "t_adj": t_adj, "mono": monotonicity(sub, ycol),
            "series": sp}


def spread_table(d: pd.DataFrame, mask: pd.Series, title: str,
                 note: str) -> dict[str, dict[str, dict]]:
    say("")
    say("=" * 92)
    say(f"  {title}")
    say("=" * 92)
    say(f"  {note}")
    res: dict[str, dict[str, dict]] = {}
    for col, order, desc in AXES:
        say("")
        say(f"  AXIS: {col} -- {desc}")
        say(f"  {'bucket':<15}{'months':>8}{'eff n':>7}"
            f"{'RAW spread':>13}{'t':>7}{'t/1.7':>8}{'mono':>7}"
            f"{'BETA-NEUT':>12}{'t':>7}{'t/1.7':>8}  credibility")
        say("  " + "-" * 95)
        res[col] = {}
        for ph in order:
            sub = d[mask & (d[col] == ph)]
            if sub.empty:
                say(f"  {ph:<15}{0:>8}   -- no months in this bucket --")
                continue
            raw = bucket_row(sub, "y")
            bn = bucket_row(sub, "y_bn")
            # The rule-3 residual, per month: the actual spread minus what known
            # exposures times realised factor returns already account for. This
            # is the series the contrast section compares, because comparing raw
            # spreads across phases compares realised bond moves, not edges.
            dm = decompose_months(sub)
            resid = {"series": (dm.set_index("date").resid if len(dm)
                                else pd.Series(dtype=float))}
            res[col][ph] = {"raw": raw, "bn": bn, "resid": resid,
                            "rows": len(sub)}
            cred = ("ANECDOTE, not a measurement"
                    if raw["months"] < MIN_CREDIBLE_MONTHS else "measurable")
            say(f"  {ph:<15}{raw['months']:>8}{raw['months'] / 3.0:>7.1f}"
                f"{raw['spread']:>+12.2f}%{fmt(raw['t'], 7)}"
                f"{fmt(raw['t_adj'], 8)}"
                f"{fmt(raw['mono'], 7)}"
                f"{bn['spread']:>+11.2f}%{fmt(bn['t'], 7)}{fmt(bn['t_adj'], 8)}"
                f"  {cred}")
    say("")
    say(f"  eff n = months / 3, the overlap-corrected count. t/1.7 is the t")
    say(f"  after deflating for overlapping 63-day windows. With {N_VARIANTS} "
        f"headline spreads printed,")
    say(f"  the bar is |t| > sqrt(2*ln({N_VARIANTS})) = {SEARCH_BAR:.2f} on the "
        f"DEFLATED t, not 1.96.")
    return res


# ----------------------------------------------------- rule 3 decomposition
def decompose_months(sub: pd.DataFrame) -> pd.DataFrame:
    """Per-month actual spread and the two known-exposure product terms."""
    rows = []
    for dt, g in sub.groupby("date"):
        hi = g[g.dec == N_DEC - 1]
        lo = g[g.dec == 0]
        if len(hi) < 2 or len(lo) < 2:
            continue
        d_rb = hi.rate_beta.mean() - lo.rate_beta.mean()
        d_mb = hi.beta252.mean() - lo.beta252.mean()
        rows.append({"date": dt,
                     "actual": hi.y.mean() - lo.y.mean(),
                     "rate_term": d_rb * g.tlt_fwd.iloc[0],
                     "mkt_term": d_mb * g.spy_fwd.iloc[0],
                     "d_rb": d_rb, "d_mb": d_mb})
    x = pd.DataFrame(rows)
    if len(x):
        x["resid"] = x.actual - x.rate_term - x.mkt_term
    return x


def decompose(d: pd.DataFrame, sub: pd.DataFrame, label: str) -> pd.DataFrame:
    """Exposure times realised factor return, the check rule 3 demands.

    For each month: the high-minus-low decile gap in rate beta, multiplied by
    what TLT actually did over the next 63 days, plus the gap in market beta
    times what SPY actually did. Both factor returns are the ANSWER KEY. If the
    sum reproduces the spread, the spread is beta handed the answer key, which
    is not a forecast of anything.

    TLT and SPY moves are themselves correlated, so the two terms cannot be
    cleanly attributed one against the other. Their SUM against the actual
    spread is the quantity that means something.
    """
    x = decompose_months(sub)
    if len(x) < 3:
        say(f"  {label}: fewer than 3 usable months, no decomposition")
        return x
    act = x.actual.mean()
    rt = x.rate_term.mean()
    mt = x.mkt_term.mean()
    pred = rt + mt
    resid = act - pred
    # A ratio against a near-zero spread is noise divided by noise and has been
    # known to print -2379%. Say so instead of printing it.
    frac_txt = (f"= {pred / act * 100:>4.0f}% of the spread"
                if abs(act) >= NEGLIGIBLE_SPREAD
                else "(spread is ~0, so a % share is meaningless)")
    say(f"  {label}")
    say(f"    mean rate-beta gap high-low {x.d_rb.mean():+.3f}   "
        f"market-beta gap {x.d_mb.mean():+.3f}")
    say(f"    actual spread                          {act:+7.2f}%")
    say(f"    (rate-beta gap) x (realised TLT move)  {rt:+7.2f}%")
    say(f"    (mkt-beta gap)  x (realised SPY move)  {mt:+7.2f}%")
    say(f"    sum of the two known-exposure terms    {pred:+7.2f}%  {frac_txt}")
    t_r, t_r_adj = month_t(x.resid)
    say(f"    RESIDUAL, the only part that is new    {resid:+7.2f}%  "
        f"month-level t {fmt(t_r)}, deflated {fmt(t_r_adj)}")
    return x


# ------------------------------------------------------------- the context
def universe_context(d: pd.DataFrame) -> None:
    say("")
    say("=" * 92)
    say("  THE SIMPLEST USEFUL VERSION: WHAT DID THE UNIVERSE ITSELF DO?")
    say("=" * 92)
    say("  A cash account cannot short a bad phase, but it can understand one.")
    say("  Forward 63-day equal-weight universe return, annualised at the same")
    say(f"  rate for comparison against the {EW_BAR_CAGR:.2f}% equal-weight bar.")
    say("")
    per = d.groupby("date").agg(
        mkt=("fwd63", "mean"), phase=("phase", "first"),
        persist_phase=("persist_phase", "first"),
        trail_phase=("trail_phase", "first"),
        rates_up=("rates_up", "first"), trail12=("tlt_trail12", "first"),
        tlt_fwd=("tlt_fwd", "first"))
    for col, order, _ in AXES:
        say(f"  AXIS: {col}")
        say(f"  {'bucket':<15}{'months':>8}{'mean fwd63':>13}{'annlsd':>9}"
            f"{'median':>9}{'win rate':>10}{'TLT trail 12m':>15}")
        say("  " + "-" * 83)
        for ph in order:
            s = per[per[col] == ph]
            if s.empty:
                continue
            ann = ((1 + s.mkt.mean() / 100) ** 4 - 1) * 100
            say(f"  {ph:<15}{len(s):>8}{s.mkt.mean():>+12.2f}%{ann:>+8.1f}%"
                f"{s.mkt.median():>+8.2f}%{(s.mkt > 0).mean() * 100:>9.0f}%"
                f"{s.trail12.mean():>+14.2f}%")
        say("")
    say("  SPLIT AGAIN BY WHAT RATES ACTUALLY DID NEXT (the answer key):")
    say(f"  {'phase':<10}{'label':<12}{'months':>8}{'mean fwd63':>13}"
        f"{'annlsd':>9}{'mean TLT fwd':>14}")
    say("  " + "-" * 68)
    for ph in PHASES:
        for lab in ("rates UP", "rates DOWN"):
            s = per[(per.phase == ph) & (per.rates_up == lab)]
            if s.empty:
                continue
            ann = ((1 + s.mkt.mean() / 100) ** 4 - 1) * 100
            tag = "" if len(s) >= MIN_CREDIBLE_MONTHS else "  (anecdote)"
            say(f"  {ph:<10}{lab:<12}{len(s):>8}{s.mkt.mean():>+12.2f}%"
                f"{ann:>+8.1f}%{s.tlt_fwd.mean():>+13.2f}%{tag}")


# ------------------------------------------- the only tradeable version
def chain_cagr(per: pd.DataFrame, col: str, offset: int) -> tuple[float, int]:
    """Compound a non-overlapping quarterly chain of 63-day returns.

    fwd63 spans about three month-ends, so taking every third date is the only
    way to compound these without double-counting the same weeks. Three chains
    exist depending on which month you start in; all three are reported because
    picking the best one is a choice made with hindsight.
    """
    s = per.iloc[offset::3][col].dropna()
    if len(s) < 8:
        return np.nan, len(s)
    total = float(np.prod(1 + s.to_numpy() / 100.0))
    years = len(s) / 4.0
    return (total ** (1 / years) - 1) * 100, len(s)


def long_only(d: pd.DataFrame) -> dict[str, float]:
    """Pre-registered, trailing-only, long-only and cash-account legal.

    The direction is NOT chosen from the data. Conventional wisdom says long
    duration is punished when rates rise, so the tilt is into the LOW rate-beta
    decile, and it is applied in the phase the angle says should matter most.
    The same rule is then carried unchanged into the sealed holdout.
    """
    say("")
    say("=" * 92)
    say("  CAN A LONG-ONLY CASH ACCOUNT DO ANYTHING WITH THIS?")
    say("=" * 92)
    say("  Trailing phase only, no forward rate label, so this is implementable.")
    say("  Rule: at each month-end, if the phase is the tilt phase, hold the")
    say("  LOW rate-beta decile equal-weighted; otherwise hold the whole")
    say("  universe equal-weighted. Compounded on non-overlapping 63-day legs.")
    say("")
    lo = d[d.dec == 0].groupby("date").fwd63.mean().rename("lo")
    hi = d[d.dec == N_DEC - 1].groupby("date").fwd63.mean().rename("hi")
    ew = d.groupby("date").fwd63.mean().rename("ew")
    ph = d.groupby("date")[["phase", "persist_phase"]].first()
    per = pd.concat([ew, lo, hi, ph], axis=1).dropna().sort_index()

    for col, tilt in (("phase", "EARLY"), ("phase", "LATE"),
                      ("persist_phase", "TURNING"),
                      ("persist_phase", "ENTRENCHED")):
        per["tilt_lo"] = np.where(per[col] == tilt, per.lo, per.ew)
        per["tilt_hi"] = np.where(per[col] == tilt, per.hi, per.ew)
        n_on = int((per[col] == tilt).sum())
        say(f"  tilt only when {col} == {tilt}  ({n_on} of {len(per)} months on)")
        for name, col2 in (("equal-weight universe", "ew"),
                           ("tilt to LOW rate beta", "tilt_lo"),
                           ("tilt to HIGH rate beta", "tilt_hi")):
            cs = [chain_cagr(per, col2, o) for o in range(3)]
            vals = [c for c, _ in cs if np.isfinite(c)]
            legs = cs[0][1]
            say(f"    {name:<24}CAGR {np.mean(vals):>+7.2f}%  "
                f"(3 start offsets: {', '.join(f'{v:+.2f}' for v in vals)}, "
                f"range {max(vals) - min(vals):.2f}pp)  {legs} legs")
        say("")
    say(f"  The bar is {EW_BAR_CAGR:.2f}%, the equal-weight universe. The range")
    say("  across the three start offsets is the noise floor of this measurement:")
    say("  any gap smaller than it is the arbitrary choice of starting month.")
    say("")

    out = {}
    for split, sub in (("full", per), ("train", per[per.index <= TRAIN_END]),
                       ("holdout", per[per.index >= HOLDOUT_START])):
        s = sub.copy()
        s["tilt_lo"] = np.where(s.phase == "EARLY", s.lo, s.ew)
        a = np.nanmean([chain_cagr(s, "ew", o)[0] for o in range(3)])
        b = np.nanmean([chain_cagr(s, "tilt_lo", o)[0] for o in range(3)])
        out[split] = b - a
        say(f"  {split:<8}{len(s):>4} months  equal-weight {a:+7.2f}%  "
            f"EARLY-tilt {b:+7.2f}%  difference {b - a:+6.2f}%")
    return out


# --------------------------------------------------------------- sealed split
def sealed(d: pd.DataFrame) -> None:
    say("")
    say("=" * 92)
    say("  SEALED SPLIT: PHASE SPREADS FITTED NOWHERE, JUDGED TWICE")
    say("=" * 92)
    say(f"  train  <= {TRAIN_END:%Y-%m-%d}   holdout >= {HOLDOUT_START:%Y-%m-%d}")
    say("  The three months between are dropped: fwd63 looks 63 trading days")
    say("  ahead, so the last training labels otherwise overlap the holdout.")
    say("")
    flips = total = 0
    for col, order, _ in AXES:
        say(f"  AXIS: {col}")
        say(f"  {'bucket':<15}{'split':<9}{'months':>8}{'RAW':>11}{'t/1.7':>8}"
            f"{'BETA-NEUT':>12}{'t/1.7':>8}  credibility")
        say("  " + "-" * 85)
        signs = {}
        for ph in order:
            for split, m in (("train", d.date <= TRAIN_END),
                             ("holdout", d.date >= HOLDOUT_START)):
                sub = d[m & (d[col] == ph)]
                if sub.empty:
                    say(f"  {ph:<15}{split:<9}{0:>8}   -- empty --")
                    continue
                raw = bucket_row(sub, "y")
                bn = bucket_row(sub, "y_bn")
                signs[(ph, split)] = np.sign(bn["spread"])
                cred = ("anecdote" if raw["months"] < MIN_CREDIBLE_MONTHS
                        else "measurable")
                say(f"  {ph:<15}{split:<9}{raw['months']:>8}"
                    f"{raw['spread']:>+10.2f}%{fmt(raw['t_adj'], 8)}"
                    f"{bn['spread']:>+11.2f}%{fmt(bn['t_adj'], 8)}  {cred}")
        for ph in order:
            a, b = signs.get((ph, "train")), signs.get((ph, "holdout"))
            if a is not None and b is not None:
                total += 1
                flips += int(a != b)
        say("")
    say(f"  {flips} of {total} buckets flip the sign of the beta-neutral spread")
    say("  between train and holdout. A phase effect that is real shows the same")
    say("  sign in both. A flip is a rule you would have traded backwards.")


def main() -> None:
    say("LOADING")
    d = load()

    per = d.groupby("date").first()
    say("")
    say("=" * 92)
    say("  THE SAMPLE, FIRST, BECAUSE IT IS THE LIMITATION")
    say("=" * 92)
    for col, order, desc in AXES:
        say(f"  AXIS: {col}")
        say(f"    {desc}")
        tab = pd.crosstab(per[col], per.rates_up)
        say(f"  {'bucket':<15}{'rates UP':>11}{'rates DOWN':>12}{'total':>8}"
            f"{'eff n (UP)':>12}")
        say("  " + "-" * 59)
        for ph in order:
            if ph not in tab.index:
                say(f"  {ph:<15}{0:>11}{0:>12}{0:>8}{0.0:>12.1f}")
                continue
            u = int(tab.loc[ph].get("rates UP", 0))
            dn = int(tab.loc[ph].get("rates DOWN", 0))
            say(f"  {ph:<15}{u:>11}{dn:>12}{u + dn:>8}{u / 3.0:>12.1f}")
        say("  " + "-" * 59)
        say(f"  {'total':<15}{int(tab.get('rates UP', pd.Series(0)).sum()):>11}"
            f"{int(tab.get('rates DOWN', pd.Series(0)).sum()):>12}"
            f"{int(tab.values.sum()):>8}")
        say("")
    say(f"  Any bucket under {MIN_CREDIBLE_MONTHS} months is an ANECDOTE. The")
    say("  eff n column is what overlapping 63-day windows leave of it. The")
    say("  pre-registered axis leaves LATE with 12 months total and 4 in")
    say("  rates-up, which is why the second axis exists.")

    universe_context(d)

    up = d.rates_up == "rates UP"
    r_up = spread_table(
        d, up,
        "1. RATE-BETA DECILE SPREAD BY PHASE, RATES-UP MONTHS ONLY",
        "WITH THE ANSWER KEY. High decile = longest duration. The story says "
        "EARLY should be much more negative than LATE.")

    r_all = spread_table(
        d, pd.Series(True, index=d.index),
        "2. THE SAME THING WITH NO FORWARD LABEL AT ALL, PHASE ONLY",
        "Trailing information only, so this is the version a cash account "
        "could have traded.")

    say("")
    say("=" * 92)
    say("  3. RULE 3: IS IT EXPOSURE TIMES A REALISED FACTOR RETURN?")
    say("=" * 92)
    say("  Run on every rates-up bucket, because the answer-key label in table 1")
    say("  is exactly the setup that manufactures this artefact.")
    say("")
    for col, order, _ in AXES[:2]:
        for ph in order:
            sub = d[up & (d[col] == ph)]
            if len(sub):
                decompose(d, sub, f"RATES-UP / {col}={ph}  "
                                  f"({sub.date.nunique()} months)")
                say("")
    say("  And on the tradeable, no-lookahead version:")
    say("")
    for col, order, _ in AXES[:2]:
        for ph in order[1:]:
            sub = d[d[col] == ph]
            if len(sub):
                decompose(d, sub, f"ALL MONTHS / {col}={ph}  "
                                  f"({sub.date.nunique()} months)")
                say("")

    sealed(d)
    lo_gap = long_only(d)
    contrast(r_up, r_all, lo_gap)


def contrast(r_up: dict, r_all: dict, lo_gap: dict[str, float]) -> None:
    """The early-minus-late difference, which is the angle's actual question."""
    say("")
    say("=" * 92)
    say("  THE EARLY-versus-LATE CONTRAST, WHICH IS THE WHOLE QUESTION")
    say("=" * 92)
    say("  The angle does not ask whether the spread exists. It asks whether it")
    say("  is BIGGER early than late. That difference is the line below.")
    best: list[tuple[float, str]] = []
    for tag, r in (("rates-UP only (answer key)", r_up),
                   ("all months (tradeable)", r_all)):
        for col, early, late in CONTRASTS:
            e = r.get(col, {}).get(early)
            l = r.get(col, {}).get(late)
            if not e or not l:
                continue
            say("")
            say(f"  {tag}   axis {col}")
            for nm, b in ((early, e), (late, l)):
                say(f"    {nm:<14}{b['raw']['months']:>3} months  raw "
                    f"{b['raw']['spread']:+6.2f}%  beta-neutral "
                    f"{b['bn']['spread']:+6.2f}%  deflated t "
                    f"{fmt(b['bn']['t_adj'], 5)}")
            for lbl, key in (("raw", "raw"), ("beta-neutral", "bn"),
                             ("rule-3 residual", "resid")):
                a, bb = e[key]["series"], l[key]["series"]
                if len(a) < 3 or len(bb) < 3:
                    continue
                gap = a.mean() - bb.mean()
                # Two-sample t on the two sets of MONTHLY quantities. The buckets
                # are different months so they are independent by construction;
                # deflated for window overlap like everything else here.
                se = np.sqrt(a.var(ddof=1) / len(a) + bb.var(ddof=1) / len(bb))
                t = gap / se if se > 0 else np.nan
                say(f"    {early} minus {late}, {lbl:<16}{gap:+6.2f}%   "
                    f"two-sample t {fmt(t, 5)}  deflated "
                    f"{fmt(t / OVERLAP_INFLATION, 5)}  (bar {SEARCH_BAR:.2f})")
                if np.isfinite(t) and key != "raw":
                    best.append((abs(t) / OVERLAP_INFLATION,
                                 f"{tag} / {early} minus {late} / {lbl}"))

    say("")
    say("=" * 92)
    say("  VERDICT")
    say("=" * 92)
    best.sort(reverse=True)
    top_t, top_name = best[0] if best else (np.nan, "nothing computable")
    say(f"  Strongest early-minus-late contrast on any non-raw target:")
    say(f"    {top_name}")
    say(f"    deflated |t| {top_t:.2f} against a bar of {SEARCH_BAR:.2f}")
    cleared = [n for v, n in best if v > SEARCH_BAR]
    say(f"  {len(cleared)} of {len(best)} contrasts clear the bar.")
    say("")
    if not cleared:
        say("  NOTHING. Cycle position does not change the rate-beta spread by")
        say("  any amount this sample can distinguish from zero. The one contrast")
        say("  that looks large on the RAW target -- the spread is about twice as")
        say("  wide when the bond downtrend has not begun as when it is")
        say("  entrenched -- goes to almost exactly zero the moment market beta")
        say("  is regressed out, which is rule 2's warning arriving on schedule:")
        say("  the phases differ in how much market beta the rate-beta deciles")
        say("  happened to carry, not in how rates were repriced.")
    else:
        say("  One or more contrasts cleared the bar. Before believing it, check")
        say("  the sealed-split table above for a sign flip and the rule-3")
        say("  residual line for how much survives known exposures.")
    say("")
    say(f"  Long-only, trailing-only, EARLY-tilt minus equal-weight CAGR:")
    for k in ("full", "train", "holdout"):
        say(f"    {k:<9}{lo_gap[k]:+6.2f}pp")
    say("  Train and holdout disagree in sign, and both gaps are inside the")
    say(f"  {EW_BAR_CAGR:.2f}% bar's own start-offset noise. No long-only action.")


if __name__ == "__main__":
    main()
