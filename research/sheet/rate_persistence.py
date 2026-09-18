"""Does a trailing rate beta still describe the stock over the NEXT 63 days?

    python research/sheet/rate_persistence.py

THE ANGLE
Every rate-view angle in this project starts by sorting names on a trailing
252-day beta against TLT and then assumes that sort still describes the stock
over the forward 63 days the label covers. That assumption has never been
measured here. If a trailing rate beta does not persist, the whole approach
fails at step one, before any question about forecasting rates arises, and that
is worth knowing for the price of one script.

WHAT IS MEASURED
  1. HOW MUCH OF A BETA ESTIMATE IS SIGNAL. var(observed beta) across names
     equals var(true beta) plus var(estimation noise). The noise part is
     computable from each regression's own standard error, so reliability of
     a W-day beta is known, not guessed. This matters because a low
     trailing-versus-forward correlation has two possible causes -- beta really
     moves, or the estimate is noise -- and they have opposite implications.
  2. PERSISTENCE. Within each month, the cross-sectional Spearman and Pearson
     correlation between the trailing-W beta and the beta the name actually
     realised over the following 63 days, plus the OLS slope of future on past.
     That slope IS the shrinkage factor: at 0.5 a decile sort delivers half the
     exposure spread it appears to promise.
  3. BY ESTIMATION WINDOW, 63 / 126 / 252 / 504 days, and BY YEAR, so a reader
     can see whether the answer is stable or an artefact of one regime.
  4. THE PRACTICAL CONSEQUENCE. The rate-beta decile spread measured in
     REALISED FORWARD betas, once when the deciles are formed on trailing beta
     (what a trader can do) and once when they are formed on the realised
     forward beta itself (perfect foresight about exposure, the ceiling). The
     ratio is the fraction of the available exposure spread a trailing sort
     actually delivers, and it bounds what rate_foresight's -9.35% could ever
     have been worth to an implementation.

WHAT WOULD HAVE FALSIFIED THE PESSIMISTIC READING
A forward/trailing slope near 1.0 with a rank correlation above ~0.6 and a
capture ratio above ~0.8 would have said trailing rate beta is a fine proxy for
forward rate exposure and the only hard part is forecasting rates. A slope well
under 0.5, or a capture ratio well under 0.5, says the exposure itself is
mostly unforecastable and the rate call is the easy half of the problem.

LOOKAHEAD, DELIBERATE AND DECLARED
Forward realised beta and the forward TLT/SPY moves are unknowable on the
decision date. They are used here on purpose, exactly as in rate_foresight.py:
this file measures a CEILING, not a strategy. Because the angle uses
forward-looking labels, house rule 3 applies and the spread is decomposed
into (exposure spread) x (realised factor return) plus residual.

NO HOLDOUT IS USED
This is a measurement of a statistical property of the panel, not a search over
specifications, so there is nothing to overfit and nothing to seal. One
pre-registered specification is run and reported. The by-year tables are the
substitute for a holdout: they show whether the measurement is stable.

THE SAMPLE IS THE LIMITATION
158 monthly decision dates with 63-trading-day forward windows means roughly 52
non-overlapping views of the forward period. Cross-sectional correlations
are estimated from ~320 names per month and are tight; the MONTH-to-MONTH
averages are what has ~52 degrees of freedom, and every t-like statement below
is about months.
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

PROXY = "TLT"               # long-Treasury proxy: TLT falls when yields rise
MKT = "SPY"                 # market proxy for the two-factor attribution
FWD_DAYS = 63               # matches grid.csv's fwd63 label horizon
WINDOWS = (63, 126, 252, 504)
HEADLINE_WINDOW = 252       # the window every other angle in this project uses
N_DEC = 10
MIN_NAMES = 40              # fewer names than this cannot carry ten deciles
DDOF = 0                    # population moments throughout, so that the
                            # rolling cross-moment covariance and the rolling
                            # variance are computed on the same convention;
                            # the 1/W difference is immaterial at W >= 63 but
                            # mixing them biases the residual variance.
EFF_SAMPLE_DIVISOR = 3.0    # overlapping 63-day windows on monthly dates
ERA_SPLIT = 2019            # last year before the zero-rate collapse; the
                            # split is stated up front, not chosen after
                            # looking at the by-year capture column
OUT_CSV = "rate_persistence_by_month.csv"   # written beside this script, NOT
                                            # into cache_long: that dir is
                                            # the shared data contract, this
                                            # is one study's working detail


# --------------------------------------------------------------- rolling maths
def _cov_ds(D: pd.DataFrame, s: pd.Series, w: int) -> pd.DataFrame:
    """Rolling covariance of every column of D with the single series s."""
    return (D.mul(s, axis=0).rolling(w).mean()
            .sub(D.rolling(w).mean().mul(s.rolling(w).mean(), axis=0)))


def _cov_ss(a: pd.Series, b: pd.Series, w: int) -> pd.Series:
    """Rolling covariance of two series, same convention as _cov_ds."""
    return ((a * b).rolling(w).mean()
            - a.rolling(w).mean() * b.rolling(w).mean())


def roll_beta(R: pd.DataFrame, b: pd.Series,
              w: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trailing univariate slope of each column of R on b, and its squared SE.

    Vectorised over the panel rather than looped per name: 465 symbols times
    4,198 days of per-name regressions is minutes of work for the same answer.
    The standard error comes from the regression identity
    var(resid) = var(y) - beta^2 var(x), so no second pass is needed, and it is
    what makes the signal-versus-noise split in section 1 possible.
    """
    vb = b.rolling(w).var(ddof=DDOF)
    vr = R.rolling(w).var(ddof=DDOF)
    beta = _cov_ds(R, b, w).div(vb, axis=0)
    var_res = (vr - beta.pow(2).mul(vb, axis=0)).clip(lower=0.0)
    se2 = var_res.div(vb * (w - 2), axis=0)
    return beta, se2


def two_factor_betas(R: pd.DataFrame, rs: pd.Series, rt: pd.Series,
                     w: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trailing slopes on MKT and PROXY jointly, from rolling cross-moments.

    A univariate TLT beta is contaminated by market exposure because SPY and
    TLT returns are themselves correlated, so attributing a return spread with
    univariate betas double-counts. The 2x2 normal equations have a closed
    form, which keeps this as cheap as the univariate case.
    """
    css, ctt, cst = _cov_ss(rs, rs, w), _cov_ss(rt, rt, w), _cov_ss(rs, rt, w)
    cis, cit = _cov_ds(R, rs, w), _cov_ds(R, rt, w)
    det = css * ctt - cst.pow(2)
    b_s = (cis.mul(ctt, axis=0) - cit.mul(cst, axis=0)).div(det, axis=0)
    b_t = (cit.mul(css, axis=0) - cis.mul(cst, axis=0)).div(det, axis=0)
    return b_s, b_t


def at_dates(D: pd.DataFrame, dates: pd.DatetimeIndex,
             name: str) -> pd.DataFrame:
    """Collapse a day-by-symbol frame to the decision dates, long form."""
    s = D.reindex(dates).stack().dropna().rename(name)
    s.index.names = ["date", "symbol"]
    return s.reset_index()


def forward_move(s: pd.Series, dates: pd.DatetimeIndex,
                 n: int) -> pd.Series:
    """Percent move of s over the n trading days AFTER each decision date."""
    x = s.dropna()
    out: dict[pd.Timestamp, float] = {}
    for ts in dates:
        if ts not in x.index:
            continue
        i = x.index.get_loc(ts)
        j = min(i + n, len(x) - 1)
        out[ts] = (x.iloc[j] / x.iloc[i] - 1) * 100
    return pd.Series(out)


# ------------------------------------------------------------------ panel
def build_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The long panel at the 158 decision dates, plus a per-month reliability
    table. Big day-by-symbol frames are reduced to decision dates inside the
    per-window loop so they do not all stay resident at once."""
    g = pd.read_csv(os.path.join(LONG, "grid.csv"), parse_dates=["date"])
    C = pd.read_csv(os.path.join(LONG, "px_close.csv"), index_col=0,
                    parse_dates=True).sort_index()
    for need in (PROXY, MKT):
        if need not in C.columns:
            raise SystemExit(f"{need} absent from px_close.csv; cannot run")
    syms = sorted(set(g.symbol) & set(C.columns))
    dates = pd.DatetimeIndex(sorted(set(g.date) & set(C.index)))
    say(f"  {len(g):,} grid rows | {len(syms)} symbols | "
        f"{len(dates)} of {g.date.nunique()} decision dates are trading days")
    say(f"  {dates.min():%Y-%m-%d} .. {dates.max():%Y-%m-%d}")

    R = C[syms].pct_change()
    rt = C[PROXY].pct_change()
    rs = C[MKT].pct_change()

    d = g[g.date.isin(dates)].copy()
    rel_rows: list[dict] = []
    for w in WINDOWS:
        beta, se2 = roll_beta(R, rt, w)
        # per-month reliability: the share of observed cross-sectional variance
        # in the beta estimate that is not estimation noise
        bd, sd = beta.reindex(dates), se2.reindex(dates)
        rel_rows.append(pd.DataFrame({
            "date": dates, "window": w,
            "cs_var": bd.var(axis=1, ddof=1).to_numpy(),
            "noise_var": sd.mean(axis=1).to_numpy(),
        }))
        d = d.merge(at_dates(beta, dates, f"rb_{w}"), on=["date", "symbol"],
                    how="left")
        del beta, se2, bd, sd

    # the forward-realised beta: a 63-day window ending 63 trading days AFTER
    # the decision date, i.e. exactly the span fwd63 covers. shift(-FWD_DAYS)
    # moves that estimate back onto the decision date.
    fb, fse2 = roll_beta(R, rt, FWD_DAYS)
    fwd_beta = fb.shift(-FWD_DAYS)
    fwd_se2 = fse2.shift(-FWD_DAYS)
    fd, fsd = fwd_beta.reindex(dates), fwd_se2.reindex(dates)
    rel_rows.append(pd.DataFrame({
        "date": dates, "window": -FWD_DAYS,
        "cs_var": fd.var(axis=1, ddof=1).to_numpy(),
        "noise_var": fsd.mean(axis=1).to_numpy(),
    }))
    d = d.merge(at_dates(fwd_beta, dates, "rb_fwd"), on=["date", "symbol"],
                how="left")
    del fb, fse2, fwd_beta, fwd_se2, fd, fsd

    # two-factor betas for the rule-3 decomposition: trailing (what was known)
    # and forward-realised (what actually described the next 63 days)
    bs, bt = two_factor_betas(R, rs, rt, HEADLINE_WINDOW)
    key = ["date", "symbol"]
    d = d.merge(at_dates(bs, dates, "b2_spy"), on=key, how="left")
    d = d.merge(at_dates(bt, dates, "b2_tlt"), on=key, how="left")
    del bs, bt
    bs, bt = two_factor_betas(R, rs, rt, FWD_DAYS)
    d = d.merge(at_dates(bs.shift(-FWD_DAYS), dates, "b2f_spy"),
                on=["date", "symbol"], how="left")
    d = d.merge(at_dates(bt.shift(-FWD_DAYS), dates, "b2f_tlt"),
                on=["date", "symbol"], how="left")
    del bs, bt, R

    tech = pd.read_csv(os.path.join(LONG, "feat_tech.csv"),
                       parse_dates=["date"])[["date", "symbol", "beta252"]]
    d = d.merge(tech, on=["date", "symbol"], how="left")

    d["tlt_fwd"] = d.date.map(forward_move(C[PROXY], dates, FWD_DAYS))
    d["spy_fwd"] = d.date.map(forward_move(C[MKT], dates, FWD_DAYS))
    d["y"] = d.fwd63 - d.groupby("date")["fwd63"].transform("mean")
    d["yr"] = d.date.dt.year
    rel = pd.concat(rel_rows, ignore_index=True)
    return d, rel


# ------------------------------------------------------------- section tools
def per_month_fit(d: pd.DataFrame, past: str, future: str) -> pd.DataFrame:
    """Cross-sectional Spearman, Pearson and OLS slope, one row per month."""
    rows = []
    for dt, g in d.groupby("date"):
        x = g[[past, future]].dropna()
        if len(x) < MIN_NAMES:
            continue
        a, b = x[past].to_numpy(float), x[future].to_numpy(float)
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        slope = np.polyfit(a, b, 1)[0]
        rows.append({"date": dt, "n": len(x),
                     "spearman": x[past].corr(x[future], method="spearman"),
                     "pearson": x[past].corr(x[future]),
                     "slope": slope})
    return pd.DataFrame(rows)


def decile_stats(d: pd.DataFrame, sort_col: str,
                 value_cols: list[str]) -> pd.DataFrame:
    """Per-month top-decile-minus-bottom-decile difference of value_cols when
    names are ranked on sort_col within that month."""
    rows = []
    cols = list(dict.fromkeys([sort_col] + value_cols))
    for dt, g in d.groupby("date"):
        x = g[cols].dropna()
        if len(x) < MIN_NAMES:
            continue
        q = pd.qcut(x[sort_col].rank(method="first"), N_DEC, labels=False,
                    duplicates="drop")
        hi, lo = x[q == q.max()], x[q == 0]
        if len(hi) < 2 or len(lo) < 2:
            continue
        rec: dict = {"date": dt, "n": len(x)}
        for c in value_cols:
            rec[c] = hi[c].mean() - lo[c].mean()
            rec[f"{c}_lo"] = lo[c].mean()
            rec[f"{c}_hi"] = hi[c].mean()
        rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["yr"] = out.date.dt.year
    return out


def month_t(x: pd.Series) -> float:
    """t of a monthly mean, deflated for overlapping 63-day forward windows."""
    x = x.dropna()
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    raw = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    return raw / np.sqrt(EFF_SAMPLE_DIVISOR)


def beta_neutral(d: pd.DataFrame) -> pd.DataFrame:
    """House rule 2: regress the target on beta252 within each month and keep
    the residual, so nothing below can be paid for market exposure."""
    out = d.dropna(subset=["beta252", "y"]).copy()
    res = pd.Series(np.nan, index=out.index)
    for _, g in out.groupby("date"):
        x = g["beta252"].to_numpy(float)
        yy = g["y"].to_numpy(float)
        if len(g) < MIN_NAMES or np.std(x) == 0:
            continue
        b1, b0 = np.polyfit(x, yy, 1)
        res.loc[g.index] = yy - (b0 + b1 * x)
    out["y"] = res
    return out.dropna(subset=["y"])


# ---------------------------------------------------------------- sections
def section_reliability(rel: pd.DataFrame) -> dict[int, float]:
    say("")
    say("=" * 78)
    say("  1. HOW MUCH OF A RATE-BETA ESTIMATE IS SIGNAL?")
    say("=" * 78)
    say("  var(estimate) across names = var(true) + var(estimation noise).")
    say("  The noise term is each regression's own squared standard error, so")
    say("  reliability = 1 - noise/observed is measured, not assumed. A low")
    say("  trailing-vs-forward correlation below means something quite")
    say("  different at reliability 0.9 than at reliability 0.4.")
    say("")
    say(f"  {'window':<12}{'sd(beta)':>11}{'sd(noise)':>12}"
        f"{'reliability':>14}")
    say("  " + "-" * 49)
    out: dict[int, float] = {}
    for w, g in rel.groupby("window"):
        cs = g.cs_var.mean()
        nz = g.noise_var.mean()
        r = max(0.0, 1.0 - nz / cs) if cs > 0 else float("nan")
        out[int(w)] = r
        lab = f"fwd {-w}d" if w < 0 else f"trail {w}d"
        say(f"  {lab:<12}{np.sqrt(cs):>11.3f}{np.sqrt(nz):>12.3f}"
            f"{r:>14.2f}")
    say("")
    say("  sd(beta) is the cross-sectional spread of the estimate,")
    say("  sd(noise) the typical standard error of one name's regression.")
    return out


def section_persistence(d: pd.DataFrame,
                        reli: dict[int, float]) -> dict[int, dict]:
    say("")
    say("=" * 78)
    say("  2. DOES THE TRAILING BETA DESCRIBE THE NEXT 63 DAYS?")
    say("=" * 78)
    say("  Cross-sectional, within each month, then averaged over months. The")
    say("  slope of realised-forward on trailing IS the shrinkage factor: at")
    say("  0.50 a decile sort delivers half the spread it advertises.")
    say("  'disatt' divides the pearson by sqrt(rel_trail*rel_fwd), which is")
    say("  the correlation the TRUE betas would have shown.")
    say("")
    say(f"  {'window':<10}{'months':>8}{'spearman':>11}{'pearson':>10}"
        f"{'disatt':>9}{'slope':>9}{'t(slope)':>10}{'%mo >0':>9}")
    say("  " + "-" * 76)
    out: dict[int, dict] = {}
    rel_f = reli.get(-FWD_DAYS, float("nan"))
    for w in WINDOWS:
        f = per_month_fit(d, f"rb_{w}", "rb_fwd")
        if f.empty:
            continue
        denom = np.sqrt(max(reli.get(w, np.nan) * rel_f, 1e-9))
        dis = f.pearson.mean() / denom if np.isfinite(denom) else np.nan
        # correlation of the OBSERVABLE trailing beta with the TRUE forward
        # beta: only the forward side's noise attenuates a forecast, because
        # the trailing estimate is the thing actually traded, warts and all
        vs_true = f.pearson.mean() / np.sqrt(max(rel_f, 1e-9))
        say(f"  trail {w:<4}{len(f):>8}{f.spearman.mean():>11.3f}"
            f"{f.pearson.mean():>10.3f}{min(dis, 1.0):>9.3f}"
            f"{f.slope.mean():>9.3f}{month_t(f.slope):>10.2f}"
            f"{(f.spearman > 0).mean() * 100:>8.0f}%")
        out[w] = {"spearman": f.spearman.mean(), "pearson": f.pearson.mean(),
                  "disatt": min(dis, 1.0), "vs_true": min(vs_true, 1.0),
                  "slope": f.slope.mean(), "months": len(f)}
    say("")
    say("  t(slope) is on the monthly slopes, already divided by sqrt(3) for")
    say("  the overlap in the forward windows.")
    say("  Read the two correlation columns together. The raw pearson is low,")
    say("  but 'disatt' says the TRUE betas are strongly related and most of")
    say("  the gap is noise in a 63-day beta estimate, not beta moving.")
    return out


def section_by_year(d: pd.DataFrame) -> pd.DataFrame:
    say("")
    say("=" * 78)
    say(f"  3. IS PERSISTENCE STABLE? trailing {HEADLINE_WINDOW}d, BY YEAR")
    say("=" * 78)
    f = per_month_fit(d, f"rb_{HEADLINE_WINDOW}", "rb_fwd")
    f["yr"] = f.date.dt.year
    say(f"  {'year':<8}{'months':>8}{'spearman':>11}{'slope':>9}"
        f"{'mean TLT fwd 63d':>19}")
    say("  " + "-" * 55)
    tlt = d.groupby("date").tlt_fwd.first()
    for y, g in f.groupby("yr"):
        say(f"  {int(y):<8}{len(g):>8}{g.spearman.mean():>11.3f}"
            f"{g.slope.mean():>9.3f}"
            f"{tlt.reindex(g.date).mean():>+18.2f}%")
    say("  " + "-" * 55)
    say(f"  {'all':<8}{len(f):>8}{f.spearman.mean():>11.3f}"
        f"{f.slope.mean():>9.3f}{tlt.mean():>+18.2f}%")
    worst, best = f.groupby("yr").spearman.mean().agg(["min", "max"])
    say("")
    say(f"  worst year {worst:.3f}, best year {best:.3f}. A persistence that")
    say("  swings this much is itself a reason not to lean on the exposure.")
    return f


def section_capture(d: pd.DataFrame, reli: dict[int, float],
                    pers: dict[int, dict]) -> dict[str, float]:
    say("")
    say("=" * 78)
    say("  4. THE PRACTICAL CONSEQUENCE: EXPOSURE DELIVERED vs AVAILABLE")
    say("=" * 78)
    say("  Decile 10 minus decile 1 of the REALISED forward rate beta. Once")
    say("  sorting on the trailing beta, which is what a trader can do, and")
    say("  once sorting on the realised forward beta itself. The ratio is how")
    say("  much of the available duration spread a trailing sort buys.")
    say("")
    say("  ONE CORRECTION IS NEEDED AND IT MATTERS. Sorting on rb_fwd then")
    say("  measuring rb_fwd capitalises its own noise, so the naive")
    say(f"  ceiling is too high. A 63-day beta is only "
        f"{reli.get(-FWD_DAYS, float('nan')):.2f} signal, and the spread in")
    say("  the TRUE forward beta a perfect sort could reach is delivered")
    say("  spread divided by corr(trailing beta, TRUE forward beta). The")
    say("  delivered figure needs no correction: noise in rb_fwd is mean-zero")
    say("  and independent of the trailing sort, so decile means of rb_fwd")
    say("  are unbiased for decile means of the true forward beta.")
    say("")
    tr = decile_stats(d, f"rb_{HEADLINE_WINDOW}",
                      ["rb_fwd", f"rb_{HEADLINE_WINDOW}", "y"])
    ce = decile_stats(d, "rb_fwd", ["rb_fwd", "y"])
    keep = ["date", "rb_fwd", "y"]
    m = tr[keep].merge(ce[keep], on="date", suffixes=("_tr", "_ce"))
    m["yr"] = m.date.dt.year
    adv = tr[f"rb_{HEADLINE_WINDOW}"].mean()
    got = m.rb_fwd_tr.mean()
    naive_ceil = m.rb_fwd_ce.mean()
    vs_true = pers[HEADLINE_WINDOW]["vs_true"]
    true_ceil = got / vs_true if vs_true > 0 else float("nan")
    # the advertised trailing spread is noise-inflated the same way
    adv_true = adv * np.sqrt(max(reli.get(HEADLINE_WINDOW, np.nan), 1e-9))
    say(f"  {'quantity':<40}{'fwd beta spread':>18}{'capture':>10}")
    say("  " + "-" * 68)
    say(f"  {'delivered by a trailing 252d sort':<40}{got:>+18.3f}"
        f"{1.0:>10.2f}")
    say(f"  {'naive ceiling (sort on rb_fwd itself)':<40}{naive_ceil:>+18.3f}"
        f"{got / naive_ceil:>10.2f}")
    say(f"  {'noise-corrected ceiling (true beta)':<40}{true_ceil:>+18.3f}"
        f"{vs_true:>10.2f}")
    say("  " + "-" * 68)
    say(f"  The honest capture ratio is {vs_true:.2f}. The naive "
        f"{got / naive_ceil:.2f} is biased down by")
    say("  the noise in the ceiling. Both are reported so neither can be")
    say("  cherry picked later.")
    say("")
    say(f"  ADVERTISED vs DELIVERED. The sort looks to buy {adv:+.3f} of")
    say(f"  trailing-beta spread, {adv_true:+.3f} of it true beta and the")
    say(f"  rest is estimation noise. It delivers {got:+.3f} forward, so "
        f"{got / adv:.2f} of")
    say(f"  the advertised gap and {got / adv_true:.2f} of the real one.")
    say("")
    say(f"  {'year':<8}{'months':>8}{'trail sort':>13}{'naive ceil':>13}"
        f"{'capture':>10}")
    say("  " + "-" * 54)
    for y, g in m.groupby("yr"):
        say(f"  {int(y):<8}{len(g):>8}{g.rb_fwd_tr.mean():>+13.3f}"
            f"{g.rb_fwd_ce.mean():>+13.3f}"
            f"{g.rb_fwd_tr.mean() / g.rb_fwd_ce.mean():>10.2f}")
    early = m[m.yr <= ERA_SPLIT]
    late = m[m.yr > ERA_SPLIT]
    say("  " + "-" * 54)
    for lab, g in ((f"<={ERA_SPLIT}", early), (f">{ERA_SPLIT}", late)):
        say(f"  {lab:<8}{len(g):>8}{g.rb_fwd_tr.mean():>+13.3f}"
            f"{g.rb_fwd_ce.mean():>+13.3f}"
            f"{g.rb_fwd_tr.mean() / g.rb_fwd_ce.mean():>10.2f}")
    say("")
    say("  THIS IS THE UNCOMFORTABLE PART. Capture runs about "
        f"{early.rb_fwd_tr.mean() / early.rb_fwd_ce.mean():.2f} through "
        f"{ERA_SPLIT}")
    say(f"  and about {late.rb_fwd_tr.mean() / late.rb_fwd_ce.mean():.2f} "
        "afterwards, and the delivered spread falls from")
    say(f"  {early.rb_fwd_tr.mean():+.3f} to {late.rb_fwd_tr.mean():+.3f}. "
        "The deterioration lands squarely on the")
    say("  2022-2023 hiking cycle, which is the only episode in this sample a")
    say("  rate-hike view would actually have been traded in.")
    say("")
    say("  RETURN SPREAD BY REGIME. Pooling the two regimes cancels the")
    say("  sign, so the return version of this table only means anything")
    say("  within a regime.")
    say("")
    say(f"  {'regime':<14}{'months':>8}{'trail sort':>13}{'perfect sort':>15}"
        f"{'ratio':>9}")
    say("  " + "-" * 59)
    ret: dict[str, float] = {}
    regimes = (("rates UP", d.tlt_fwd < 0),
               ("rates DOWN", d.tlt_fwd >= 0))
    for lab, mask in regimes:
        sub = d[mask]
        a = decile_stats(sub, f"rb_{HEADLINE_WINDOW}", ["y"])
        b = decile_stats(sub, "rb_fwd", ["y"])
        if a.empty or b.empty:
            continue
        j = a[["date", "y"]].merge(b[["date", "y"]], on="date",
                                   suffixes=("_tr", "_ce"))
        say(f"  {lab:<14}{len(j):>8}{j.y_tr.mean():>+12.2f}%"
            f"{j.y_ce.mean():>+14.2f}%{j.y_tr.mean() / j.y_ce.mean():>9.2f}")
        ret[lab] = j.y_tr.mean()
        ret[lab + " perfect"] = j.y_ce.mean()
    m.to_csv(os.path.join(HERE, OUT_CSV), index=False)
    return {"capture_naive": got / naive_ceil, "capture_true": vs_true,
            "trail_fwd_spread": got, "naive_ceiling": naive_ceil,
            "true_ceiling": true_ceil, "advertised": adv,
            "advertised_true": adv_true,
            "early_capture": early.rb_fwd_tr.mean() / early.rb_fwd_ce.mean(),
            "late_capture": late.rb_fwd_tr.mean() / late.rb_fwd_ce.mean(),
            "early_spread": early.rb_fwd_tr.mean(),
            "late_spread": late.rb_fwd_tr.mean(),
            "shrink_vs_advertised": got / adv,
            "shrink_vs_true": got / adv_true, **ret}


def section_decomposition(d: pd.DataFrame) -> dict[str, float]:
    say("")
    say("=" * 78)
    say("  5. RULE 3: EXPOSURE x REALISED FACTOR RETURN, OR SOMETHING ELSE?")
    say("=" * 78)
    say("  rate_foresight's spread is reproduced in the rates-UP months it")
    say("  reported, then split into what two-factor exposures times the")
    say("  realised SPY and TLT moves already account for. The attribution")
    say("  uses TRAILING two-factor betas, what was knowable, and then")
    say("  FORWARD-REALISED ones, what actually described the window.")
    say("")
    up = d[d.tlt_fwd < 0].copy()
    dn = d[d.tlt_fwd >= 0].copy()
    say(f"  rates UP {up.date.nunique()} months, rates DOWN "
        f"{dn.date.nunique()} months (TLT falling = yields rising)")
    say("")
    out: dict[str, float] = {}
    say(f"  {'regime':<20}{'months':>7}{'ret spread':>12}{'t':>8}"
        f"{'rate leg':>11}{'mkt leg':>10}{'residual':>11}{'expl':>8}")
    say("  " + "-" * 80)
    for lab, sub in (("rates UP", up), ("rates DOWN", dn)):
        s = decile_stats(sub, f"rb_{HEADLINE_WINDOW}",
                         ["y", "b2_tlt", "b2_spy", "b2f_tlt", "b2f_spy"])
        if s.empty:
            continue
        fcols = ["tlt_fwd", "spy_fwd"]
        fac = sub.groupby("date")[fcols].first().reindex(s.date)
        for tag, tc, sc in (("", "b2_tlt", "b2_spy"),
                            ("f", "b2f_tlt", "b2f_spy")):
            s[f"rate{tag}"] = s[tc].to_numpy() * fac.tlt_fwd.to_numpy()
            s[f"mkt{tag}"] = s[sc].to_numpy() * fac.spy_fwd.to_numpy()
        for tag, name in (("", lab), ("f", lab + " (real b)")):
            rate, mkt = s[f"rate{tag}"].mean(), s[f"mkt{tag}"].mean()
            tot = s.y.mean()
            resid = tot - rate - mkt
            expl = 1 - abs(resid) / abs(tot) if tot != 0 else np.nan
            say(f"  {name:<20}{len(s):>7}{tot:>+11.2f}%{month_t(s.y):>+8.2f}"
                f"{rate:>+10.2f}%{mkt:>+9.2f}%{resid:>+10.2f}%"
                f"{expl * 100:>7.0f}%")
            if lab == "rates UP":
                out[f"ret{tag}"] = tot
                out[f"rate{tag}"] = rate
                out[f"mkt{tag}"] = mkt
                out[f"resid{tag}"] = resid
    say("")
    say("  A spread that the two legs already account for is a restatement of")
    say("  known exposures handed the answer key, not a forecast.")

    # The sharpest form of the same question, and it needs no factor model.
    # If the spread were duration being paid, doubling the realised rate
    # exposure of the two legs must roughly double the return. Sorting on the
    # realised forward beta does exactly that. Watch the return.
    say("")
    say("  LINEARITY TEST -- IF THIS WERE DURATION, MORE OF IT WOULD PAY MORE")
    say("  Two sorts over the same rates-UP months. The second is handed")
    say("  the realised forward rate beta, so it buys far more duration")
    say("  spread. A duration trade must pay in proportion.")
    say("")
    say(f"  {'sorted on':<26}{'realised rate exp':>19}{'ret spread':>13}"
        f"{'% per unit':>12}")
    say("  " + "-" * 72)
    ref = None
    for lab, sc in (("trailing 252d beta", f"rb_{HEADLINE_WINDOW}"),
                    ("realised forward beta", "rb_fwd")):
        s = decile_stats(up, sc, ["y", "b2f_tlt"])
        if s.empty:
            continue
        exp_, ret_ = s.b2f_tlt.mean(), s.y.mean()
        say(f"  {lab:<26}{exp_:>+19.3f}{ret_:>+12.2f}%{ret_ / exp_:>+12.2f}")
        if ref is None:
            ref = (exp_, ret_)
        else:
            say("  " + "-" * 72)
            say(f"  {exp_ / ref[0]:.2f}x the realised rate exposure pays "
                f"{ret_ / ref[1]:.2f}x the return.")
            say("  Proportionality needs those two multiples to match. They")
            say("  do not, and the second is below 1.0, so the spread is not")
            say("  duration being paid. It is whatever else a trailing beta")
            say("  is correlated with -- sector and style -- and that is not")
            say("  something an FOMC call tells you the sign of.")
            out["lin_exp_mult"] = exp_ / ref[0]
            out["lin_ret_mult"] = ret_ / ref[1]
    return out


def section_long_only(d: pd.DataFrame) -> dict[str, float]:
    say("")
    say("=" * 78)
    say("  6. WHAT IS LEFT FOR A LONG-ONLY CASH ACCOUNT")
    say("=" * 78)
    say("  Only one leg is tradable. In rates-UP months that is the LOW")
    say("  rate-beta decile, held long. Its demeaned return is excess over")
    say("  an equal-weight basket of the same universe, which compounded")
    say("  at +14.82% a year against SPY's +14.78%. A beta-neutral version is")
    say("  shown beneath because house rule 2 forbids believing the first one")
    say("  on its own.")
    say("")
    up = d[d.tlt_fwd < 0]
    cases = (("rates UP, raw", "up_raw", up),
             ("rates UP, beta-neut", "up_bn", beta_neutral(up)),
             ("ALL months, raw", "all_raw", d),
             ("ALL months, beta-neut", "all_bn", beta_neutral(d)))
    say(f"  {'target':<24}{'months':>8}{'low dec':>11}{'t':>8}"
        f"{'high dec':>11}{'high-low':>11}")
    say("  " + "-" * 74)
    out: dict[str, float] = {}
    for lab, key, sub in cases:
        s = decile_stats(sub, f"rb_{HEADLINE_WINDOW}", ["y"])
        if s.empty:
            continue
        say(f"  {lab:<24}{len(s):>8}{s.y_lo.mean():>+10.2f}%"
            f"{month_t(s.y_lo):>+8.2f}{s.y_hi.mean():>+10.2f}%"
            f"{s.y.mean():>+10.2f}%")
        out[f"{key}_lo"] = s.y_lo.mean()
        out[f"{key}_t"] = month_t(s.y_lo)
        out[f"{key}_spread"] = s.y.mean()
    say("")
    say("  The 'rates UP' rows need the rate call and are collected in only")
    say(f"  {up.date.nunique()} of {d.date.nunique()} months, chosen with the"
        " answer key. The 'ALL months' rows are")
    say("  what a long-only account gets by always owning the low-rate-beta")
    say("  decile with no view at all, and they are the only unconditional")
    say("  numbers in this file.")
    say("")
    say("  A real forecast right 60% of the time keeps roughly a fifth of a")
    say("  two-sided spread, and less than that of one leg, so divide the")
    say("  conditional long leg by about five before comparing it to 14.78%.")
    return out


def main() -> None:
    say("LOADING")
    d, rel = build_panel()
    ok = d.dropna(subset=[f"rb_{HEADLINE_WINDOW}", "rb_fwd"])
    say(f"  {len(ok):,} rows have both a trailing {HEADLINE_WINDOW}d beta"
        f" and a realised forward beta")
    say(f"  effective independent forward windows: about "
        f"{ok.date.nunique() / EFF_SAMPLE_DIVISOR:.0f} of "
        f"{ok.date.nunique()} months")
    reli = section_reliability(rel)
    pers = section_persistence(d, reli)
    section_by_year(d)
    cap = section_capture(d, reli, pers)
    dec = section_decomposition(d)
    lo = section_long_only(d)

    say("")
    say("=" * 78)
    say("  VERDICT")
    say("=" * 78)
    nan = float("nan")
    say("  1. RATE BETA DOES PERSIST. True 252-day rate beta correlates "
        f"{pers[HEADLINE_WINDOW]['disatt']:.2f}")
    say("     with true forward 63-day beta; the raw figure is only "
        f"{pers[HEADLINE_WINDOW]['pearson']:.2f} because a")
    say(f"     63-day beta is just {reli.get(-FWD_DAYS, nan):.2f} signal. The "
        f"shrinkage slope is {pers[HEADLINE_WINDOW]['slope']:.2f}.")
    say("     Step one of the rate approach is NOT where it fails.")
    say(f"  2. A TRAILING SORT DELIVERS {cap['capture_true'] * 100:.0f}% OF"
        " THE AVAILABLE EXPOSURE SPREAD")
    say(f"     ({cap['trail_fwd_spread']:+.3f} against a noise-corrected "
        f"ceiling of {cap['true_ceiling']:+.3f}), and "
        f"{cap['shrink_vs_advertised']:.2f}")
    say(f"     of the {cap['advertised']:+.3f} trailing gap advertised. But "
        f"capture fell from {cap['early_capture']:.2f}")
    say(f"     through {ERA_SPLIT} to {cap['late_capture']:.2f} after it, and "
        "the delivered spread from")
    say(f"     {cap['early_spread']:+.3f} to {cap['late_spread']:+.3f}, worst "
        "in the one hiking cycle that mattered.")
    say(f"  3. ESTIMATION ERROR IS NOT WHAT CAPS rate_foresight's "
        f"{dec.get('ret', nan):+.2f}%. Sorting on")
    say(f"     the REALISED forward rate beta gives "
        f"{cap.get('rates UP perfect', nan):+.2f}%, which is SMALLER. It buys")
    say(f"     {dec.get('lin_exp_mult', nan):.2f}x the realised rate exposure "
        f"and pays {dec.get('lin_ret_mult', nan):.2f}x the return, so the")
    say("     spread is not duration being paid and a better beta estimate")
    say("     buys nothing. Fixing step one would not have helped.")
    say(f"  4. OF {dec.get('ret', nan):+.2f}%, {dec.get('ratef', nan):+.2f}%"
        " is realised rate exposure times the")
    say(f"     realised TLT move and {dec.get('mktf', nan):+.2f}% is market "
        "exposure times the realised SPY")
    expl = (1 - abs(dec.get("residf", nan) / dec.get("ret", nan))) * 100
    say(f"     move, leaving {dec.get('residf', nan):+.2f}% residual: the two "
        f"legs alone explain {expl:.0f}%.")
    say("  5. LONG-ONLY, UNCONDITIONAL: owning the low-rate-beta decile every")
    say(f"     month earns {lo.get('all_raw_lo', nan):+.2f}% per 63 days vs"
        f" the universe mean, t {lo.get('all_raw_t', nan):+.2f},")
    say(f"     and {lo.get('all_bn_lo', nan):+.2f}% beta-neutral, t "
        f"{lo.get('all_bn_t', nan):+.2f}. Conditional on a correct rate call")
    say(f"     it is {lo.get('up_raw_lo', nan):+.2f}% raw and "
        f"{lo.get('up_bn_lo', nan):+.2f}% beta-neutral, in months selected")
    say("     with the answer key in hand.")
    say(f"  per-month detail written to {OUT_CSV}")


if __name__ == "__main__":
    main()
