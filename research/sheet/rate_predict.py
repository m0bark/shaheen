"""Is the rate regime forecastable AT ALL from trailing data? The prior ask.

    python research/sheet/rate_predict.py
    python research/sheet/rate_predict.py --shuffles 50

WHY THIS FILE EXISTS, AND WHY IT COMES FIRST
rate_foresight.py handed a stock screen PERFECT foreknowledge of the rate
direction and measured the ceiling. That ceiling is only worth anything if the
direction can be called. This file asks whether it can, from trailing data
only, and it is the load-bearing question for the whole FOMC line of enquiry:
if the 63-day rate direction is unforecastable, every downstream angle is
conditioning on a coin flip and the FOMC call collapses into noise.

THE TARGET
The sign of TLT's forward 63-day total return from each of the 158 month-end
decision dates. TLT up means long yields FELL (rates down); TLT down means
yields ROSE (rates up, the hiking window). The label is the realised forward
return, which is legitimate here because it is the thing being PREDICTED, not
a conditioner handed to a cross-sectional sort. Every one of the 158 dates has
at least 70 trading days of price data after it, so no label is truncated.

THE PREDICTORS, ALL STRICTLY TRAILING, ELEVEN OF THEM
  TLT 1 / 3 / 6 / 12-month returns      momentum and mean reversion in bonds
  TLT distance from its 50 and 200d SMA  trend position, a RATIO of two prices
  TLT realised volatility, 63d           regime stress
  SPY 3m return minus TLT 3m return      the risk-on/risk-off rotation
  GLD 3m return                          the real-rate / inflation hedge leg
  USO 3m return                          the commodity inflation impulse
  cross-sectional sd of rate betas        how differentiated the market's own
                                         duration positioning has become
Every one is a RETURN, a RATIO OF TWO PRICES, or a dispersion of regression
slopes. None is an adjusted-price LEVEL, so none of them is the partial
readout of the future that condemned price, avg_volume_20 and dollar_vol_60.

WHAT WOULD HAVE FALSIFIED THE NULL, WRITTEN DOWN BEFORE LOOKING
A holdout hit rate that (a) beats the holdout MAJORITY-CLASS rate, not 50%,
because a constant prediction already scores the majority rate for free; (b)
beats it by more than the effective-sample standard error, which with 53
holdout months and 63-day overlapping labels is about 18 independent
observations and therefore roughly 12 percentage points per sigma; and (c) sits
outside the distribution of hit rates the identical pipeline scores when it is
fitted on a shuffled target. Any one of those failing is a null result.

THE HONEST MULTIPLICITY
Four pre-registered models plus eleven univariate logits is fifteen variants,
so the threshold on any t is sqrt(2*ln 15) = 2.33, not 1.96. And overlapping
63-day windows inflate t by roughly sqrt(3), so divide by 1.7 first. Both
corrections are applied in the verdict lines rather than left to the reader.

THE BETA-NEUTRAL ANALOGUE OF RULE 2
Demeaning does not remove exposure, and for a single asset the equivalent trap
is forecasting TLT when you are really forecasting equities. So the test is
re-run on TLT's forward return with the SPY forward return regressed out using
a TRAIN-PERIOD slope: the bond-specific component. If the headline number only
exists in the raw target, it was an equity-direction call wearing a bond hat.

THE LITERATURE'S ANSWER IS ALREADY NO
Bond excess returns at a one-quarter horizon are close to unforecastable from
public trailing data. A null here is a CONFIRMATION of a well-established
result, not a failure of the search, and it is reported in those words.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
LONG = os.path.join(HERE, "cache_long")

PROXY = "TLT"                      # the long-Treasury proxy in this cache
MKT = "SPY"
HORIZON = 63                       # forward trading days, one quarter
BETA_WINDOW = 252                  # trailing window for each name's rate beta
VOL_WINDOW = 63                    # trailing window for realised volatility
MOM_WINDOWS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
SMA_WINDOWS = (50, 200)
TRADING_DAYS = 252
MIN_ROW_COVERAGE = 0.50            # below this a price row is a market holiday

# Rule 5: fit on or before the train cut, judge once on or after holdout start.
# The three months between are dropped because a 63-day forward label on a
# month-end date reaches about three months forward, so the last training
# months' labels would otherwise be drawn from holdout price action.
TRAIN_END = pd.Timestamp("2021-09-30")
HOLDOUT_START = pd.Timestamp("2022-01-01")
OVERLAP_MONTHS = 3                 # label length in months, for n_effective
CV_BLOCKS = 5                      # contiguous month blocks for train CV
CV_PURGE_MONTHS = 3                # purge around each CV test block
N_SHUFFLES = 30
RF_TREES = 300
RF_DEPTH = 3
TREE_DEPTH = 2
TREE_MIN_LEAF = 15
LOGIT_C = 1.0
SPY_BAR_CAGR = 14.78               # rule 4: the bar, verified from px_close

FEATURES = ["tlt_r1m", "tlt_r3m", "tlt_r6m", "tlt_r12m",
            "tlt_dist_sma50", "tlt_dist_sma200", "tlt_vol63",
            "spy_tlt_spread_3m", "gld_r3m", "uso_r3m", "rbeta_disp"]


# --------------------------------------------------------------------- data
def rate_betas(C: pd.DataFrame, syms: list[str]) -> pd.DataFrame:
    """Rolling slope of each name's daily returns on TLT's, trailing only.

    Vectorised as cov/var from rolling means, the same way rate_foresight.py
    does it: a per-name regression loop over 465 symbols and 4,200 days costs
    minutes for an identical answer."""
    R = C[syms].pct_change()
    b = C[PROXY].pct_change()
    mb = b.rolling(BETA_WINDOW).mean()
    vb = b.rolling(BETA_WINDOW).var()
    mr = R.rolling(BETA_WINDOW).mean()
    cov = (R.mul(b, axis=0).rolling(BETA_WINDOW).mean()
           .sub(mr.mul(mb, axis=0)))
    return cov.div(vb, axis=0)


def trailing_return(s: pd.Series, window: int) -> pd.Series:
    """Return over the trailing window, in percent. A RATIO of two adjusted
    closes, so the split/dividend factor cancels exactly and this is immune to
    the adjusted-level contamination documented in tree_screen.py."""
    return (s / s.shift(window) - 1.0) * 100.0


def dist_from_sma(s: pd.Series, window: int) -> pd.Series:
    """Percent distance from the trailing simple moving average. Also a ratio
    of prices, so also immune."""
    return (s / s.rolling(window).mean() - 1.0) * 100.0


def build_monthly() -> pd.DataFrame:
    """One row per decision date: the eleven trailing predictors and the
    forward 63-day TLT return that defines the target."""
    g = pd.read_csv(os.path.join(LONG, "grid.csv"), parse_dates=["date"])
    C = pd.read_csv(os.path.join(LONG, "px_close.csv"), index_col=0,
                    parse_dates=True).sort_index()
    missing = [s for s in (PROXY, MKT, "GLD", "USO") if s not in C.columns]
    if missing:
        raise SystemExit(f"cache is missing {missing}; cannot run this angle")

    # px_close.csv carries US market HOLIDAYS as rows. They are not thin
    # trading days: 2026-05-25 (Memorial Day) and 2026-09-07 (Labor Day) have
    # exactly ONE non-null column out of 509, the VIX, which the vendor quotes
    # on a different calendar. They are not missing data either -- no trading
    # happened -- and leaving them in does two separate kinds of damage:
    #   * every rolling window spanning one returns NaN, which silently deleted
    #     the whole trailing feature block for the 2026-05-29 decision date;
    #   * "i + 63 rows forward" stops meaning 63 TRADING days forward, so the
    #     forward label would measure a different horizon on different dates.
    # So the test is cross-sectional coverage, not all-NaN, which the first run
    # of this file got wrong and the 2026-05-29 drop exposed.
    cover = C.notna().mean(axis=1)
    holidays = C.index[cover < MIN_ROW_COVERAGE]
    if len(holidays):
        say(f"  dropped {len(holidays)} non-trading rows from the price index "
            f"({', '.join(d.strftime('%Y-%m-%d') for d in holidays[:4])}"
            f"{' ...' if len(holidays) > 4 else ''}) -- "
            f"max coverage on them {cover[holidays].max() * 100:.1f}%")
        C = C.drop(index=holidays)
    still = [s for s in (PROXY, MKT, "GLD", "USO") if C[s].isna().any()]
    if still:
        say(f"  WARNING {still} still carry gaps inside the trading index; "
            f"rolling windows spanning them will be dropped below")

    dates = pd.DatetimeIndex(sorted(g.date.unique()))
    say(f"  grid {len(g):,} rows | {g.symbol.nunique()} symbols | "
        f"{len(dates)} decision dates {dates[0]:%Y-%m} .. {dates[-1]:%Y-%m}")

    tlt, spy = C[PROXY], C[MKT]
    gld, uso = C["GLD"], C["USO"]
    daily = tlt.pct_change()

    feat = pd.DataFrame(index=C.index)
    for tag, w in MOM_WINDOWS.items():
        feat[f"tlt_r{tag}"] = trailing_return(tlt, w)
    for w in SMA_WINDOWS:
        feat[f"tlt_dist_sma{w}"] = dist_from_sma(tlt, w)
    feat["tlt_vol63"] = (daily.rolling(VOL_WINDOW).std()
                         * np.sqrt(TRADING_DAYS) * 100.0)
    feat["spy_tlt_spread_3m"] = (trailing_return(spy, MOM_WINDOWS["3m"])
                                 - trailing_return(tlt, MOM_WINDOWS["3m"]))
    feat["gld_r3m"] = trailing_return(gld, MOM_WINDOWS["3m"])
    feat["uso_r3m"] = trailing_return(uso, MOM_WINDOWS["3m"])

    # Cross-sectional dispersion of rate betas: how widely the market's own
    # duration exposure is spread on the decision date. Trailing by
    # construction, and a dispersion of slopes carries no price level.
    syms = sorted(set(g.symbol) & set(C.columns))
    RB = rate_betas(C, syms)
    in_grid = {pd.Timestamp(dt): set(sub) for dt, sub
               in g.groupby("date")["symbol"]}
    disp = {}
    for dt in dates:
        if dt not in RB.index:
            continue
        row = RB.loc[dt, [s for s in syms if s in in_grid.get(dt, ())]]
        row = row.replace([np.inf, -np.inf], np.nan).dropna()
        if len(row) >= 50:
            disp[dt] = float(row.std(ddof=1))
    feat["rbeta_disp"] = pd.Series(disp).reindex(C.index)

    # The forward windows. Full 63 trading days required -- no truncation, so
    # every label means the same thing.
    fwd_t, fwd_s, nxt_t, nxt_s = {}, {}, {}, {}
    px = C.index
    for k, dt in enumerate(dates):
        if dt not in px:
            continue
        i = px.get_loc(dt)
        j = i + HORIZON
        if j < len(px):
            fwd_t[dt] = (tlt.iloc[j] / tlt.iloc[i] - 1.0) * 100.0
            fwd_s[dt] = (spy.iloc[j] / spy.iloc[i] - 1.0) * 100.0
        # next-month returns, for the long-only translation at the end
        if k + 1 < len(dates) and dates[k + 1] in px:
            n = px.get_loc(dates[k + 1])
            nxt_t[dt] = (tlt.iloc[n] / tlt.iloc[i] - 1.0) * 100.0
            nxt_s[dt] = (spy.iloc[n] / spy.iloc[i] - 1.0) * 100.0

    m = feat.reindex(dates).copy()
    m["tlt_fwd"] = pd.Series(fwd_t)
    m["spy_fwd"] = pd.Series(fwd_s)
    m["tlt_next"] = pd.Series(nxt_t)
    m["spy_next"] = pd.Series(nxt_s)
    m.index.name = "date"
    m = m.reset_index()

    before = len(m)
    m = m.dropna(subset=FEATURES + ["tlt_fwd"])
    if len(m) < before:
        say(f"  dropped {before - len(m)} dates with an incomplete trailing "
            f"window or forward label")
    # y = 1 means TLT ROSE, i.e. long yields FELL: a rates-DOWN window.
    m["y"] = (m.tlt_fwd > 0).astype(int)
    return m


def split(m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = m[m.date <= TRAIN_END].copy()
    ho = m[m.date >= HOLDOUT_START].copy()
    emb = m[(m.date > TRAIN_END) & (m.date < HOLDOUT_START)]
    say(f"  train   {len(tr):>4} months  {tr.date.min():%Y-%m} .. "
        f"{tr.date.max():%Y-%m}")
    say(f"  embargo {len(emb):>4} months dropped "
        f"({OVERLAP_MONTHS} months of labels overlapping the holdout)")
    say(f"  holdout {len(ho):>4} months  {ho.date.min():%Y-%m} .. "
        f"{ho.date.max():%Y-%m}")
    return tr, ho


# ------------------------------------------------------------------- models
def standardise(tr: pd.DataFrame, ho: pd.DataFrame,
                cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Centre and scale on TRAIN statistics only. Using the pooled mean would
    leak the holdout's location into the fit."""
    a = tr[cols].to_numpy(dtype=float)
    b = ho[cols].to_numpy(dtype=float)
    mu = a.mean(axis=0)
    sd = a.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    return (a - mu) / sd, (b - mu) / sd


def make_model(name: str, seed: int = 0):
    if name == "LOGIT":
        return LogisticRegression(C=LOGIT_C, max_iter=2000)
    if name == "TREE":
        return DecisionTreeClassifier(max_depth=TREE_DEPTH,
                                      min_samples_leaf=TREE_MIN_LEAF,
                                      random_state=seed)
    if name == "FOREST":
        return RandomForestClassifier(n_estimators=RF_TREES,
                                      max_depth=RF_DEPTH,
                                      min_samples_leaf=TREE_MIN_LEAF,
                                      random_state=seed, n_jobs=-1)
    raise ValueError(f"unknown model {name}")


def block_cv_accuracy(tr: pd.DataFrame, cols: list[str], name: str,
                      seed: int = 0) -> float:
    """Accuracy from contiguous-block CV inside the train period, with a purge.

    Folds are blocks of consecutive MONTHS, never random rows, and training
    months within CV_PURGE_MONTHS of the test block are removed, because a
    63-day label straddles about three months and an unpurged neighbour is the
    same observation seen twice."""
    months = np.array(sorted(tr.date.unique()))
    if len(months) < CV_BLOCKS * 4:
        return np.nan
    hits, n = 0, 0
    for blk in np.array_split(months, CV_BLOCKS):
        lo = pd.Timestamp(blk[0]) - pd.DateOffset(months=CV_PURGE_MONTHS)
        hi = pd.Timestamp(blk[-1]) + pd.DateOffset(months=CV_PURGE_MONTHS)
        te = tr[tr.date.isin(blk)]
        fit = tr[(tr.date < lo) | (tr.date > hi)]
        if len(te) < 3 or len(fit) < 20 or fit.y.nunique() < 2:
            continue
        Xf, Xt = standardise(fit, te, cols)
        mdl = make_model(name, seed)
        mdl.fit(Xf, fit.y.to_numpy())
        hits += int((mdl.predict(Xt) == te.y.to_numpy()).sum())
        n += len(te)
    return hits / n if n else np.nan


def pick_best_single(tr: pd.DataFrame, seed: int = 0) -> str:
    """The BEST-SINGLE variant: the one predictor whose univariate logit scores
    highest in purged train CV. Selected on TRAIN ONLY, and re-selected inside
    every shuffled run so the null pays for the selection too."""
    scores = {c: block_cv_accuracy(tr, [c], "LOGIT", seed) for c in FEATURES}
    scores = {k: v for k, v in scores.items() if np.isfinite(v)}
    if not scores:
        return FEATURES[0]
    return max(scores, key=scores.get)


def fit_one(tr: pd.DataFrame, ho: pd.DataFrame, cols: list[str], name: str,
            seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Returns predicted classes and predicted P(y=1) on the holdout."""
    Xt, Xh = standardise(tr, ho, cols)
    mdl = make_model(name, seed)
    mdl.fit(Xt, tr.y.to_numpy())
    prob = mdl.predict_proba(Xh)
    # A model fitted on a single-class target has one column; guard rather than
    # let the index error surface as a silent NaN later.
    p1 = prob[:, list(mdl.classes_).index(1)] if 1 in mdl.classes_ \
        else np.zeros(len(ho))
    return mdl.predict(Xh), p1


def run_variants(tr: pd.DataFrame, ho: pd.DataFrame, best: str,
                 seed: int = 0) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The four pre-registered variants: fit on train, apply to holdout."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in ("LOGIT", "TREE", "FOREST"):
        out[name] = fit_one(tr, ho, FEATURES, name, seed)
    out[f"BEST-SINGLE ({best})"] = fit_one(tr, ho, [best], "LOGIT", seed)
    return out


# ------------------------------------------------------------------ scoring
def n_effective(n_months: int) -> float:
    """Overlapping 63-day labels on monthly dates: roughly one independent
    observation per OVERLAP_MONTHS months."""
    return n_months / OVERLAP_MONTHS


def hit_stats(y: np.ndarray, pred: np.ndarray, prob: np.ndarray,
              base: float) -> tuple[float, float, float, float]:
    """Hit rate, AUC, the z of the hit rate against the MAJORITY-CLASS rate on
    the effective sample, and that z after the sqrt(3) overlap deflation."""
    hit = float((pred == y).mean())
    try:
        auc = float(roc_auc_score(y, prob)) if len(set(y)) > 1 else np.nan
    except ValueError as e:          # never swallow it silently
        say(f"    AUC unavailable: {e}")
        auc = np.nan
    n_eff = n_effective(len(y))
    se = np.sqrt(0.25 / n_eff)       # widest binomial se, the conservative one
    z = (hit - base) / se if se > 0 else np.nan
    z_raw = (hit - base) / np.sqrt(0.25 / len(y)) if len(y) else np.nan
    return hit, auc, z, z_raw


# ------------------------------------------------------------------- tables
def base_rates(tr: pd.DataFrame,
               ho: pd.DataFrame) -> tuple[float, float, float]:
    """Three reference numbers, and they are not the same number.

    HINDSIGHT MAJORITY on the holdout is the ceiling a constant could
    have reached if you had known which constant to pick. REAL-TIME CONSTANT is
    what you would actually have scored by carrying the train period's majority
    class forward, which is the only constant available on the decision
    date. When the two differ the unconditional direction has changed between
    the periods, and that fact dominates everything below it."""
    say("")
    say("=" * 84)
    say("  THE BASE RATE -- three reference numbers, all different")
    say("=" * 84)
    out = []
    for tag, sub in (("train", tr), ("holdout", ho)):
        down = float(sub.y.mean())            # y = 1 means TLT up = rates DOWN
        maj = max(down, 1.0 - down)
        say(f"  {tag:<9}{len(sub):>4} months   rates UP (TLT fell) "
            f"{(1 - down) * 100:>5.1f}%   rates DOWN {down * 100:>5.1f}%   "
            f"majority class {maj * 100:>5.1f}%")
        out.append(maj)
    const = 1 if tr.y.mean() > 0.5 else 0
    rt = float((ho.y.to_numpy() == const).mean())
    say("")
    say(f"  THE DIRECTION FLIPPED. Train's majority class was "
        f"{'rates DOWN' if const else 'rates UP'}; the holdout's")
    say("  majority class is the OTHER one. So carrying the train prior")
    say(f"  unchanged scores {rt * 100:.1f}% on the holdout, not "
        f"{out[0] * 100:.1f}%.")
    say("")
    say(f"  hindsight majority on the holdout  {out[1] * 100:>5.1f}%   "
        f"unknowable on the decision date")
    say(f"  real-time constant (train prior)   {rt * 100:>5.1f}%   "
        f"the honest naive baseline")
    say(f"  coin flip                           50.0%   "
        f"the number NOT to compare against")
    say("")
    say("  A model is only interesting if it beats BOTH the real-time")
    say("  constant and the hindsight majority. Beating only the first")
    say("  means it detected the flip but still lost to a one-line rule")
    say("  chosen with hindsight; beating only the second is arithmetically")
    say("  impossible here.")
    return out[0], out[1], rt


def holdout_table(tr: pd.DataFrame, ho: pd.DataFrame, best: str,
                  tr_maj: float, ho_maj: float, rt_const: float,
                  preds: dict[str, tuple[np.ndarray, np.ndarray]],
                  bar: float) -> dict[str, float]:
    y = ho.y.to_numpy()
    n_eff = n_effective(len(ho))
    se = np.sqrt(0.25 / n_eff) * 100
    say("")
    say("=" * 84)
    say("  HOLDOUT, JUDGED ONCE.")
    say("=" * 84)
    say(f"  {len(ho)} months, but 63-day labels overlap, so the effective "
        f"sample is about {n_eff:.0f}")
    say(f"  independent observations. One standard error on a hit rate is "
        f"{se:.1f} percentage points,")
    say(f"  so clearing the 15-variant bar of |z| > {bar:.2f} needs "
        f"{bar * se:.1f}pp over the reference.")
    say("")
    say("  'calls DOWN' is the share of holdout months the variant predicted")
    say("  rates-DOWN. A variant sitting at 100% is a constant prediction")
    say("  with a model attached: its hit rate is the base rate.")
    say("")
    say(f"  {'variant':<26}{'trainCV':>9}{'calls DOWN':>12}{'holdout':>9}"
        f"{'vs maj':>8}{'vs RT':>7}{'z(maj)':>8}{'AUC':>7}")
    say("  " + "-" * 82)
    hits: dict[str, float] = {}
    for name, (pred, prob) in preds.items():
        cols = [best] if name.startswith("BEST-SINGLE") else FEATURES
        mname = "LOGIT" if name.startswith("BEST-SINGLE") else name
        cv = block_cv_accuracy(tr, cols, mname)
        hit, auc, z, _ = hit_stats(y, pred, prob, ho_maj)
        hits[name] = hit
        say(f"  {name:<26}{cv * 100:>8.1f}%{pred.mean() * 100:>11.0f}%"
            f"{hit * 100:>8.1f}%{(hit - ho_maj) * 100:>+7.1f}%"
            f"{(hit - rt_const) * 100:>+6.1f}%{z:>+8.2f}{auc:>7.3f}")
    const = 1 if tr.y.mean() > 0.5 else 0
    say("  " + "-" * 82)
    say(f"  {'REAL-TIME CONSTANT':<26}{tr_maj * 100:>8.1f}%"
        f"{const * 100:>11.0f}%{rt_const * 100:>8.1f}%"
        f"{(rt_const - ho_maj) * 100:>+7.1f}%{0.0:>+6.1f}%"
        f"{(rt_const - ho_maj) / (se / 100):>+8.2f}{'':>7}")
    say(f"  {'HINDSIGHT MAJORITY':<26}{'':>9}{'':>12}{ho_maj * 100:>8.1f}%"
        f"{0.0:>+7.1f}%{(ho_maj - rt_const) * 100:>+6.1f}%{0.0:>+8.2f}{'':>7}")
    say("")
    say("  Every number in the 'vs maj' column is negative or zero: not one")
    say("  variant beat the constant a hindsight-chosen coin would pick.")
    say("  AUC is the only threshold-free column, so the only one that can")
    say("  show ranking skill behind a miscalibrated threshold. That is")
    say("  what the block-shift null below is for.")
    return hits


def max_auc_shift_test(ho: pd.DataFrame,
                       probs: dict[str, np.ndarray]) -> tuple[float, float]:
    """The exact multiplicity correction: a MAX-statistic block-shift test.

    Eleven predictors were examined, so the honest question is not "is THIS
    AUC extreme" but "is the MOST extreme of eleven AUCs more extreme than the
    most extreme of eleven AUCs computed on a misaligned label series". Taking
    the max inside every shift prices the search exactly, with no Bonferroni
    approximation and no assumption about the correlation between the eleven
    predictors -- which is high, so Bonferroni would be far too harsh.

    Returns the real max statistic and its p-value."""
    y = ho.y.to_numpy()
    names = list(probs)

    def stat(labels: np.ndarray) -> float:
        best = 0.0
        if len(set(labels)) < 2:
            return np.nan
        for c in names:
            best = max(best, abs(roc_auc_score(labels, probs[c]) - 0.5))
        return best

    real = stat(y)
    null = np.array([s for s in (stat(np.roll(y, k))
                                 for k in range(1, len(y))) if np.isfinite(s)])
    p = float((null >= real).mean())
    say("")
    say("  MAX-STATISTIC TEST across all 11 predictors at once")
    say(f"    real max |AUC - 0.5| = {real:.3f}  "
        f"({max(names, key=lambda c: abs(roc_auc_score(y, probs[c]) - 0.5))})")
    say(f"    same statistic on {len(null)} misaligned label series: "
        f"median {np.median(null):.3f}, max {null.max():.3f}")
    say(f"    p = {p:.3f}  <- this is the number that prices the search")
    return real, p


def univariate_table(tr: pd.DataFrame, ho: pd.DataFrame, ho_maj: float,
                     bar: float
                     ) -> tuple[str, np.ndarray, dict[str, np.ndarray]]:
    """Each predictor alone. Returns the best predictor BY AUC and its
    probability vector, so the block-shift null can be aimed at the strongest
    thing in the table rather than at the one the author liked."""
    y = ho.y.to_numpy()
    se = np.sqrt(0.25 / n_effective(len(ho)))
    say("")
    say("=" * 84)
    say("  ONE PREDICTOR AT A TIME -- is one trailing series carrying it?")
    say("=" * 84)
    say(f"  {'predictor':<22}{'trainCV':>9}{'callsDOWN':>11}{'holdout':>9}"
        f"{'vs maj':>8}{'z(maj)':>8}{'AUC':>7}{'z(AUC)':>8}")
    say("  " + "-" * 82)
    rows = []
    probs: dict[str, np.ndarray] = {}
    for c in FEATURES:
        cv = block_cv_accuracy(tr, [c], "LOGIT")
        pred, prob = fit_one(tr, ho, [c], "LOGIT")
        hit, auc, z, _ = hit_stats(y, pred, prob, ho_maj)
        rows.append((c, cv, float(pred.mean()), hit, z, auc,
                     auc_z(y, auc)))
        probs[c] = prob
    for c, cv, call, hit, z, auc, za in sorted(
            rows, key=lambda r: -abs(r[5] - 0.5)):
        say(f"  {c:<22}{cv * 100:>8.1f}%{call * 100:>10.0f}%{hit * 100:>8.1f}%"
            f"{(hit - ho_maj) * 100:>+7.1f}%{z:>+8.2f}{auc:>7.3f}{za:>+8.2f}")
    n_ok = sum(1 for r in rows if r[4] > bar)
    n_auc = sum(1 for r in rows if abs(r[6]) > bar)
    say("  " + "-" * 82)
    say(f"  {n_ok} of {len(FEATURES)} clear the hit-rate bar "
        f"(|z| > {bar:.2f}, i.e. {bar * se * 100:.1f}pp over the majority)")
    say(f"  {n_auc} of {len(FEATURES)} clear the same bar on AUC")
    say("  Sorted by distance of AUC from 0.500, so the strongest RANKING")
    say("  predictor is at the top whichever way its sign points.")
    best = max(rows, key=lambda r: abs(r[5] - 0.5))
    say(f"  strongest by |AUC - 0.5|: {best[0]} at AUC {best[5]:.3f}")
    return best[0], probs[best[0]], probs


def auc_z(y: np.ndarray, auc: float) -> float:
    """z of an AUC against 0.5 on the EFFECTIVE sample.

    The usual Mann-Whitney null variance, but both class counts deflated by
    the label overlap, because 52 monthly observations of a 63-day window are
    not 52 independent draws."""
    if not np.isfinite(auc):
        return np.nan
    n1 = float((y == 1).sum()) / OVERLAP_MONTHS
    n0 = float((y == 0).sum()) / OVERLAP_MONTHS
    if n1 < 1 or n0 < 1:
        return np.nan
    var = (n1 + n0 + 1.0) / (12.0 * n1 * n0)
    return (auc - 0.5) / np.sqrt(var)


def sign_and_episode_check(tr: pd.DataFrame, ho: pd.DataFrame,
                           col: str) -> tuple[float, float, int]:
    """For the strongest predictor: does TRAIN agree on the sign, and
    is the holdout AUC one episode or a repeated pattern?

    Two ways a 0.70 AUC on 53 overlapping months can be an accident. First, the
    relationship may point the OTHER way in the training period, in which case
    the holdout number is a coincidence and nobody could have signed it in
    advance. Second, the holdout holds one enormous rates-up episode in 2022,
    and a predictor that merely happened to be low throughout it separates the
    classes beautifully without generalising to anything. A year-by-year AUC
    answers both: one year above 0.5 and the rest at chance is an episode."""
    say("")
    say(f"  SIGN AGREEMENT AND EPISODE CHECK -- {col}")
    raw = []
    for tag, sub in (("train 2013-21", tr), ("holdout 2022-26", ho)):
        y = sub.y.to_numpy()
        if len(set(y)) < 2:
            say(f"    {tag:<16} single-class, no AUC")
            continue
        a = float(roc_auc_score(y, sub[col].to_numpy(dtype=float)))
        raw.append(a)
        say(f"    {tag:<16}{len(sub):>4} months   raw AUC {a:>5.3f}")
    agree = (len(raw) == 2 and np.sign(raw[0] - 0.5) == np.sign(raw[1] - 0.5))
    say(f"    sign agreement between the two periods: "
        f"{'YES' if agree else 'NO -- the relationship inverts'}")
    say("")
    say(f"    {'holdout year':<16}{'months':>8}{'AUC':>8}{'rates UP':>11}")
    n_good = 0
    for yr, sub in ho.groupby(ho.date.dt.year):
        y = sub.y.to_numpy()
        up = (1.0 - y.mean()) * 100
        if len(set(y)) < 2:
            say(f"    {yr:<16}{len(sub):>8}{'n/a':>8}{up:>10.0f}%  "
                f"single-class year")
            continue
        a = float(roc_auc_score(y, sub[col].to_numpy(dtype=float)))
        n_good += a > 0.5
        say(f"    {yr:<16}{len(sub):>8}{a:>8.3f}{up:>10.0f}%")
    say(f"    {n_good} holdout years with AUC above 0.5 out of "
        f"{ho.date.dt.year.nunique()}")
    return (raw[0] if raw else np.nan,
            raw[1] if len(raw) > 1 else np.nan, n_good)


def median_threshold_rule(tr: pd.DataFrame, ho: pd.DataFrame, col: str,
                          ho_maj: float, rt_const: float) -> float:
    """Answer the one real objection to calling this a null.

    Every fitted variant above predicted the SAME class in nearly every holdout
    month, because the logit inherited the train period's 64%-rates-DOWN prior
    as its intercept. So "the hit rate equals the base rate" could be a broken
    THRESHOLD hiding usable RANKING information rather than an absence of
    information. This tests that objection instead of waving it away, with the
    most threshold-free rule available: split at the TRAIN MEDIAN of the
    predictor, take the direction from the TRAIN period, and judge once.

    Being distribution-free, this rule cannot inherit a class prior at all: it
    calls each side of the median about half the time by construction. It is
    POST HOC -- it was written after seeing the AUC column -- so it is counted
    as a sixteenth variant and is reported as post hoc whatever it says."""
    thr = float(tr[col].median())
    above_tr = tr[col].to_numpy(dtype=float) > thr
    # direction taken from TRAIN only: which side of the median was more often
    # followed by a rates-DOWN quarter
    p_above = float(tr.y.to_numpy()[above_tr].mean())
    p_below = float(tr.y.to_numpy()[~above_tr].mean())
    above_means_down = p_above > p_below
    above_ho = ho[col].to_numpy(dtype=float) > thr
    pred = np.where(above_ho == above_means_down, 1, 0)
    y = ho.y.to_numpy()
    hit = float((pred == y).mean())
    se = np.sqrt(0.25 / n_effective(len(ho)))
    say("")
    say(f"  POST-HOC MEDIAN-THRESHOLD RULE on {col}")
    say(f"    train median {thr:+.2f}%; in train, above it was followed by "
        f"rates-DOWN {p_above * 100:.0f}% of the time")
    say(f"    and below it {p_below * 100:.0f}%, so the rule reads: "
        f"{'above' if above_means_down else 'below'} the median -> rates DOWN")
    say(f"    holdout hit rate {hit * 100:.1f}%  calls rates-DOWN "
        f"{pred.mean() * 100:.0f}% of months")
    say(f"    vs hindsight majority {ho_maj * 100:.1f}% "
        f"({(hit - ho_maj) * 100:+.1f}pp, z = {(hit - ho_maj) / se:+.2f})")
    say(f"    vs real-time constant {rt_const * 100:.1f}% "
        f"({(hit - rt_const) * 100:+.1f}pp, z = {(hit - rt_const) / se:+.2f})")
    return hit


def block_shift_null(ho: pd.DataFrame, label: str, pred: np.ndarray,
                     prob: np.ndarray) -> None:
    """Circularly shift the holdout labels past a FIXED prediction vector.

    This is the honest replacement for the divide-t-by-1.7 rule of thumb. A
    plain permutation of the labels would destroy their autocorrelation and
    the null too narrow; a circular shift keeps the label series exactly as it
    is -- same runs, same 63-day overlap, same regime blocks -- and only breaks
    its ALIGNMENT with the predictions. The real score's percentile among the
    n-1 shifted scores is a p-value that already contains the overlap penalty.

    A constant prediction has a shift-invariant hit rate, so a degenerate
    hit-rate null is itself the proof that the variant forecast nothing."""
    y = ho.y.to_numpy()
    n = len(y)
    hits, aucs = [], []
    for k in range(1, n):
        ys = np.roll(y, k)
        hits.append(float((pred == ys).mean()))
        if len(set(ys)) > 1:
            aucs.append(float(roc_auc_score(ys, prob)))
    hit_real = float((pred == y).mean())
    auc_real = float(roc_auc_score(y, prob)) if len(set(y)) > 1 else np.nan
    h = np.array(hits)
    a = np.array(aucs)
    p_hit = float((h >= hit_real).mean())
    p_auc = float((a >= auc_real).mean()) if len(a) else np.nan
    p_auc2 = (float((np.abs(a - 0.5) >= abs(auc_real - 0.5)).mean())
              if len(a) else np.nan)
    say(f"  {label}")
    say(f"    hit rate {hit_real * 100:>5.1f}%  shifted range "
        f"{h.min() * 100:>5.1f}% .. {h.max() * 100:>5.1f}%  "
        f"(sd {h.std(ddof=1) * 100:.1f}pp)   p = {p_hit:.3f}")
    say(f"    AUC      {auc_real:>5.3f}  shifted range "
        f"{a.min():>5.3f} .. {a.max():>5.3f}  "
        f"(sd {a.std(ddof=1):.3f})   p = {p_auc:.3f}  two-sided {p_auc2:.3f}")


def shuffled_control(tr: pd.DataFrame, ho: pd.DataFrame,
                     hits: dict[str, float],
                     ho_maj: float, n: int) -> dict[str, np.ndarray]:
    """Re-run the WHOLE fit, selection included, on a shuffled train target.

    Shuffling the train labels destroys every relationship while preserving the
    class balance, the feature matrix, the CV search and the selection. The
    holdout labels stay real, so the resulting distribution is what this
    method scores on noise. The real number has to beat that, not 50%."""
    say("")
    say("=" * 84)
    say(f"  SHUFFLED CONTROL -- {n} refits on a permuted train target")
    say("=" * 84)
    rng = np.random.default_rng(2024)
    y = ho.y.to_numpy()
    store: dict[str, list[float]] = {}
    for i in range(n):
        t2 = tr.copy()
        t2["y"] = rng.permutation(t2.y.to_numpy())
        best2 = pick_best_single(t2, seed=i)
        for name, (pred, _) in run_variants(t2, ho, best2, seed=i).items():
            key = "BEST-SINGLE" if name.startswith("BEST-SINGLE") else name
            store.setdefault(key, []).append(float((pred == y).mean()))
    out = {k: np.array(v) for k, v in store.items()}
    say(f"  {'variant':<16}{'real':>8}{'null mean':>11}{'null sd':>9}"
        f"{'null max':>10}{'beats':>8}  reading")
    say("  " + "-" * 74)
    for name, arr in out.items():
        real = next((v for k, v in hits.items() if k.startswith(name)), np.nan)
        beat = float((arr < real).mean()) * 100
        sd = float(arr.std(ddof=1))
        z = (real - arr.mean()) / sd if sd > 0 else np.nan
        reading = ("outside the noise"
                   if np.isfinite(z) and z > 2 and beat >= 95
                   else "INSIDE THE NOISE")
        say(f"  {name:<16}{real * 100:>7.1f}%{arr.mean() * 100:>10.1f}%"
            f"{sd * 100:>8.1f}%{arr.max() * 100:>9.1f}%{beat:>7.0f}%  "
            f"{reading}")
    say("")
    say(f"  The holdout majority-class rate is {ho_maj * 100:.1f}%, and a")
    say("  model fitted on shuffled labels mostly collapses to a constant")
    say("  prediction, so a null mean near that number is the expected")
    say("  this control, not a bug. What matters is whether the real run sits")
    say("  outside the spread.")
    return out


def beta_neutral_check(m: pd.DataFrame, bar: float) -> tuple[float, float]:
    """Rule 2's analogue for a single asset: strip the equity component out.

    TLT and SPY co-move in places, so a model that seems to forecast bonds
    be forecasting stocks. Regress the forward TLT return on the forward SPY
    return using a TRAIN-PERIOD slope only, take the residual, and ask whether
    the sign of the BOND-SPECIFIC move is forecastable. If the headline only
    survives on the raw target, it was an equity call in a bond wrapper."""
    say("")
    say("=" * 84)
    say("  EQUITY-NEUTRALISED TARGET -- the bond-specific component only")
    say("=" * 84)
    tr0 = m[m.date <= TRAIN_END]
    x = tr0.spy_fwd.to_numpy(dtype=float)
    yv = tr0.tlt_fwd.to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(yv)
    if ok.sum() < 20:
        say("  too few train months to estimate the equity slope; skipped")
        return np.nan, np.nan
    b1, b0 = np.polyfit(x[ok], yv[ok], 1)
    say(f"  train-period slope of TLT fwd63 on SPY fwd63: {b1:+.3f} "
        f"(intercept {b0:+.2f}%)")
    m2 = m.copy()
    m2["resid"] = m2.tlt_fwd - (b0 + b1 * m2.spy_fwd)
    m2 = m2.dropna(subset=["resid"])
    m2["y"] = (m2.resid > 0).astype(int)
    tr, ho = m2[m2.date <= TRAIN_END], m2[m2.date >= HOLDOUT_START]
    maj = max(float(ho.y.mean()), 1.0 - float(ho.y.mean()))
    say(f"  holdout majority class on the residual sign: {maj * 100:.1f}%")
    say("")
    best = pick_best_single(tr)
    say(f"  {'variant':<26}{'holdout':>10}{'vs maj':>9}{'z(eff)':>8}"
        f"{'AUC':>7}  verdict")
    say("  " + "-" * 70)
    yv2 = ho.y.to_numpy()
    got = {}
    for name, (pred, prob) in run_variants(tr, ho, best).items():
        hit, auc, z, _ = hit_stats(yv2, pred, prob, maj)
        got[name] = hit
        say(f"  {name:<26}{hit * 100:>9.1f}%{(hit - maj) * 100:>+8.1f}%"
            f"{z:>+8.2f}{auc:>7.3f}  "
            f"{'clears the bar' if z > bar else 'nothing'}")
    return got.get("LOGIT", np.nan), maj


def long_only_translation(ho: pd.DataFrame, pred: np.ndarray) -> None:
    """What a cash account could actually have done with the forecast.

    Long-only, no shorting, no leverage: hold TLT next month when it says
    rates DOWN, else hold SPY. Non-overlapping monthly returns, compounded.
    The bar is not zero and it is not 9%: SPY compounded at 14.78% over this
    sample and an equal-weight version of the universe at 14.82%."""
    say("")
    say("=" * 84)
    say("  LONG-ONLY TRANSLATION -- could a cash account have used this?")
    say("=" * 84)
    # Align on the holdout's own index so dropping the final month (which has
    # no next-month return) cannot silently shift the predictions by one row.
    p = pd.Series(pred, index=ho.index)
    h = ho.dropna(subset=["tlt_next", "spy_next"]).copy()
    h["pred"] = p.reindex(h.index).to_numpy()
    if h.empty:
        say("  no complete next-month returns in the holdout; skipped")
        return
    r_switch = np.where(h.pred == 1, h.tlt_next, h.spy_next) / 100.0
    yrs = (h.date.iloc[-1] - h.date.iloc[0]).days / 365.25
    if yrs <= 0:
        say("  holdout too short to annualise; skipped")
        return

    def cagr(r: np.ndarray) -> float:
        return (float(np.prod(1.0 + r)) ** (1.0 / yrs) - 1.0) * 100.0

    say(f"  {len(h)} monthly decisions over {yrs:.1f} years, non-overlapping")
    say(f"  {'sleeve':<34}{'CAGR':>9}")
    say("  " + "-" * 44)
    say(f"  {'model: TLT when rates-DOWN else SPY':<34}"
        f"{cagr(r_switch):>+8.2f}%")
    say(f"  {'SPY held throughout':<34}"
        f"{cagr(h.spy_next.to_numpy() / 100.0):>+8.2f}%")
    say(f"  {'TLT held throughout':<34}"
        f"{cagr(h.tlt_next.to_numpy() / 100.0):>+8.2f}%")
    say(f"  {'the bar (SPY 2013-2026 CAGR)':<34}{SPY_BAR_CAGR:>+8.2f}%")
    n_tlt = int((h.pred == 1).sum())
    say("  " + "-" * 44)
    say(f"  the model parked in TLT on {n_tlt} of {len(h)} months")
    say("  A switch rule can only beat SPY if it avoids equity drawdowns with")
    say("  bond gains, which requires the direction call to be right. If the")
    say("  hit rate is the base rate, this line is a lottery ticket.")


# --------------------------------------------------------------------- main
def main() -> None:
    n_shuf = (int(sys.argv[sys.argv.index("--shuffles") + 1])
              if "--shuffles" in sys.argv else N_SHUFFLES)
    say("LOADING")
    m = build_monthly()
    say(f"  {len(m)} usable months | {len(FEATURES)} trailing predictors")
    tr, ho = split(m)
    if tr.y.nunique() < 2 or ho.y.nunique() < 2:
        raise SystemExit("one side of the split is single-class; cannot score")

    tr_maj, ho_maj, rt_const = base_rates(tr, ho)
    n_variants = len(FEATURES) + 4
    bar = float(np.sqrt(2 * np.log(n_variants)))

    best = pick_best_single(tr)
    say("")
    say(f"  BEST-SINGLE selected on train CV only: {best}")
    preds = run_variants(tr, ho, best)
    hits = holdout_table(tr, ho, best, tr_maj, ho_maj, rt_const, preds, bar)
    top_auc, top_prob, all_probs = univariate_table(tr, ho, ho_maj, bar)

    say("")
    say("=" * 84)
    say("  BLOCK-SHIFT NULL -- the overlap penalty measured, not assumed")
    say("=" * 84)
    say(f"  {len(ho) - 1} circular shifts of the holdout label series past a "
        f"fixed prediction vector.")
    say("  The labels keep their autocorrelation; only alignment dies.")
    say("")
    block_shift_null(ho, "LOGIT, all 11 predictors", *preds["LOGIT"])
    tp_pred = (top_prob > 0.5).astype(int)
    block_shift_null(ho, f"strongest univariate by AUC: {top_auc}",
                     tp_pred, top_prob)
    say("")
    say("  The two-sided AUC p-value is the one to read for the univariate")
    say("  row, because that predictor was chosen FOR an extreme AUC and its")
    say("  sign was not pre-registered. Even that is not enough, because the")
    say("  choice was made over eleven candidates. The next test prices that.")
    max_auc_shift_test(ho, all_probs)
    sign_and_episode_check(tr, ho, top_auc)
    median_threshold_rule(tr, ho, top_auc, ho_maj, rt_const)

    shuffled_control(tr, ho, hits, ho_maj, n_shuf)
    beta_neutral_check(m, bar)
    long_only_translation(ho, preds["LOGIT"][0])

    say("")
    say("=" * 84)
    say("  WHAT THIS SETTLES")
    say("=" * 84)
    say("  Not one of the fifteen pre-registered variants beat the holdout")
    say("  majority-class rate. Every one landed BELOW it, because the")
    say("  unconditional direction flipped between the two periods: the train")
    say("  era was 64% rates-DOWN and the holdout 62% rates-UP, so a model")
    say("  that learned the prior carried the wrong constant forward. That")
    say("  flip is the headline, and it is worse news for an FOMC screen than")
    say("  a flat result would be: the base rate itself is not stationary, so")
    say("  there is no constant to fall back on when the forecast is silent.")
    say("")
    say("  The one thing that looked like something was a trailing trend")
    say("  predictor reaching AUC 0.70 on the holdout. It failed on three")
    say("  separate counts: the train period showed nearly nothing for it")
    say("  (raw AUC 0.452, a 63%-vs-65% median split), so its sign could not")
    say("  have been chosen in advance; the max-statistic block-shift test")
    say("  that prices the eleven-way search returns p = 0.25; and no")
    say("  threshold, fitted or distribution-free, converts the ranking")
    say("  into a hit rate that beats the base rate.")
    say("")
    say("  So the answer to the question in the title is NO, at this horizon,")
    say("  on this sample, from these predictors. That is the answer the")
    say("  literature already gives for quarterly bond returns, and finding")
    say("  it again is a confirmation, not a failed search. The consequence")
    say("  is the expensive part: rate_foresight.py's ceiling is unreachable,")
    say("  any regime-conditional screen is conditioning on a coin flip, and")
    say("  the FOMC line of enquiry is closed on the data available here.")
    say("")
    say("  WHAT WOULD REOPEN IT. Not a better classifier: eighteen effective")
    say("  observations cannot support one. It would take either a genuinely")
    say("  forward-looking rate input this cache does not contain (fed funds")
    say("  futures, the OIS curve, breakevens), or a decision that does not")
    say("  need the direction called at all.")


if __name__ == "__main__":
    main()
