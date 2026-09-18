from AlgorithmImports import *
import numpy as np


class CapitulationFreq(QCAlgorithm):
    """Capitulation entry - DIAGNOSTIC BUILD. Paste as main.py on QuantConnect.

    WHY THIS FILE EXISTS
    The first QC run of capitulation.py (2005-2026, $25k) produced 43 orders in
    21 years and the owner asked to loosen the rule to ~0.7 entries a month.
    A local replay of the EXACT rule on a 2005-2026 S&P panel says the rule
    already fires ~2.35 entries a month (606 entries / 258 months, monthly,
    -40%, 10% off the low, ~460 names) - and that panel is survivors only, so
    the true point-in-time count is higher still. 0.7/month is BELOW what the
    rule produces. Therefore 43 orders is not the rule: the QC run is failing
    to execute it, and loosening thresholds would tune around a bug.

    Two known ways this dies silently on QC, both instrumented here:
      A. History dataframe indexing. hist.loc[sym]["close"] can miss for every
         symbol depending on how LEAN indexes the frame, and the except:continue
         swallows it - zero candidates, forever, no error. This build unstacks
         the frame once (time x symbol) and counts how many names had history.
      B. Cash-account order rejection. Buys can be refused for unsettled cash;
         n_entries then climbs while fills do not. OnOrderEvent logs every
         invalid/cancelled order with the broker message.

    Thresholds are the MEASURED originals, deliberately unchanged. Run this,
    read the Logs tab, and the per-rebalance line tells you where entries go:
        rebal 2009-03-02 syms=612 hist=598 cands=41 take=20 held=0
    If cands is large and orders stay tiny -> B. If hist is ~0 -> A.

    PRE-REGISTERED GATE (unchanged from capitulation.py)
      Keep only if mean per-trade tilt vs SPY > 0 with t > 2.
      The headline return is mostly parked SPY and proves nothing.
    """

    # ------------------------------------------------------------ parameters
    REBALANCE = "monthly"         # "monthly" (measured rule) | "weekly"
    DD_TRIGGER = -0.40            # measured threshold - NOT tuned
    OFF_LOW_MIN = 0.10            # measured falling-knife filter
    LOW_WINDOW = 60
    LOOKBACK = 252
    HOLD_DAYS = 63
    MAX_POSITIONS = 20
    UNIVERSE_SIZE = 500
    MIN_PRICE = 3.0
    MIN_DOLLAR_VOL = 5e6
    INVESTED = 0.90
    USE_SPY_FALLBACK = True
    LOG_EVERY_REBALANCE = True

    def Initialize(self):
        self.SetStartDate(2005, 1, 1)
        self.SetEndDate(2026, 6, 12)
        self.SetCash(25000)
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage,
                               AccountType.Cash)
        self.Settings.FreePortfolioValuePercentage = 0.05
        self.Settings.MinimumOrderMarginPortfolioPercentage = 0.005

        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.Coarse, self.Fine)
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol

        self.held = {}
        self.bar = 0
        self.n_entries = 0
        self.n_exits = 0
        self.wins = 0
        self.losses = 0
        self.entry_px = {}
        self.entry_spy = {}
        self.rels = []
        self.rebals_total = 0
        self.rebals_none = 0
        self.cap_weight = []
        # diagnostics
        self.n_rejected = 0
        self.n_filled_buys = 0
        self.hist_zero_rebals = 0

        rule = (self.DateRules.WeekStart(self.spy) if self.REBALANCE == "weekly"
                else self.DateRules.MonthStart(self.spy))
        self.Schedule.On(rule, self.TimeRules.AfterMarketOpen(self.spy, 30),
                         self.Rebalance)
        self.SetWarmUp(self.LOOKBACK + 10, Resolution.Daily)

    def Coarse(self, coarse):
        ok = [c for c in coarse
              if c.HasFundamentalData and c.Price > self.MIN_PRICE
              and c.DollarVolume > self.MIN_DOLLAR_VOL]
        ok.sort(key=lambda c: c.DollarVolume, reverse=True)
        return [c.Symbol for c in ok[:self.UNIVERSE_SIZE]]

    def Fine(self, fine):
        return [f.Symbol for f in fine if f.MarketCap and f.MarketCap > 5e8]

    def OnData(self, data):
        if not self.IsWarmingUp:
            self.bar += 1

    def OnOrderEvent(self, e):
        # B: make cash-account rejections visible instead of silent
        if e.Status in (OrderStatus.Invalid, OrderStatus.Canceled):
            self.n_rejected += 1
            if self.n_rejected <= 40:
                self.Log("ORDER %s %s qty=%s msg=%s" % (
                    e.Status, e.Symbol, e.FillQuantity if e.FillQuantity else e.Quantity,
                    getattr(e, "Message", "")))
        elif e.Status == OrderStatus.Filled and e.FillQuantity > 0 \
                and e.Symbol != self.spy:
            self.n_filled_buys += 1

    def Rebalance(self):
        if self.IsWarmingUp:
            return
        self.rebals_total += 1

        symbols = [s for s in self.ActiveSecurities.Keys if s != self.spy]
        if len(symbols) < 50:
            if self.LOG_EVERY_REBALANCE:
                self.Log("rebal %s syms=%d  (<50, skipped)" % (self.Time.date(), len(symbols)))
            return
        hist = self.History(symbols, self.LOOKBACK + 5, Resolution.Daily)
        if hist.empty or "close" not in hist.columns:
            self.hist_zero_rebals += 1
            if self.LOG_EVERY_REBALANCE:
                self.Log("rebal %s syms=%d  HISTORY EMPTY" % (self.Time.date(), len(symbols)))
            return
        # A: one robust reshape instead of per-symbol .loc that can silently miss
        try:
            closes = hist["close"].unstack(level=0)
        except Exception as ex:
            self.hist_zero_rebals += 1
            self.Log("rebal %s unstack failed: %s" % (self.Time.date(), ex))
            return

        closing = []
        spy_now = float(self.Securities[self.spy].Price)
        for sym in list(self.held.keys()):
            if self.bar - self.held[sym] >= self.HOLD_DAYS:
                if self.Portfolio[sym].Invested:
                    px = float(self.Securities[sym].Price)
                    e = self.entry_px.get(sym, px)
                    if px > e:
                        self.wins += 1
                    else:
                        self.losses += 1
                    e_spy = self.entry_spy.get(sym, 0.0)
                    if e > 0 and e_spy > 0 and spy_now > 0:
                        self.rels.append(((px / e - 1.0)
                                          - (spy_now / e_spy - 1.0)) * 100)
                    closing.append(sym)
                    self.n_exits += 1
                self.held.pop(sym, None)
                self.entry_px.pop(sym, None)
                self.entry_spy.pop(sym, None)

        cands = []
        n_hist = 0
        for sym in closes.columns:
            if sym in self.held:
                continue
            c = closes[sym].dropna()
            if len(c) < self.LOOKBACK - 20:
                continue
            n_hist += 1
            hi = float(c.max())
            spot = float(c.iloc[-1])
            if hi <= 0 or spot <= 0:
                continue
            dd = spot / hi - 1.0
            if dd > self.DD_TRIGGER:
                continue
            lo = float(c.tail(self.LOW_WINDOW).min())
            if lo <= 0 or (spot / lo - 1.0) < self.OFF_LOW_MIN:
                continue
            cands.append((sym, dd))

        room = self.MAX_POSITIONS - len(self.held)
        if cands and room > 0:
            cands.sort(key=lambda t: t[1])
            take = [s for s, _ in cands[:room]]
        else:
            take = []
        if not cands:
            self.rebals_none += 1

        if self.LOG_EVERY_REBALANCE:
            self.Log("rebal %s syms=%d hist=%d cands=%d take=%d held=%d"
                     % (self.Time.date(), len(symbols), n_hist, len(cands),
                        len(take), len(self.held)))

        targets = [PortfolioTarget(s, 0) for s in closing]
        n_slots = max(len(self.held) + len(take), 1)
        w = self.INVESTED / max(n_slots, 1)
        n_cap = 0
        for sym in list(self.held.keys()) + take:
            if sym in self.Securities and self.Securities[sym].Price > 0:
                targets.append(PortfolioTarget(sym, w))
                n_cap += 1
        self.cap_weight.append(w * n_cap)

        if self.USE_SPY_FALLBACK:
            leftover = max(0.0, self.INVESTED - w * len(targets))
            if leftover > 0.02:
                targets.append(PortfolioTarget(self.spy, leftover))
        elif not targets:
            self.Liquidate()

        if targets:
            self.SetHoldings(targets)
            for sym in take:
                self.held[sym] = self.bar
                self.entry_px[sym] = float(self.Securities[sym].Price)
                self.entry_spy[sym] = spy_now
                self.n_entries += 1

    def OnEndOfAlgorithm(self):
        months = max(1, (self.EndDate.year - self.StartDate.year) * 12
                     + self.EndDate.month - self.StartDate.month)
        n = len(self.rels)
        mean = float(np.mean(self.rels)) if n else float("nan")
        t = (mean / (float(np.std(self.rels, ddof=1)) / np.sqrt(n))
             if n > 2 else float("nan"))
        self.Log("=== CAPITULATION DIAGNOSTIC  rebalance=%s DD=%.2f OFF_LOW=%.2f universe=%d"
                 % (self.REBALANCE, self.DD_TRIGGER, self.OFF_LOW_MIN, self.UNIVERSE_SIZE))
        self.Log("signal entries %d  exits %d  -> %.2f entries/month  (local replay expects ~2.3)"
                 % (self.n_entries, self.n_exits, self.n_entries / months))
        self.Log("FILLED buys %d   REJECTED/cancelled orders %d   rebalances with empty history %d / %d"
                 % (self.n_filled_buys, self.n_rejected, self.hist_zero_rebals, self.rebals_total))
        self.Log("rebalances with zero candidates %d / %d" % (self.rebals_none, self.rebals_total))
        self.Log("wins %d  losses %d" % (self.wins, self.losses))
        self.Log("PER-TRADE TILT vs SPY over the same hold: mean %+.2f%%  t=%.2f  n=%d"
                 % (mean, t, n))
        self.Log("book weight in the signal (mean): %.0f%%  rest parked in SPY"
                 % (100 * float(np.mean(self.cap_weight)) if self.cap_weight else 0))
        self.Log("READ IT: entries high but FILLED low -> cash rejections (B). "
                 "hist ~0 -> indexing (A). GATE: tilt > 0 and t > 2.")
