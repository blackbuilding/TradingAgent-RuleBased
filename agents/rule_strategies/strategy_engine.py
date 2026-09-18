#!/usr/bin/env python3
"""Rule-based swing-trade / day-trade / support-resistance entry-exit strategies.

A standalone sibling to agents/asset_analysis (A3C model) and
agents/strategy_backtester (A3C walk-forward) — no imports from either, by
design, so this agent has no dependency on the RL pipeline at all. Indicator
formulas (RSI/MACD/MAE) are duplicated locally rather than shared, matching
the convention already used across this repo's agents.

Three classic technical strategies, each with real stop-loss/take-profit
exit logic (something the A3C model has none of):

- support_resistance: mean-reversion — buy near a confirmed swing-low
  support level, exit at the next swing-high resistance or a stop just
  below support.
- swing: trend continuation on a MACD crossover (RSI 40-70, price above the
  MAE mid), wider stop (2x ATR) and target (1:2.5 reward:risk) for a
  multi-day/week hold.
- daytrade: the same entry condition as swing, but a tight stop (0.75x ATR),
  a smaller target (1:1.5 R:R), and a forced time exit after max_hold_bars
  if neither stop nor target is hit (FX trades 24/5, so this approximates a
  same-session constraint).

Position sizing is risk-based (risk a fixed % of balance per trade, sized by
stop distance), not a flat % of cash, per standard risk-management practice.

Output is a self-contained interactive HTML report (Plotly, embedded JS —
opens in any browser, no server or internet needed): MT5 Strategy
Tester-style summary, a zoomable/pannable equity curve + net-profit curve +
price/RSI/MACD panel with hover tooltips carrying each trade's rationale,
and the full trade log as a table.

This produces a decision-support signal, not investment advice, and never
places a real order — everything here is simulated.
"""
import argparse
import datetime as dt
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "input"
REPORTS_DIR = REPO_ROOT / "output"

RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
MAE_PERIOD, MAE_PCT = 20, 0.025
ATR_PERIOD = 14
SR_LOOKBACK = 10

# Extra bars fetched/loaded before `start` so RSI/MACD/MAE/ATR are warmed up
# with real history by the time trading begins at `start`, instead of being
# NaN (dropped) or biased over the first bars of the requested window — the
# window itself still only trades from `start` onward (see start_idx below).
WARMUP_BARS = 2 * MAE_PERIOD

# Generous calendar-day buffer per warmup bar, keyed by rate, sized to
# guarantee WARMUP_BARS of real history regardless of the instrument's
# trading calendar (24/5 FX, ~6.5h/day equity sessions, weekends, holidays).
# Overshooting is harmless — the extra bars only warm up indicators and are
# never traded (see start_idx in run_strategy).
CALENDAR_DAYS_PER_WARMUP_BAR = {
    "M1": 0.01, "M5": 0.05, "M15": 0.15, "M30": 0.3,
    "H1": 0.5, "H4": 1.5, "D1": 2.5,
}


def compute_warmup_start(start: dt.date, rate: str) -> dt.date:
    days_per_bar = CALENDAR_DAYS_PER_WARMUP_BAR.get(rate, 2.5)
    buffer_days = max(5, int(WARMUP_BARS * days_per_bar))
    return start - dt.timedelta(days=buffer_days)

STRATEGY_CHOICES = ("swing", "daytrade", "support_resistance")


# --- indicators (self-contained; same formulas as asset_analysis/plot_report, duplicated on purpose) ---

def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def compute_mae(close: pd.Series, period=MAE_PERIOD, pct=MAE_PCT):
    middle = close.rolling(period).mean()
    return middle, middle * (1 + pct), middle * (1 - pct)


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def load_asset_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    df["rsi"] = compute_rsi(df["close"])
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(df["close"])
    df["mae_mid"], df["mae_upper"], df["mae_lower"] = compute_mae(df["close"])
    df["atr"] = compute_atr(df)
    return df.dropna().reset_index(drop=True)


# --- data resolution (own copy of the CSV-fallback logic, no cross-agent import) ---

CSV_FILENAME_RE = re.compile(r"^(?P<symbol>.+)_(?P<rate>[A-Za-z0-9]+)_(?P<start>\d{8})_(?P<end>\d{8})\.csv$")


def find_broader_csv(symbol: str, rate: str, start: dt.date, end: dt.date):
    """Finds an existing CSV that actually covers [start, end], verified by
    reading each candidate's real min/max time — not inferred from its
    filename, since a dated file's literal range can come back short (e.g. an
    MT5 fetch that had less history than requested). Legacy whole-history
    files are tried first (most likely to cover any range), then dated files
    widest-start-first."""
    start_ts, end_ts = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    candidates = []
    legacy = DATA_DIR / f"{symbol}_{rate}.csv"
    if legacy.exists():
        candidates.append(legacy)
    dated = []
    for p in DATA_DIR.glob(f"{symbol}_{rate}_*_*.csv"):
        m = CSV_FILENAME_RE.match(p.name)
        if m:
            dated.append((m.group("start"), m.group("end"), p))
    dated.sort(key=lambda t: (t[0], t[1]))  # earliest start first, then earliest end
    candidates.extend(p for _, _, p in dated)

    for p in candidates:
        df = pd.read_csv(p, usecols=["time"], parse_dates=["time"])
        if df["time"].min() <= start_ts and df["time"].max() >= end_ts:
            return p
    return None


def ensure_csv(csv_path: Path, symbol: str, rate: str, start: dt.date, end: dt.date):
    if csv_path.exists():
        return
    source = find_broader_csv(symbol, rate, start, end)
    if source is None:
        raise FileNotFoundError(
            f"No historical data at {csv_path}, and no existing {symbol} {rate} CSV covers [{start}, {end}]. "
            f"Fetch it first (this agent fetches its own data via MT5 when invoked normally, or run "
            f"historical-data-collector / strategy-backtester for symbol={symbol} rate={rate})."
        )
    df = pd.read_csv(source, parse_dates=["time"])
    start_ts, end_ts = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    window = df[(df["time"] >= start_ts) & (df["time"] <= end_ts)]
    if window.empty:
        raise FileNotFoundError(f"No historical data at {csv_path}, and {source} doesn't cover [{start}, {end}].")
    window.to_csv(csv_path, index=False)
    print(f"Sliced {len(window)} bars from {source} into {csv_path}")


def resolve_paths(symbol: str, rate: str, start: dt.date, end: dt.date, strategy: str, data_start: dt.date = None):
    range_tag = f"{symbol}_{rate}_{start:%Y%m%d}_{end:%Y%m%d}"
    data_range_tag = f"{symbol}_{rate}_{(data_start or start):%Y%m%d}_{end:%Y%m%d}"
    tag = f"{range_tag}_{strategy}"
    return {
        "tag": tag,
        # csv_path spans [data_start, end] (includes the warmup buffer before
        # `start`); report naming stays keyed to the user-requested [start, end]
        "csv_path": DATA_DIR / f"{data_range_tag}.csv",  # shared across strategies/agents, no strategy suffix
        "report_path": REPORTS_DIR / f"{tag}_report.html",
    }


# --- support/resistance: fractal swing points, exposed without lookahead ---

def find_swing_points(df: pd.DataFrame, lookback: int = SR_LOOKBACK):
    """Confirmed swing highs/lows. A swing point at bar i isn't knowable until
    `lookback` bars after it (needs that many future bars to confirm it was
    the local extreme), so confirm_idx = i + lookback — walk-forward code
    must only use levels with confirm_idx <= the current bar."""
    highs, lows = df["high"].values, df["low"].values
    n = len(df)
    levels = []
    for i in range(lookback, n - lookback):
        window_hi = highs[i - lookback : i + lookback + 1]
        window_lo = lows[i - lookback : i + lookback + 1]
        if highs[i] == window_hi.max():
            levels.append({"pivot_idx": i, "confirm_idx": i + lookback, "price": float(highs[i]), "kind": "swing_high"})
        if lows[i] == window_lo.min():
            levels.append({"pivot_idx": i, "confirm_idx": i + lookback, "price": float(lows[i]), "kind": "swing_low"})
    levels.sort(key=lambda l: l["confirm_idx"])
    return levels


class SRTracker:
    """Exposes only swing levels confirmed as of the current bar — advance_to()
    must be called with a non-decreasing bar_idx each step to preserve that."""

    def __init__(self, levels):
        self.levels = levels
        self._next = 0
        self.available = []

    def advance_to(self, bar_idx: int):
        while self._next < len(self.levels) and self.levels[self._next]["confirm_idx"] <= bar_idx:
            self.available.append(self.levels[self._next])
            self._next += 1

    def nearest_support(self, price: float):
        below = [l["price"] for l in self.available if l["price"] < price]
        return max(below) if below else None

    def nearest_resistance(self, price: float):
        above = [l["price"] for l in self.available if l["price"] > price]
        return min(above) if above else None


# --- strategies: each returns a buy signal (with stop/target) or hold; exits
# themselves are handled uniformly by run_strategy (stop/target/time), not here ---

TREND_LOOKBACK = 10  # bars; long-term regime filter — MAE mid must have net-risen/fallen over this window, not just favor the entry on this bar alone


def _trend_entry_ok(df: pd.DataFrame, i: int) -> bool:
    """The bullish MACD cross must have happened by the previous bar and
    still hold now, rather than firing on the raw cross bar itself — this
    filters same-bar/next-bar whipsaws where the cross reverses almost
    immediately, at the cost of entering one bar later (and a bit further
    from the crossover price) than an unconfirmed entry would. Also requires
    a standing uptrend: MAE mid higher than it was TREND_LOOKBACK bars ago,
    not just a single-bar close above it — filters crossovers that fire in a
    sideways/choppy MAE mid (real trend continuation vs. noise)."""
    if i < max(2, TREND_LOOKBACK):
        return False
    row, prev, prev2 = df.iloc[i], df.iloc[i - 1], df.iloc[i - 2]
    crossed_up = prev2["macd"] <= prev2["macd_signal"] and prev["macd"] > prev["macd_signal"]
    still_bullish = row["macd"] > row["macd_signal"]
    trend_up = row["mae_mid"] > df.iloc[i - TREND_LOOKBACK]["mae_mid"]
    return bool(crossed_up and still_bullish and trend_up and 40 <= row["rsi"] <= 70 and row["close"] > row["mae_mid"])


def _trend_entry_short_ok(df: pd.DataFrame, i: int) -> bool:
    """Bearish mirror of _trend_entry_ok: MACD crosses below signal (confirmed
    one bar later, same whipsaw filter as the long side), a standing
    downtrend (MAE mid lower than TREND_LOOKBACK bars ago), RSI in a healthy
    (not yet oversold) 30-60 band, price below the MAE mid."""
    if i < max(2, TREND_LOOKBACK):
        return False
    row, prev, prev2 = df.iloc[i], df.iloc[i - 1], df.iloc[i - 2]
    crossed_down = prev2["macd"] >= prev2["macd_signal"] and prev["macd"] < prev["macd_signal"]
    still_bearish = row["macd"] < row["macd_signal"]
    trend_down = row["mae_mid"] < df.iloc[i - TREND_LOOKBACK]["mae_mid"]
    return bool(crossed_down and still_bearish and trend_down and 30 <= row["rsi"] <= 60 and row["close"] < row["mae_mid"])


def support_resistance_signal(df, i, sr: SRTracker, position, tolerance_pct: float = 0.003):
    row = df.iloc[i]
    price = row["close"]
    support = sr.nearest_support(price)
    if support is None or (price - support) / support > tolerance_pct or row["rsi"] >= 65:
        return {"action": "hold"}
    resistance = sr.nearest_resistance(price)
    stop = support * 0.995
    target = resistance if resistance is not None else price + (price - stop) * 2
    return {
        "action": "buy",
        "stop": stop,
        "target": target,
        "reason": f"price {price:.5f} within {tolerance_pct * 100:.1f}% of support {support:.5f}, RSI {row['rsi']:.1f}",
    }


def swing_signal(df, i, sr: SRTracker, position, stop_atr_mult: float = 2.0, target_rr: float = 2.5):
    row = df.iloc[i]
    price, atr = row["close"], row["atr"]

    if _trend_entry_ok(df, i):
        stop = price - stop_atr_mult * atr
        support = sr.nearest_support(price)
        if support is not None and support > stop:
            stop = support * 0.998
        risk = price - stop
        target = price + risk * target_rr
        resistance = sr.nearest_resistance(price)
        if resistance is not None and resistance < target:
            target = resistance
        return {
            "action": "buy",
            "stop": stop,
            "target": target,
            "reason": f"MACD crossed up, RSI {row['rsi']:.1f} (40-70), price above MAE mid; stop {stop_atr_mult}x ATR",
        }

    if _trend_entry_short_ok(df, i):
        stop = price + stop_atr_mult * atr
        resistance = sr.nearest_resistance(price)
        if resistance is not None and resistance < stop:
            stop = resistance * 1.002
        risk = stop - price
        target = price - risk * target_rr
        support = sr.nearest_support(price)
        if support is not None and support > target:
            target = support
        return {
            "action": "sell_short",
            "stop": stop,
            "target": target,
            "reason": f"MACD crossed down, RSI {row['rsi']:.1f} (30-60), price below MAE mid; stop {stop_atr_mult}x ATR",
        }

    return {"action": "hold"}


def day_trade_signal(df, i, sr: SRTracker, position, stop_atr_mult: float = 0.75, target_rr: float = 1.5):
    if not _trend_entry_ok(df, i):
        return {"action": "hold"}
    row = df.iloc[i]
    price, atr = row["close"], row["atr"]
    stop = price - stop_atr_mult * atr
    risk = price - stop
    target = price + risk * target_rr
    return {
        "action": "buy",
        "stop": stop,
        "target": target,
        "reason": f"MACD crossed up, RSI {row['rsi']:.1f} (40-70), price above MAE mid; tight {stop_atr_mult}x ATR stop",
    }


STRATEGIES = {
    "support_resistance": support_resistance_signal,
    "swing": swing_signal,
    "daytrade": day_trade_signal,
}


def run_strategy(
    df: pd.DataFrame,
    strategy_name: str,
    initial_balance: float = 10000,
    risk_pct: float = 0.02,
    sr_lookback: int = SR_LOOKBACK,
    reentry_cooldown: int = 3,
    max_hold_bars: int = 24,
    start_idx: int = 0,
    **strategy_kwargs,
):
    """`df` may include warmup bars before `start_idx` (real history used to
    seed indicators/support-resistance so decisions right at `start_idx` use
    proper prior-bar context, e.g. a MACD crossover on the first traded bar);
    no entries are evaluated and nothing is recorded before `start_idx`."""
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy_name!r}; choices: {list(STRATEGIES)}")
    signal_fn = STRATEGIES[strategy_name]

    levels = find_swing_points(df, lookback=sr_lookback)
    sr = SRTracker(levels)

    balance = initial_balance
    position = None
    equity_curve = [{"time": df.iloc[start_idx]["time"], "net_worth": initial_balance}]
    trade_log = []
    last_exit_idx = start_idx - reentry_cooldown
    next_ticket = 1  # simulated per-position ticket (sequential); no order_send is ever called, so there's no real MT5 ticket to report

    for i in range(len(df)):
        sr.advance_to(i)
        if i < start_idx:
            continue  # warmup-only: builds indicator/S-R history, no trading yet
        row = df.iloc[i]
        price = row["close"]

        if position is not None:
            direction = position["direction"]
            exit_price = exit_reason = None
            if direction == "long":
                if row["low"] <= position["stop"]:
                    exit_price, exit_reason = position["stop"], "stop_loss"
                elif row["high"] >= position["target"]:
                    exit_price, exit_reason = position["target"], "take_profit"
            else:  # short: price rising hits the stop, price falling hits the target
                if row["high"] >= position["stop"]:
                    exit_price, exit_reason = position["stop"], "stop_loss"
                elif row["low"] <= position["target"]:
                    exit_price, exit_reason = position["target"], "take_profit"
            if exit_price is None and strategy_name == "daytrade" and (i - position["entry_idx"]) >= max_hold_bars:
                exit_price, exit_reason = price, "time_exit"

            if exit_price is not None:
                shares = position["shares"]
                if direction == "long":
                    revenue = shares * exit_price
                    realized_pnl = revenue - position["cost_basis"]
                    balance += revenue
                    exit_action, label = "sell", "SELL"
                else:
                    buyback_cost = shares * exit_price
                    realized_pnl = position["cost_basis"] - buyback_cost
                    balance -= buyback_cost
                    exit_action, label = "buy_to_cover", "BUY TO COVER"
                trade_log.append(
                    {
                        "time": row["time"],
                        "type": "exit",
                        "action": exit_action,
                        "price": float(exit_price),
                        "shares": float(shares),
                        "net_worth_after": float(balance),
                        "realized_pnl": float(realized_pnl),
                        "ticket": position["ticket"],
                        "rationale": f"[#{position['ticket']}] {label} — {exit_reason} at {exit_price:.5f} (entry {position['entry_price']:.5f}); P&L ${realized_pnl:,.2f}",
                    }
                )
                position = None
                last_exit_idx = i

        if position is None and (i - last_exit_idx) >= reentry_cooldown:
            sig = signal_fn(df, i, sr, position, **strategy_kwargs)
            action = sig.get("action")
            if action in ("buy", "sell_short"):
                stop, target = sig["stop"], sig["target"]
                direction = "long" if action == "buy" else "short"
                risk_per_share = (price - stop) if direction == "long" else (stop - price)
                if risk_per_share > 0 and balance > 0:
                    risk_amount = balance * risk_pct
                    # Long is capped by available cash; short is capped the same way as a
                    # simplified stand-in for margin capacity (no separate margin model here).
                    shares = min(risk_amount / risk_per_share, balance / price)
                    if shares > 0:
                        if direction == "long":
                            cost = shares * price
                            balance -= cost
                            cost_basis, log_action, label = cost, "buy", "BUY"
                        else:
                            proceeds = shares * price
                            balance += proceeds
                            cost_basis, log_action, label = proceeds, "sell_short", "SELL SHORT"
                        position = {
                            "entry_idx": i,
                            "entry_time": row["time"],
                            "entry_price": price,
                            "shares": shares,
                            "stop": stop,
                            "target": target,
                            "cost_basis": cost_basis,
                            "direction": direction,
                            "ticket": next_ticket,
                        }
                        next_ticket += 1
                        net_worth_after = balance + shares * price if direction == "long" else balance - shares * price
                        trade_log.append(
                            {
                                "time": row["time"],
                                "type": "entry",
                                "action": log_action,
                                "price": float(price),
                                "shares": float(shares),
                                "net_worth_after": float(net_worth_after),
                                "realized_pnl": None,
                                "ticket": position["ticket"],
                                "rationale": f"[#{position['ticket']}] {label} — {sig['reason']}; stop {stop:.5f}, target {target:.5f}",
                            }
                        )

        if position is None:
            net_worth = balance
        elif position["direction"] == "long":
            net_worth = balance + position["shares"] * price
        else:
            net_worth = balance - position["shares"] * price
        equity_curve.append({"time": row["time"], "net_worth": net_worth})

    summary = compute_summary(equity_curve, trade_log, initial_balance)
    return equity_curve, trade_log, summary, sr


def compute_summary(equity_curve, trade_log, initial_balance):
    """MT5 Strategy Tester-style summary fields (own implementation — see
    agents/strategy_backtester/backtest_engine.py for the sibling version
    used by the A3C walk-forward; intentionally not shared)."""
    exits = [t for t in trade_log if t["type"] == "exit"]
    wins = [t for t in exits if t["realized_pnl"] > 0]
    losses = [t for t in exits if t["realized_pnl"] <= 0]
    equity_values = np.array([e["net_worth"] for e in equity_curve])
    running_max = np.maximum.accumulate(equity_values)
    drawdown = (equity_values - running_max) / running_max * 100
    returns = np.diff(equity_values) / equity_values[:-1]
    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)) if len(returns) > 1 else 0.0

    net_profit = float(equity_curve[-1]["net_worth"] - initial_balance)
    gross_profit = float(sum(t["realized_pnl"] for t in wins))
    gross_loss = float(-sum(t["realized_pnl"] for t in losses))
    dd_dollar = running_max - equity_values
    balance_dd_absolute = float(max(0.0, initial_balance - equity_values.min()))
    balance_dd_maximal = float(dd_dollar.max()) if len(dd_dollar) else 0.0

    max_win_streak = max_win_streak_pnl = 0
    max_loss_streak = max_loss_streak_pnl = 0
    cur_streak = cur_streak_pnl = 0
    cur_is_win = None
    for t in exits:
        is_win = t["realized_pnl"] > 0
        if is_win == cur_is_win:
            cur_streak += 1
            cur_streak_pnl += t["realized_pnl"]
        else:
            cur_is_win, cur_streak, cur_streak_pnl = is_win, 1, t["realized_pnl"]
        if cur_is_win and cur_streak > max_win_streak:
            max_win_streak, max_win_streak_pnl = cur_streak, cur_streak_pnl
        if not cur_is_win and cur_streak > max_loss_streak:
            max_loss_streak, max_loss_streak_pnl = cur_streak, cur_streak_pnl

    return {
        "initial_balance": initial_balance,
        "final_net_worth": float(equity_curve[-1]["net_worth"]),
        "total_return_pct": float(net_profit / initial_balance * 100),
        "max_drawdown_pct": float(drawdown.min()) if len(drawdown) else 0.0,
        "sharpe_ratio": sharpe,
        "num_entries": sum(1 for t in trade_log if t["type"] == "entry"),
        "num_exits": len(exits),
        "win_rate_pct": float(len(wins) / len(exits) * 100) if exits else None,
        "total_realized_pnl": float(sum(t["realized_pnl"] for t in exits)),
        "net_profit": net_profit,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0),
        "expected_payoff": float(sum(t["realized_pnl"] for t in exits) / len(exits)) if exits else 0.0,
        "balance_dd_absolute": balance_dd_absolute,
        "balance_dd_maximal": balance_dd_maximal,
        "recovery_factor": float(net_profit / balance_dd_maximal) if balance_dd_maximal > 0 else 0.0,
        "profit_trades": len(wins),
        "loss_trades": len(losses),
        "largest_profit_trade": float(max((t["realized_pnl"] for t in wins), default=0.0)),
        "largest_loss_trade": float(min((t["realized_pnl"] for t in losses), default=0.0)),
        "average_profit_trade": float(np.mean([t["realized_pnl"] for t in wins])) if wins else 0.0,
        "average_loss_trade": float(np.mean([t["realized_pnl"] for t in losses])) if losses else 0.0,
        "max_consecutive_wins": max_win_streak,
        "max_consecutive_wins_pnl": float(max_win_streak_pnl),
        "max_consecutive_losses": max_loss_streak,
        "max_consecutive_losses_pnl": float(max_loss_streak_pnl),
    }


def open_file(path: Path):
    try:
        if platform.system() == "Windows":
            os.startfile(str(path))  # noqa: S606
        elif platform.system() == "Darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except OSError as exc:
        print(f"Could not auto-open {path}: {exc}", file=sys.stderr)


def render_report(report_path: Path, tag: str, df_window, equity_curve, trade_log, summary, sr_levels=None):
    """Self-contained interactive HTML report (Plotly, embedded JS — no
    internet needed to view): zoomable/pannable equity curve, net-profit
    curve, and price/RSI/MACD panel, each with hover tooltips carrying the
    trade rationale, plus the MT5 Strategy Tester-style summary and full
    trade log as plain HTML alongside it."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    report_path.parent.mkdir(parents=True, exist_ok=True)
    eq_times = [e["time"] for e in equity_curve]
    eq_values = [e["net_worth"] for e in equity_curve]
    entries = [t for t in trade_log if t["type"] == "entry"]
    exits = [t for t in trade_log if t["type"] == "exit"]

    # Marker shape follows the transaction (▲ = buy, ▼ = sell); color follows
    # position direction, so entry/exit x long/short all read as distinct.
    TRADE_MARKER_STYLES = {
        "entry_long": {"symbol": "triangle-up", "color": "#16a34a", "label": "Entry (long)"},
        "exit_long": {"symbol": "triangle-down", "color": "#dc2626", "label": "Exit (long)"},
        "entry_short": {"symbol": "triangle-down", "color": "#f97316", "label": "Entry (short)"},
        "exit_short": {"symbol": "triangle-up", "color": "#2563eb", "label": "Exit (short)"},
    }
    trades_by_kind = {
        "entry_long": [t for t in entries if t["action"] == "buy"],
        "entry_short": [t for t in entries if t["action"] == "sell_short"],
        "exit_long": [t for t in exits if t["action"] == "sell"],
        "exit_short": [t for t in exits if t["action"] == "buy_to_cover"],
    }
    seen_legend = set()

    def add_trade_markers(fig, row, y_key):
        for kind, style in TRADE_MARKER_STYLES.items():
            pts = trades_by_kind[kind]
            if not pts:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[t["time"] for t in pts], y=[t[y_key] for t in pts], mode="markers",
                    marker=dict(symbol=style["symbol"], color=style["color"], size=10, line=dict(width=1, color="#111827")),
                    name=style["label"], legendgroup=kind, showlegend=kind not in seen_legend,
                    hovertext=[f"{pd.Timestamp(t['time'])}<br>{t['rationale']}" for t in pts], hoverinfo="text",
                ),
                row=row, col=1,
            )
            seen_legend.add(kind)

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.22, 0.16, 0.30, 0.16, 0.16],
        subplot_titles=("Equity curve", "Net profit (realized, cumulative)", "Price, MAE mid, support/resistance", "RSI", "MACD"),
    )

    # Row 1: equity curve
    fig.add_trace(go.Scatter(x=eq_times, y=eq_values, mode="lines", line=dict(color="#1f2937", width=1.5), name="Net worth", showlegend=False, hovertemplate="%{x}<br>$%{y:,.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[eq_times[0], eq_times[-1]], y=[summary["initial_balance"]] * 2, mode="lines", line=dict(color="#9ca3af", width=1, dash="dash"), showlegend=False, hoverinfo="skip"), row=1, col=1)
    add_trade_markers(fig, 1, "net_worth_after")

    # Row 2: cumulative realized net profit (step chart), win/loss exits marked
    net_profit_times = [eq_times[0]]
    net_profit_values = [0.0]
    running = 0.0
    for t in exits:
        running += t["realized_pnl"]
        net_profit_times.append(t["time"])
        net_profit_values.append(running)
    fig.add_trace(go.Scatter(x=net_profit_times, y=net_profit_values, mode="lines", line=dict(color="#1f2937", width=1.5, shape="hv"), name="Net profit", showlegend=False, hovertemplate="%{x}<br>$%{y:,.2f}<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=[eq_times[0], eq_times[-1]], y=[0, 0], mode="lines", line=dict(color="#9ca3af", width=1, dash="dash"), showlegend=False, hoverinfo="skip"), row=2, col=1)
    win_pts = [(t, v) for t, v in zip(exits, net_profit_values[1:]) if t["realized_pnl"] > 0]
    loss_pts = [(t, v) for t, v in zip(exits, net_profit_values[1:]) if t["realized_pnl"] <= 0]
    if win_pts:
        fig.add_trace(go.Scatter(x=[t["time"] for t, _ in win_pts], y=[v for _, v in win_pts], mode="markers", marker=dict(color="#16a34a", size=7), name="Winning exit", hovertext=[f"{pd.Timestamp(t['time'])}<br>{t['rationale']}" for t, _ in win_pts], hoverinfo="text"), row=2, col=1)
    if loss_pts:
        fig.add_trace(go.Scatter(x=[t["time"] for t, _ in loss_pts], y=[v for _, v in loss_pts], mode="markers", marker=dict(color="#dc2626", size=7), name="Losing exit", hovertext=[f"{pd.Timestamp(t['time'])}<br>{t['rationale']}" for t, _ in loss_pts], hoverinfo="text"), row=2, col=1)

    # Row 3: price / MAE mid / confirmed support-resistance / trade markers
    x_min, x_max = df_window["time"].iloc[0], df_window["time"].iloc[-1]
    fig.add_trace(go.Scatter(x=df_window["time"], y=df_window["close"], mode="lines", line=dict(color="#1f2937", width=1.2), name="Close"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_window["time"], y=df_window["mae_mid"], mode="lines", line=dict(color="#2563eb", width=1), name="MAE mid"), row=3, col=1)
    if sr_levels:
        seen_sr = set()
        for lvl in sr_levels:
            is_support = lvl["kind"] == "swing_low"
            color, label = ("#16a34a", "Support") if is_support else ("#dc2626", "Resistance")
            fig.add_trace(
                go.Scatter(
                    x=[x_min, x_max], y=[lvl["price"]] * 2, mode="lines",
                    line=dict(color=color, width=1, dash="dot"), opacity=0.4,
                    name=label, legendgroup=label, showlegend=label not in seen_sr, hoverinfo="skip",
                ),
                row=3, col=1,
            )
            seen_sr.add(label)
    add_trade_markers(fig, 3, "price")

    # Row 4: RSI
    fig.add_trace(go.Scatter(x=df_window["time"], y=df_window["rsi"], mode="lines", line=dict(color="#7c3aed", width=1.2), name="RSI", showlegend=False), row=4, col=1)
    for level in (70, 30):
        fig.add_trace(go.Scatter(x=[x_min, x_max], y=[level, level], mode="lines", line=dict(color="#9ca3af", width=1, dash="dash"), showlegend=False, hoverinfo="skip"), row=4, col=1)
    fig.update_yaxes(range=[0, 100], row=4, col=1)

    # Row 5: MACD
    fig.add_trace(go.Scatter(x=df_window["time"], y=df_window["macd"], mode="lines", line=dict(color="#2563eb", width=1.2), name="MACD"), row=5, col=1)
    fig.add_trace(go.Scatter(x=df_window["time"], y=df_window["macd_signal"], mode="lines", line=dict(color="#dc2626", width=1.2), name="Signal"), row=5, col=1)

    fig.update_layout(
        title=f"{tag} — strategy backtest",
        height=1400,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0),
        margin=dict(t=110, b=40, l=60, r=30),
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Net worth ($)", row=1, col=1)
    fig.update_yaxes(title_text="Net profit ($)", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=3, col=1)
    fig.update_yaxes(title_text="RSI", row=4, col=1)
    fig.update_yaxes(title_text="MACD", row=5, col=1)
    fig.update_xaxes(title_text="Date", row=5, col=1)
    for r in (1, 2, 3, 4):
        fig.update_xaxes(showticklabels=False, row=r, col=1)

    plot_html = fig.to_html(full_html=False, include_plotlyjs=True, config={"scrollZoom": True})

    pf = summary["profit_factor"]
    pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
    summary_rows = [
        ("Initial Deposit", f"${summary['initial_balance']:,.2f}"),
        ("Final Balance", f"${summary['final_net_worth']:,.2f}"),
        ("Total Net Profit", f"${summary['net_profit']:,.2f} ({summary['total_return_pct']:+.2f}%)"),
        ("Gross Profit", f"${summary['gross_profit']:,.2f}"),
        ("Gross Loss", f"${summary['gross_loss']:,.2f}"),
        ("Profit Factor", pf_str),
        ("Expected Payoff", f"${summary['expected_payoff']:,.2f}"),
        ("Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}"),
        ("Recovery Factor", f"{summary['recovery_factor']:.2f}"),
        ("Balance Drawdown Absolute", f"${summary['balance_dd_absolute']:,.2f}"),
        ("Balance Drawdown Maximal", f"${summary['balance_dd_maximal']:,.2f} ({summary['max_drawdown_pct']:.2f}%)"),
        ("Total Positions Opened / Closed", f"{summary['num_entries']} / {summary['num_exits']}"),
        ("Profit Trades", f"{summary['profit_trades']}" + (f" ({summary['profit_trades'] / summary['num_exits'] * 100:.1f}%)" if summary["num_exits"] else "")),
        ("Loss Trades", f"{summary['loss_trades']}" + (f" ({summary['loss_trades'] / summary['num_exits'] * 100:.1f}%)" if summary["num_exits"] else "")),
        ("Largest Profit / Loss Trade", f"${summary['largest_profit_trade']:,.2f} / ${summary['largest_loss_trade']:,.2f}"),
        ("Average Profit / Loss Trade", f"${summary['average_profit_trade']:,.2f} / ${summary['average_loss_trade']:,.2f}"),
        ("Max Consecutive Wins", f"{summary['max_consecutive_wins']} (${summary['max_consecutive_wins_pnl']:,.2f})"),
        ("Max Consecutive Losses", f"{summary['max_consecutive_losses']} (${summary['max_consecutive_losses_pnl']:,.2f})"),
    ]
    summary_html = "\n".join(f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in summary_rows)

    def trade_row_html(t):
        pnl_str = "" if t["realized_pnl"] is None else f"${t['realized_pnl']:,.2f}"
        return (
            f"<tr><td>{t.get('ticket', '')}</td><td>{pd.Timestamp(t['time'])}</td><td>{t['type']}</td>"
            f"<td>{t['action']}</td><td>{t['price']:.5f}</td><td>{t['shares']:.4f}</td>"
            f"<td>{pnl_str}</td><td>{t['rationale']}</td></tr>"
        )

    trade_rows_html = "\n".join(trade_row_html(t) for t in trade_log)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{tag} — strategy backtest</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Arial, sans-serif; margin: 24px; color: #111827; background: #fff; }}
  h1 {{ font-size: 1.3rem; margin-bottom: 4px; }}
  h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
  .disclaimer {{ color: #6b7280; font-style: italic; font-size: 0.85rem; margin: 0 0 1.5rem; }}
  table.summary {{ border-collapse: collapse; margin-bottom: 1.5rem; }}
  table.summary th, table.summary td {{ text-align: left; padding: 3px 14px 3px 0; font-family: ui-monospace, Consolas, monospace; font-size: 0.85rem; }}
  table.summary th {{ font-weight: 600; color: #374151; white-space: nowrap; }}
  table.tradelog {{ border-collapse: collapse; width: 100%; font-size: 0.8rem; }}
  table.tradelog th, table.tradelog td {{ border-bottom: 1px solid #e5e7eb; padding: 4px 8px; text-align: left; }}
  table.tradelog th {{ background: #f9fafb; position: sticky; top: 0; }}
  .tradelog-wrap {{ max-height: 480px; overflow-y: auto; border: 1px solid #e5e7eb; margin-top: 0.5rem; }}
</style>
</head>
<body>
<h1>{tag} — strategy backtest</h1>
<p class="disclaimer">Rule-based decision-support signal from a limited historical window, not investment advice. No real or simulated order was ever sent.</p>
<table class="summary">{summary_html}</table>
{plot_html}
<h2>Trade log</h2>
<div class="tradelog-wrap">
<table class="tradelog">
<thead><tr><th>Ticket</th><th>Time</th><th>Type</th><th>Action</th><th>Price</th><th>Shares</th><th>Realized P&amp;L</th><th>Rationale</th></tr></thead>
<tbody>
{trade_rows_html}
</tbody>
</table>
</div>
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")
    print(f"Report saved to {report_path}")


def run(
    symbol, rate, start, end, strategy, initial_balance, risk_pct, sr_lookback,
    stop_atr_mult, target_rr, max_hold_bars, reentry_cooldown, no_open,
):
    data_start = compute_warmup_start(start, rate)
    paths = resolve_paths(symbol, rate, start, end, strategy, data_start=data_start)
    ensure_csv(paths["csv_path"], symbol, rate, data_start, end)

    # df_full spans [data_start, end] — the extra bars before `start` are only
    # to warm up indicators/support-resistance; trading itself begins at
    # start_idx (see run_strategy). df_window is the user-requested [start, end]
    # slice, used for the bar-count check and the report's price/indicator chart.
    df_full = load_asset_csv(paths["csv_path"])
    start_ts, end_ts = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    df_window = df_full[(df_full["time"] >= start_ts) & (df_full["time"] <= end_ts)].reset_index(drop=True)
    if len(df_window) < 30:
        raise ValueError(f"Only {len(df_window)} usable bars in [{start}, {end}] — not enough to run a strategy.")

    start_idx_matches = df_full.index[df_full["time"] >= start_ts]
    if len(start_idx_matches) == 0:
        raise ValueError(f"No bars at/after {start} in the loaded data — cannot start the run.")
    start_idx = int(start_idx_matches[0])

    kwargs = {}
    if strategy == "swing":
        kwargs = {"stop_atr_mult": stop_atr_mult if stop_atr_mult is not None else 2.0, "target_rr": target_rr if target_rr is not None else 2.5}
    elif strategy == "daytrade":
        kwargs = {"stop_atr_mult": stop_atr_mult if stop_atr_mult is not None else 0.75, "target_rr": target_rr if target_rr is not None else 1.5}

    equity_curve, trade_log, summary, sr = run_strategy(
        df_full, strategy, initial_balance=initial_balance, risk_pct=risk_pct,
        sr_lookback=sr_lookback, reentry_cooldown=reentry_cooldown, max_hold_bars=max_hold_bars,
        start_idx=start_idx, **kwargs,
    )
    render_report(paths["report_path"], paths["tag"], df_window, equity_curve, trade_log, summary, sr_levels=sr.available)

    if not no_open:
        open_file(paths["report_path"])

    print(json.dumps({"tag": paths["tag"], "report_path": str(paths["report_path"]), **summary}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--rate", default="D1")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--strategy", required=True, choices=STRATEGY_CHOICES)
    parser.add_argument("--balance", type=float, default=10000)
    parser.add_argument("--risk-pct", type=float, default=0.02, help="Fraction of balance risked per trade (sizes position by stop distance)")
    parser.add_argument("--sr-lookback", type=int, default=SR_LOOKBACK)
    parser.add_argument("--stop-atr-mult", type=float, default=None, help="Overrides the strategy's default stop distance (in ATR multiples)")
    parser.add_argument("--target-rr", type=float, default=None, help="Overrides the strategy's default reward:risk ratio")
    parser.add_argument("--max-hold-bars", type=int, default=24, help="daytrade only: forced exit after this many bars if neither stop nor target hit")
    parser.add_argument("--reentry-cooldown", type=int, default=3)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    run(
        args.symbol, args.rate, start, end, args.strategy, args.balance, args.risk_pct,
        args.sr_lookback, args.stop_atr_mult, args.target_rr, args.max_hold_bars,
        args.reentry_cooldown, args.no_open,
    )


if __name__ == "__main__":
    main()
