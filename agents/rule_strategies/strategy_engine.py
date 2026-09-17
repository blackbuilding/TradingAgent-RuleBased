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

Output mirrors agents/strategy_backtester's report shape (MT5 Strategy
Tester-style summary + equity curve + indicator panel + trade log PDF) for a
familiar read, but is an independent implementation — nothing here is
imported by, or imports from, strategy_backtester.

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
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = DATA_DIR / "reports"

RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
MAE_PERIOD, MAE_PCT = 20, 0.025
ATR_PERIOD = 14
SR_LOOKBACK = 10

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


def find_broader_csv(symbol: str, rate: str):
    best, best_end = None, ""
    for p in DATA_DIR.glob(f"{symbol}_{rate}_*_*.csv"):
        m = CSV_FILENAME_RE.match(p.name)
        if m and m.group("end") > best_end:
            best, best_end = p, m.group("end")
    if best:
        return best
    legacy = DATA_DIR / f"{symbol}_{rate}.csv"
    return legacy if legacy.exists() else None


def ensure_csv(csv_path: Path, symbol: str, rate: str, start: dt.date, end: dt.date):
    if csv_path.exists():
        return
    source = find_broader_csv(symbol, rate)
    if source is None:
        raise FileNotFoundError(
            f"No historical data at {csv_path}, and no broader {symbol} {rate} CSV to slice from. "
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


def resolve_paths(symbol: str, rate: str, start: dt.date, end: dt.date, strategy: str):
    range_tag = f"{symbol}_{rate}_{start:%Y%m%d}_{end:%Y%m%d}"
    tag = f"{range_tag}_{strategy}"
    return {
        "tag": tag,
        "csv_path": DATA_DIR / f"{range_tag}.csv",  # shared across strategies/agents, no strategy suffix
        "report_path": REPORTS_DIR / f"{tag}_report.pdf",
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

def _trend_entry_ok(df: pd.DataFrame, i: int) -> bool:
    if i < 1:
        return False
    row, prev = df.iloc[i], df.iloc[i - 1]
    crossed_up = prev["macd"] <= prev["macd_signal"] and row["macd"] > row["macd_signal"]
    return bool(crossed_up and 40 <= row["rsi"] <= 70 and row["close"] > row["mae_mid"])


def _trend_entry_short_ok(df: pd.DataFrame, i: int) -> bool:
    """Bearish mirror of _trend_entry_ok: MACD crosses below signal, RSI in a
    healthy (not yet oversold) 30-60 band, price below the MAE mid."""
    if i < 1:
        return False
    row, prev = df.iloc[i], df.iloc[i - 1]
    crossed_down = prev["macd"] >= prev["macd_signal"] and row["macd"] < row["macd_signal"]
    return bool(crossed_down and 30 <= row["rsi"] <= 60 and row["close"] < row["mae_mid"])


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
    **strategy_kwargs,
):
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy_name!r}; choices: {list(STRATEGIES)}")
    signal_fn = STRATEGIES[strategy_name]

    levels = find_swing_points(df, lookback=sr_lookback)
    sr = SRTracker(levels)

    balance = initial_balance
    position = None
    equity_curve = [{"time": df.iloc[0]["time"], "net_worth": initial_balance}]
    trade_log = []
    last_exit_idx = -reentry_cooldown

    for i in range(len(df)):
        sr.advance_to(i)
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
                        "rationale": f"{label} — {exit_reason} at {exit_price:.5f} (entry {position['entry_price']:.5f}); P&L ${realized_pnl:,.2f}",
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
                        }
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
                                "rationale": f"{label} — {sig['reason']}; stop {stop:.5f}, target {target:.5f}",
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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    report_path.parent.mkdir(parents=True, exist_ok=True)
    eq_times = [e["time"] for e in equity_curve]
    eq_values = [e["net_worth"] for e in equity_curve]
    entries = [t for t in trade_log if t["type"] == "entry"]
    exits = [t for t in trade_log if t["type"] == "exit"]

    with PdfPages(report_path) as pdf:
        # Page 1: MT5 Strategy Tester-style summary + equity curve with entry/exit markers
        fig, (ax_summary, ax_eq) = plt.subplots(2, 1, figsize=(11, 8.5), gridspec_kw={"height_ratios": [1.3, 2]})
        fig.suptitle(f"{tag} — strategy backtest", fontsize=14, fontweight="bold")
        ax_summary.axis("off")
        pf = summary["profit_factor"]
        pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
        left_lines = [
            f"Initial Deposit:        \\${summary['initial_balance']:,.2f}",
            f"Final Balance:          \\${summary['final_net_worth']:,.2f}",
            f"Total Net Profit:       \\${summary['net_profit']:,.2f}  ({summary['total_return_pct']:+.2f}%)",
            f"Gross Profit:           \\${summary['gross_profit']:,.2f}",
            f"Gross Loss:             \\${summary['gross_loss']:,.2f}",
            f"Profit Factor:          {pf_str}",
            f"Expected Payoff:        \\${summary['expected_payoff']:,.2f}",
            f"Sharpe Ratio:           {summary['sharpe_ratio']:.2f}",
            f"Recovery Factor:        {summary['recovery_factor']:.2f}",
        ]
        right_lines = [
            f"Balance Drawdown Absolute: \\${summary['balance_dd_absolute']:,.2f}",
            f"Balance Drawdown Maximal:  \\${summary['balance_dd_maximal']:,.2f}  ({summary['max_drawdown_pct']:.2f}%)",
            "",
            f"Total Positions Opened: {summary['num_entries']}    Closed: {summary['num_exits']}",
            f"Profit Trades: {summary['profit_trades']}"
            + (f" ({summary['profit_trades'] / summary['num_exits'] * 100:.1f}%)" if summary["num_exits"] else ""),
            f"Loss Trades:   {summary['loss_trades']}"
            + (f" ({summary['loss_trades'] / summary['num_exits'] * 100:.1f}%)" if summary["num_exits"] else ""),
            f"Largest Profit Trade: \\${summary['largest_profit_trade']:,.2f}    Largest Loss Trade: \\${summary['largest_loss_trade']:,.2f}",
            f"Average Profit Trade: \\${summary['average_profit_trade']:,.2f}    Average Loss Trade: \\${summary['average_loss_trade']:,.2f}",
            f"Max Consecutive Wins:   {summary['max_consecutive_wins']} (\\${summary['max_consecutive_wins_pnl']:,.2f})",
            f"Max Consecutive Losses: {summary['max_consecutive_losses']} (\\${summary['max_consecutive_losses_pnl']:,.2f})",
        ]
        ax_summary.text(0, 1.0, "\n".join(left_lines), va="top", fontsize=9.5, family="monospace")
        ax_summary.text(0.52, 1.0, "\n".join(right_lines), va="top", fontsize=9.5, family="monospace")
        ax_summary.text(
            0,
            -0.08,
            "Rule-based decision-support signal from a limited historical window, not investment advice.",
            va="top",
            fontsize=8,
            style="italic",
            color="#6b7280",
            transform=ax_summary.transAxes,
        )

        ax_eq.plot(eq_times, eq_values, linewidth=1.2, color="#1f2937", zorder=1)
        ax_eq.axhline(summary["initial_balance"], color="#9ca3af", linestyle="--", linewidth=0.8)
        if entries:
            ax_eq.scatter(
                [t["time"] for t in entries], [t["net_worth_after"] for t in entries],
                marker="^", color="#16a34a", s=45, zorder=2, label="Entry (buy)",
            )
        if exits:
            ax_eq.scatter(
                [t["time"] for t in exits], [t["net_worth_after"] for t in exits],
                marker="v", color="#dc2626", s=45, zorder=2, label="Exit (sell)",
            )
        ax_eq.set_title("Equity curve")
        ax_eq.set_ylabel("Net worth ($)")
        ax_eq.legend(loc="upper left", fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: price/RSI/MACD with entry/exit markers and confirmed S/R levels
        fig, (ax_price, ax_rsi, ax_macd) = plt.subplots(
            3, 1, figsize=(11, 8.5), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]}
        )
        fig.suptitle(f"{tag} — indicators, support/resistance, entries/exits", fontsize=13, fontweight="bold")
        ax_price.plot(df_window["time"], df_window["close"], color="#1f2937", linewidth=1.0, label="Close")
        ax_price.plot(df_window["time"], df_window["mae_mid"], color="#2563eb", linewidth=0.9, label="MAE mid")
        if sr_levels:
            seen_labels = set()
            for lvl in sr_levels:
                is_support = lvl["kind"] == "swing_low"
                color = "#16a34a" if is_support else "#dc2626"
                label = "Support" if is_support else "Resistance"
                ax_price.axhline(
                    lvl["price"], color=color, linewidth=0.6, alpha=0.35, linestyle=":",
                    label=label if label not in seen_labels else None,
                )
                seen_labels.add(label)
        if entries:
            ax_price.scatter([t["time"] for t in entries], [t["price"] for t in entries], marker="^", color="#16a34a", s=40, zorder=3)
        if exits:
            ax_price.scatter([t["time"] for t in exits], [t["price"] for t in exits], marker="v", color="#dc2626", s=40, zorder=3)
        ax_price.legend(loc="upper left", fontsize=8, frameon=False)
        ax_price.set_ylabel("Price")

        ax_rsi.plot(df_window["time"], df_window["rsi"], color="#7c3aed", linewidth=1)
        ax_rsi.axhline(70, color="#9ca3af", linewidth=0.7, linestyle="--")
        ax_rsi.axhline(30, color="#9ca3af", linewidth=0.7, linestyle="--")
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_ylabel("RSI")

        ax_macd.plot(df_window["time"], df_window["macd"], color="#2563eb", linewidth=1, label="MACD")
        ax_macd.plot(df_window["time"], df_window["macd_signal"], color="#dc2626", linewidth=1, label="Signal")
        ax_macd.legend(loc="upper left", fontsize=8, frameon=False)
        ax_macd.set_ylabel("MACD")
        ax_macd.set_xlabel("Date")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3+: trade log, one rationale line per entry/exit, paginated
        rows_per_page = 28
        for i in range(0, len(trade_log), rows_per_page):
            chunk = trade_log[i : i + rows_per_page]
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")
            if i == 0:
                ax.set_title(f"{tag} — trade log", fontsize=13, fontweight="bold", loc="left")
            text_lines = [f"{pd.Timestamp(t['time'])}  {t['rationale']}" for t in chunk]
            ax.text(0, 1, "\n".join(text_lines), va="top", fontsize=8.5, family="monospace")
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Report saved to {report_path}")


def run(
    symbol, rate, start, end, strategy, initial_balance, risk_pct, sr_lookback,
    stop_atr_mult, target_rr, max_hold_bars, reentry_cooldown, no_open,
):
    paths = resolve_paths(symbol, rate, start, end, strategy)
    ensure_csv(paths["csv_path"], symbol, rate, start, end)

    df_full = load_asset_csv(paths["csv_path"])
    start_ts, end_ts = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    df_window = df_full[(df_full["time"] >= start_ts) & (df_full["time"] <= end_ts)].reset_index(drop=True)
    if len(df_window) < 30:
        raise ValueError(f"Only {len(df_window)} usable bars in [{start}, {end}] — not enough to run a strategy.")

    kwargs = {}
    if strategy == "swing":
        kwargs = {"stop_atr_mult": stop_atr_mult if stop_atr_mult is not None else 2.0, "target_rr": target_rr if target_rr is not None else 2.5}
    elif strategy == "daytrade":
        kwargs = {"stop_atr_mult": stop_atr_mult if stop_atr_mult is not None else 0.75, "target_rr": target_rr if target_rr is not None else 1.5}

    equity_curve, trade_log, summary, sr = run_strategy(
        df_window, strategy, initial_balance=initial_balance, risk_pct=risk_pct,
        sr_lookback=sr_lookback, reentry_cooldown=reentry_cooldown, max_hold_bars=max_hold_bars, **kwargs,
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
