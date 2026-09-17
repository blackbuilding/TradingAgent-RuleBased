# Rule Strategies

Three classic, rule-based entry/exit strategies — **support/resistance**,
**swing trading**, and **day trading** — each with real stop-loss/take-profit
logic. This is a standalone sibling to
[`asset_analysis`](../asset_analysis/README.md) (A3C model) and
[`strategy_backtester`](../strategy_backtester/README.md) (A3C walk-forward):
**no imports from either**, and neither of those is modified by this agent.
Indicator formulas (RSI/MACD/MAE) are duplicated locally, matching the
convention already used across this repo's agents.

This exists because the A3C model has no explicit exit logic at all — walking
it forward can produce dozens or hundreds of buys and zero sells (see the
EURUSD H1 case: 1,236 entries, 0 exits), since its "sell" probability never
meaningfully diverges from ~33% regardless of position state. These three
strategies always have a defined stop and target set at entry.

## The three strategies

| | Entry | Stop | Target | Notes |
|---|---|---|---|---|
| `support_resistance` | Price within 0.3% of a confirmed swing-low support, RSI not overbought | Just below support | Next confirmed resistance (or 2x risk if none) | Mean-reversion, long only |
| `swing` (long) | MACD crosses above signal, RSI 40-70, price above MAE mid | Entry − 2x ATR (or nearest support if tighter) | 1:2.5 reward:risk (or nearer resistance) | Trend continuation, multi-day/week hold |
| `swing` (short) | MACD crosses below signal, RSI 30-60, price below MAE mid | Entry + 2x ATR (or nearest resistance if tighter) | 1:2.5 reward:risk (or nearer support) | Mirror of the long side — `swing` is the only one of the three that trades both directions |
| `daytrade` | Same long entry as `swing` (no short side) | Entry − 0.75x ATR (tight) | 1:1.5 reward:risk | Tighter risk than swing; forced time exit after `--max-hold-bars` (default 24) if neither stop nor target hits — approximates a same-session constraint on 24/5 FX data |

Support/resistance levels are fractal swing points (`find_swing_points`): a
bar is a swing high/low if it's the extreme within a `lookback`-bar window on
each side. A level isn't "confirmed" until `lookback` bars after it forms, and
walk-forward code (`SRTracker`) only ever exposes levels confirmed as of the
current bar — no lookahead bias.

**Position sizing is risk-based**, not a flat % of cash: `shares = (balance *
risk_pct) / stop_distance_per_share`, so a trade's size reflects the
distance to its own stop (the standard "risk a fixed % of capital per trade"
practice), capped by available cash.

## Setup

Same MT5 MCP server and `.env` as the other agents:
```
pip install -r agents/rule_strategies/requirements.txt
```

## Naming convention

CSVs are shared with the other agents (same symbol/rate/range file is reused
if already fetched); reports get a strategy suffix so they never collide
with an A3C report for the same range:
```
data/<SYMBOL>_<RATE>_<START>_<END>.csv                                  # historical bars (shared)
data/reports/<SYMBOL>_<RATE>_<START>_<END>_<STRATEGY>_report.pdf        # this agent's output
```
e.g. `EURUSD_H1_20250101_20260101_daytrade`.

## Running

This script has no MT5 access itself — fetching missing data is the calling
agent's job (see `.claude/agents/rule-strategies.md`), though it will slice a
broader already-fetched CSV for the same symbol/rate if the exact range isn't
on disk yet. Once the CSV exists:
```
python3 agents/rule_strategies/strategy_engine.py run \
  --symbol EURUSD --rate H1 --start 2025-01-01 --end 2026-01-01 --strategy daytrade
```
No model to train — every run is immediate. Add `--no-open` to skip
auto-opening the PDF. Tunable: `--risk-pct` (default 0.02), `--sr-lookback`
(default 10), `--stop-atr-mult` / `--target-rr` (override the strategy's
defaults above), `--max-hold-bars` (daytrade only, default 24),
`--reentry-cooldown` (default 3 bars after an exit).

## Output

Same report shape as `strategy_backtester`'s (for a familiar read, but its
own independent implementation): page 1 is an MT5 Strategy Tester-style
summary (Net Profit, Gross Profit/Loss, Profit Factor, Expected Payoff,
Balance Drawdown Absolute/Maximal, Sharpe, Recovery Factor, profit/loss trade
counts, largest/average win-loss, max consecutive streaks) plus the equity
curve; page 2 is price/RSI/MACD with support/resistance levels and
entry/exit markers; page 3+ is a trade-by-trade rationale log.

## Source

MCP server: [Qoyyuum/mcp-metatrader5-server](https://github.com/Qoyyuum/mcp-metatrader5-server).
