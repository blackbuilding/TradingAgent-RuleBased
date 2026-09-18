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
| `swing` (long) | MACD crosses above signal *and still holds one bar later*, MAE mid net-rising over the last `TREND_LOOKBACK` (10) bars, RSI 40-70, price above MAE mid | Entry − 2x ATR (or nearest support if tighter) | 1:2.5 reward:risk (or nearer resistance) | Trend continuation, multi-day/week hold |
| `swing` (short) | MACD crosses below signal *and still holds one bar later*, MAE mid net-falling over the last `TREND_LOOKBACK` (10) bars, RSI 30-60, price below MAE mid | Entry + 2x ATR (or nearest resistance if tighter) | 1:2.5 reward:risk (or nearer support) | Mirror of the long side — `swing` is the only one of the three that trades both directions |
| `daytrade` | Same long entry as `swing` (no short side) | Entry − 0.75x ATR (tight) | 1:1.5 reward:risk | Tighter risk than swing; forced time exit after `--max-hold-bars` (default 24) if neither stop nor target hits — approximates a same-session constraint on 24/5 FX data |

Two confirmation filters sit on top of the raw MACD crossover
(`_trend_entry_ok`/`_trend_entry_short_ok`), each trading a slightly worse/
later entry for fewer bad ones:
- **One-bar confirmation**: the cross must still hold one bar after it first
  fired, filtering whipsaws that flip back almost immediately.
- **Long-term trend filter**: MAE mid must actually be higher (long) or lower
  (short) than it was `TREND_LOOKBACK` (10) bars ago — not just favor the
  entry on this one bar — so the crossover fires within a standing trend
  rather than a choppy/sideways MAE mid. It's a lagging check (confirms a
  trend that's already underway, won't catch one at its very start).

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
input/<SYMBOL>_<RATE>_<DATA_START>_<END>.csv                             # historical bars (shared, fetched input)
output/<SYMBOL>_<RATE>_<START>_<END>_<STRATEGY>_report.pdf               # this agent's output
```
e.g. `EURUSD_H1_20250101_20260101_daytrade`. `<START>`/`<END>` in the report
name are the requested range; the input CSV's `<DATA_START>` is `<START>`
minus a warmup buffer (see below) — it's a different, earlier date.

## Running

This script has no MT5 access itself — fetching missing data is the calling
agent's job (see `.claude/agents/rule-strategies.md`), though it will slice a
broader already-fetched CSV for the same symbol/rate if the exact range isn't
on disk yet.

It always fetches/loads more history than `--start`/`--end` request: an extra
`2 * MAE period` (40) bars before `--start` (translated to a generous
calendar-day buffer per `--rate`, e.g. ~20 days for H1) so RSI/MACD/MAE/ATR
are warmed up with real data by the time trading begins — otherwise the
first bars of every run would see NaN or freshly-seeded indicators instead of
proper history, silently skewing decisions right at the start of the window.
Trading itself (entries, the equity curve, the trade log) still starts
exactly at `--start`, not earlier — the buffer only feeds indicator/support-
resistance warmup, it's never itself traded or reported. Once the CSV
exists:
```
python3 agents/rule_strategies/strategy_engine.py run \
  --symbol EURUSD --rate H1 --start 2025-01-01 --end 2026-01-01 --strategy daytrade
```
No model to train — every run is immediate. Add `--no-open` to skip
auto-opening the report. Tunable: `--risk-pct` (default 0.02), `--sr-lookback`
(default 10), `--stop-atr-mult` / `--target-rr` (override the strategy's
defaults above), `--max-hold-bars` (daytrade only, default 24),
`--reentry-cooldown` (default 3 bars after an exit).

## Output

A single self-contained interactive HTML file (Plotly, JS embedded — opens
in any browser, works fully offline, no server needed): an MT5 Strategy
Tester-style summary table (Net Profit, Gross Profit/Loss, Profit Factor,
Expected Payoff, Balance Drawdown Absolute/Maximal, Sharpe, Recovery Factor,
profit/loss trade counts, largest/average win-loss, max consecutive
streaks), then a zoomable/pannable chart stack — equity curve, cumulative
realized net-profit curve, and price/RSI/MACD with support/resistance
levels — and the full trade log as a scrollable table below it. Hovering
any entry/exit marker or trade-log row shows that trade's full rationale.

Entry/exit markers use marker shape for the transaction (▲ buy, ▼ sell) and
color for position direction, so all four read as distinct on both the
equity curve and the price chart: entry long (green ▲), exit long (red ▼),
entry short (orange ▼), exit short (blue ▲). Legend items are click-to-toggle,
consistent with normal Plotly chart interaction.

## Source

MCP server: [Qoyyuum/mcp-metatrader5-server](https://github.com/Qoyyuum/mcp-metatrader5-server).
