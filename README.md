# TradingAgent-RuleBased

Rule-based (non-ML) entry/exit trading strategies for MetaTrader 5, run
through a Claude Code subagent. Moved out of
[TradingAgent](https://github.com/blackbuilding/TradingAgent) to stand on
its own, since it has no code or model dependency on that repo's A3C
reinforcement-learning pipeline — everything here is hand-written technical
rules (MACD/RSI/MAE/ATR, fractal support/resistance), not a trained model.

See [`agents/rule_strategies/README.md`](agents/rule_strategies/README.md)
for the three strategies (support/resistance, swing, day trade), their
entry/stop/target rules, and how to run them.

## Setup

1. Install Python dependencies:
   ```
   pip install -r agents/rule_strategies/requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your MT5 account credentials.
3. Open this folder in Claude Code — the `rule-strategies` subagent
   (`.claude/agents/rule-strategies.md`) is available automatically, backed
   by the `mt5` MCP server configured in `.mcp.json`
   ([Qoyyuum/mcp-metatrader5-server](https://github.com/Qoyyuum/mcp-metatrader5-server)).

## Running directly

```
python3 agents/rule_strategies/strategy_engine.py run \
  --symbol EURUSD --rate H1 --start 2025-01-01 --end 2026-01-01 --strategy daytrade
```

Fetches missing historical data from MT5 (via the subagent) or slices a
broader already-fetched CSV, then walks the range bar by bar and writes an
interactive HTML report (MT5 Strategy Tester-style summary, a zoomable
equity curve, net-profit curve, and price/RSI/MACD panel with
support/resistance levels, plus the full trade-by-trade rationale log) to
`output/`.
