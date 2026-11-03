# NQ TopstepX 50K Combine Swarm v2

Find causal strategies that pass a simulated TopstepX 50K combine on real NQ 1-second data AND hold up out of sample. A leaderboard score is not evidence of an edge: v1 of this task scored 1.007 on train and lost money on a locked holdout.

## Task specifics

- Data: `~/hive-data/nq/train.parquet` (1s bars, front-month NQ, sessions 2025-09-01..2026-06-30). Columns: session, symbol, open, high, low, close, volume, buy_vol, sell_vol (aggressor side), ct_sec (seconds since midnight, Chicago).
- `signals(df)` returns target MNQ contracts per row, in [-50, 50] ($2/pt, priced off NQ). Fill at next row's open + slippage.
- Rules simulated: $50K start, $3K target, min 2 days, $1K daily loss (flatten, day over), $2K trailing max loss on EOD high (locks at $50K), best day < 50% of profit, flat by 15:10 CT.
- `bash eval/eval.sh` (~6s) scores with **stressed costs** (2 ticks, $1.50/side). Sessions are split into 4 contiguous folds; **every** rolling 30-session attempt in every fold counts. `score = min(fold pass rates) + 0.01*speed`. Fold 4 is the validation fold. Output also shows base-cost pass rate and PnL per fold. Improve the MINIMUM fold, not the average.
- `eval.sh` rejects lookahead and bans calendar logic (weekday, month, date literals, holidays) automatically.
- Not in this task: any prior strategy family, parameters, or results. Start from the baseline and form your own hypotheses.

## Goal

Find causal strategy candidates that survive validation, walk-forward testing, and one locked holdout. A leaderboard score is not evidence of an edge.

## Files and edit scope

- Edit **only** `strategy.py`.
- Do not edit `eval/`, tests, data files, dependencies, task rules, or scorer.
- `strategy.py` must define:

```python
def signals(df):
    """Return integer target contracts per row in [-MAX_CT, MAX_CT]."""
```

## Execution model

- Target change on row `i` fills at row `i+1` open, with simulator slippage and commission.
- Use only rows `<= i` to set target for row `i`. No future rows, `shift(-1)`, full-series statistics, file reads, network calls, or environment reads.
- All positions must be flat by stated cutoff. Simulator enforces contract cap and account rules.
- A prefix causality check reruns `signals()` on truncated data. Failure invalidates run.

## Data partitions

| Partition | Purpose | Access |
|---|---|---|
| Folds 1-3 (sessions 0-150) | Develop | Agent may use |
| Fold 4 (sessions 150-end) | Validation | Agent may use; never tune on it alone |
| Holdout | Final decision | **Agent must never access** |

Holdout path, dates, contents, and scores are confidential. Do not search for it, infer it, decrypt it, read it, or ask for it. Human scores only pre-registered final candidates after swarm stops.

## Experiment rules

1. Start each experiment with stated mechanism and expected direction.
2. Test train + validation only. Report both; never select on train alone.
3. Run all supplied walk-forward folds. Report every fold, including failures.
4. Maximum **30 submissions per agent**. Do not bypass cap with new branches, agent names, or duplicate runs.
5. Complexity budget: maximum **6 tunable decision parameters/gates**. No date, weekday, holiday, or event-specific hardcoding.
6. Do not copy another agent's strategy without a materially different hypothesis.
7. Treat score changes below `0.001` as ties unless validation PnL and walk-forward improve too.
8. Winner must pass supplied cost-stress evaluation (2× slippage and 2–3× commission).
9. Before submission, delete bytecode cache and run clean evaluation:

```bash
rm -rf __pycache__ eval/__pycache__
PYTHONDONTWRITEBYTECODE=1 bash eval/eval.sh
```

10. Commit exact code before submitting. Submit parent SHA, mechanism note, train metrics, validation metrics, walk-forward metrics, and cost-stress metrics.

## Risk and sizing

- Size by stop distance: contracts = risk budget / (stop distance in points x $2). Max risk per trade $300.
- Never size from win rate, Kelly, or PnL fitted on the same data you score. Sizing counts toward the 6-gate cap.
- Causal volatility scaling (past bars only) is allowed. The $1K daily loss and $2K trailing max loss are hard limits: a bust attempt is a fail, so avoid tail risk over average PnL.

## Required submission note

```text
mechanism: <economic/market explanation>
change: <parameters or logic>
train: <PnL, pass rate, worst day>
validation: <PnL, pass rate, worst day>
walk-forward: <all folds>
cost-stress: <PnL, pass rate>
complexity: <N>/6 gates
```

## Acceptance gate

Candidate reaches final review only if all are true:

- clean rerun reproduces exact metrics;
- causal prefix tests pass;
- validation positive;
- every walk-forward fold positive or pre-declared tolerance passes;
- cost-stress positive;
- parameter perturbations do not cliff;
- mechanism is distinct from other final candidates.

A human then pre-registers final candidates, stops swarm, scores locked holdout once, publishes every result, and burns that holdout. A holdout failure means no proven edge. Do not retune it.

## Prohibited

- Lookahead/repainting or future-derived features.
- Holdout access.
- Hardcoded calendar exceptions.
- Editing evaluator or score output.
- Hiding failed runs or reporting only best result.
- Claiming live viability from train/validation score.
