"""NQ strategy for the TopstepX 50K combine. This is the only file you edit.

signals(df) returns one integer per row: target position in MNQ contracts (-50..50, + long, - short; $2/pt).
The decision on row i is filled at row i+1's open (plus slippage). Use only data at or before row i.
The eval re-runs signals() on truncated data and rejects any lookahead.

df columns (1-second bars, front-month NQ prices, which MNQ tracks; one row per second that traded):
  session (date, CME day 17:00-16:00 CT), symbol, open, high, low, close, volume,
  buy_vol, sell_vol (aggressor-side volume), ct_sec (seconds since midnight, Chicago time)
"""
import numpy as np
import pandas as pd

CONTRACTS = 10  # 10 MNQ = 1 NQ of risk
OR_START, OR_END, EXIT = 8 * 3600 + 30 * 60, 8 * 3600 + 45 * 60, 14 * 3600 + 30 * 60  # CT


def signals(df: pd.DataFrame) -> np.ndarray:
    """Baseline: 15-min opening range breakout, one trade per session, stop at the other side."""
    out = np.zeros(len(df), dtype=np.int64)
    t, c = df.ct_sec.values, df.close.values
    for _, idx in df.groupby("session", sort=False).indices.items():
        tt, cc = t[idx], c[idx]
        rng = (tt >= OR_START) & (tt < OR_END)
        if not rng.any():
            continue
        hi, lo = cc[rng].max(), cc[rng].min()
        pos, done = 0, False
        for k in range(len(idx)):
            if tt[k] < OR_END or tt[k] >= EXIT:
                pos = 0
            elif pos == 0 and not done:
                if cc[k] > hi:
                    pos, done = CONTRACTS, True
                elif cc[k] < lo:
                    pos, done = -CONTRACTS, True
            elif (pos > 0 and cc[k] < lo) or (pos < 0 and cc[k] > hi):
                pos = 0
            out[idx[k]] = pos
    return out
