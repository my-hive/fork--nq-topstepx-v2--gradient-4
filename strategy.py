"""Causal VWAP liquidity provision with volatility-normalized risk."""
import numpy as np
import pandas as pd

SESSION_START = 8 * 3600 + 30 * 60
TRADE_START = 9 * 3600
EXIT = 14 * 3600 + 30 * 60
LOOKBACK = 300
ENTRY_Z = 2.0
RISK = 300.0


def signals(df: pd.DataFrame) -> np.ndarray:
    """Fade causal VWAP extremes; stop one volatility unit beyond entry."""
    out = np.zeros(len(df), dtype=np.int64)
    for _, idx in df.groupby("session", sort=False).indices.items():
        t = df.ct_sec.values[idx]
        use = np.flatnonzero((t >= SESSION_START) & (t < EXIT))
        if len(use) <= LOOKBACK:
            continue
        rows = idx[use]
        c = df.close.values[rows]
        h, l = df.high.values[rows], df.low.values[rows]
        v = np.maximum(df.volume.values[rows], 0.0)
        price = (h + l + c) / 3.0
        cv = np.cumsum(v)
        vwap = np.divide(np.cumsum(price * v), cv, out=price.copy(), where=cv > 0)
        sd = pd.Series(c).rolling(LOOKBACK, min_periods=LOOKBACK).std().to_numpy()
        pos, armed, stop, target = 0, True, 0.0, 0.0
        for k in range(len(rows)):
            if t[use[k]] < TRADE_START or not np.isfinite(sd[k]) or sd[k] <= 0:
                pos = 0
            elif pos != 0:
                if (pos > 0 and (c[k] <= stop or c[k] >= target)) or (pos < 0 and (c[k] >= stop or c[k] <= target)):
                    hit_target = (pos > 0 and c[k] >= target) or (pos < 0 and c[k] <= target)
                    pos = 0
                    armed = hit_target
            elif not armed:
                armed = abs(c[k] - vwap[k]) < sd[k]
            elif abs(c[k] - vwap[k]) >= ENTRY_Z * sd[k]:
                direction = -1 if c[k] > vwap[k] else 1
                dist = sd[k]
                qty = int(min(50, RISK / (2.0 * max(dist, 0.25))))
                pos = direction * qty
                stop = c[k] - direction * dist
                target = vwap[k]
            out[rows[k]] = pos
    return out
