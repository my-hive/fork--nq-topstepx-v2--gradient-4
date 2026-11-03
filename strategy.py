"""Causal opening-range continuation with fixed fractional stop risk."""
import numpy as np
import pandas as pd

OR_START = 8 * 3600 + 30 * 60
OR_END = 8 * 3600 + 45 * 60
EXIT = 14 * 3600 + 30 * 60
RISK = 300.0


def signals(df: pd.DataFrame) -> np.ndarray:
    """Trade first range break; size to opposite range and exit at 1R."""
    out = np.zeros(len(df), dtype=np.int64)
    t, c = df.ct_sec.values, df.close.values
    for _, idx in df.groupby("session", sort=False).indices.items():
        tt, cc = t[idx], c[idx]
        rng = (tt >= OR_START) & (tt < OR_END)
        if not rng.any():
            continue
        hi = df.high.values[idx][rng].max()
        lo = df.low.values[idx][rng].min()
        pos, done, stop, target = 0, False, 0.0, 0.0
        for k in range(len(idx)):
            if tt[k] < OR_END or tt[k] >= EXIT:
                pos = 0
            elif pos == 0 and not done:
                if cc[k] > hi or cc[k] < lo:
                    direction = 1 if cc[k] > hi else -1
                    entry = cc[k]
                    stop = lo if direction > 0 else hi
                    dist = max(abs(entry - stop), 0.25)
                    qty = int(min(50, RISK / (2.0 * dist)))
                    pos = direction * qty
                    target = entry + direction * dist
                    done = True
            elif pos != 0 and ((pos > 0 and (cc[k] <= stop or cc[k] >= target)) or
                               (pos < 0 and (cc[k] >= stop or cc[k] <= target))):
                pos = 0
            out[idx[k]] = pos
    return out
