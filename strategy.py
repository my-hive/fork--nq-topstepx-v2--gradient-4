"""Causal failed-opening-auction reversion with stop-distance sizing."""
import numpy as np
import pandas as pd

OR_START = 8 * 3600 + 30 * 60
OR_END = 8 * 3600 + 45 * 60
EXIT = 14 * 3600 + 30 * 60
RISK = 300.0
MAX_QTY = 10


def signals(df: pd.DataFrame) -> np.ndarray:
    """Fade a range break only after it closes back inside the range."""
    out = np.zeros(len(df), dtype=np.int64)
    t = df.ct_sec.values
    c, h, l = df.close.values, df.high.values, df.low.values
    for _, idx in df.groupby("session", sort=False).indices.items():
        tt, cc, hh, ll = t[idx], c[idx], h[idx], l[idx]
        rng = (tt >= OR_START) & (tt < OR_END)
        if not rng.any():
            continue
        hi, lo = hh[rng].max(), ll[rng].min()
        pos, direction, done = 0, 0, False
        stop = target = extreme = 0.0
        for k in range(len(idx)):
            if tt[k] < OR_END or tt[k] >= EXIT:
                pos = 0
            elif pos == 0 and not done:
                if direction == 0:
                    if cc[k] > hi:
                        direction, extreme = 1, hh[k]
                    elif cc[k] < lo:
                        direction, extreme = -1, ll[k]
                elif direction > 0:
                    extreme = max(extreme, hh[k])
                    if cc[k] < hi:
                        entry, stop, target = cc[k], extreme, lo
                        dist = max(stop - entry, 0.25)
                        pos = -int(min(MAX_QTY, RISK / (2.0 * dist)))
                        done = True
                else:
                    extreme = min(extreme, ll[k])
                    if cc[k] > lo:
                        entry, stop, target = cc[k], extreme, hi
                        dist = max(entry - stop, 0.25)
                        pos = int(min(MAX_QTY, RISK / (2.0 * dist)))
                        done = True
            elif pos != 0 and ((pos > 0 and (cc[k] <= stop or cc[k] >= target)) or
                               (pos < 0 and (cc[k] >= stop or cc[k] <= target))):
                pos = 0
            out[idx[k]] = pos
    return out
