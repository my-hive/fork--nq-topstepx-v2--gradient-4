"""Causal opening-range fade with stop-distance sizing."""
import numpy as np
import pandas as pd

OR_START = 8 * 3600 + 30 * 60
OR_END = 8 * 3600 + 45 * 60
EXIT = 14 * 3600 + 30 * 60
RISK = 300.0


def signals(df: pd.DataFrame) -> np.ndarray:
    """Fade the first opening-range break; stop across the range."""
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
                if cc[k] > hi or cc[k] < lo:
                    stop = lo if cc[k] > hi else hi
                    dist = max(abs(cc[k] - stop), 0.25)
                    pos = int(min(50, RISK / (2.0 * dist)))
                    pos = -pos if cc[k] > hi else pos
                    done = True
            elif (pos > 0 and cc[k] > hi) or (pos < 0 and cc[k] < lo):
                pos = 0
            out[idx[k]] = pos
    return out
