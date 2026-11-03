"""Causal opening-range breakout confirmed by aggressive order flow."""
import numpy as np
import pandas as pd

OR_START = 8 * 3600 + 30 * 60
OR_END = 8 * 3600 + 45 * 60
EXIT = 14 * 3600 + 30 * 60
DELTA_BARS = 60
DELTA_MIN = 0.10
RISK = 300.0

def signals(df: pd.DataFrame) -> np.ndarray:
    """Take one range break only when trailing aggressor flow confirms it."""
    out = np.zeros(len(df), dtype=np.int64)
    for _, idx in df.groupby("session", sort=False).indices.items():
        t = df.ct_sec.values[idx]
        c = df.close.values[idx]
        h, l = df.high.values[idx], df.low.values[idx]
        opening = (t >= OR_START) & (t < OR_END)
        if not opening.any():
            continue
        hi, lo = h[opening].max(), l[opening].min()
        delta = df.buy_vol.values[idx] - df.sell_vol.values[idx]
        volume = np.maximum(df.volume.values[idx], 1)
        d = pd.Series(delta).rolling(DELTA_BARS, min_periods=DELTA_BARS).sum().to_numpy()
        v = pd.Series(volume).rolling(DELTA_BARS, min_periods=DELTA_BARS).sum().to_numpy()
        imbalance = d / v
        pos, done, stop = 0, False, 0.0
        for k in range(len(idx)):
            if t[k] < OR_END or t[k] >= EXIT:
                pos = 0
            elif pos == 0 and not done:
                direction = 1 if c[k] > hi else -1 if c[k] < lo else 0
                if direction and direction * imbalance[k] >= DELTA_MIN:
                    stop = lo if direction > 0 else hi
                    dist = max(abs(c[k] - stop), 0.25)
                    pos = direction * int(min(50, RISK / (2.0 * dist)))
                    done = True
            elif pos != 0 and ((pos > 0 and c[k] <= stop) or (pos < 0 and c[k] >= stop)):
                pos = 0
            out[idx[k]] = pos
    return out
