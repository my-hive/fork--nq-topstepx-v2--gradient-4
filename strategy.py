"""gts-2: causal intraday range-breakout momentum with volatility-scaled stop sizing."""
import numpy as np
import pandas as pd

START = 8 * 3600 + 30 * 60      # RTH open, CT
LAST_ENTRY = 15 * 3600          # no new entries after
FLAT = 15 * 3600 + 9 * 60       # flatten before 15:10 cutoff
N_MIN = 10                      # gate 1: breakout lookback, completed minutes
K_STOP = 0.3                    # gate 2: stop = K * ATR(30m) * sqrt(N)/2
RISK = 100.0                    # gate 3: $ risk per trade (<= 300 rule)
MAX_TRADES = 1                  # gate 4: entries per session
ATR_MIN = 30


def signals(df: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(df), dtype=np.int64)
    sec = df.ct_sec.values
    for _, idx in df.groupby("session", sort=False).indices.items():
        t = sec[idx]
        key = np.where(t >= 61200, t - 86400, t)          # session-ordered seconds
        minute = key // 60
        # completed-minute OHLC, built only from rows already seen
        mb = pd.DataFrame({"m": minute, "h": df.high.values[idx], "l": df.low.values[idx],
                           "c": df.close.values[idx]}).groupby("m", sort=True).agg(h=("h", "max"), l=("l", "min"), c=("c", "last"))
        mids = mb.index.values
        h, l, c = mb.h.values, mb.l.values, mb.c.values
        pc = np.r_[c[0], c[:-1]]
        tr = pd.Series(np.maximum(h, pc) - np.minimum(l, pc)).rolling(ATR_MIN).mean().values
        hh = pd.Series(h).rolling(N_MIN).max().values
        ll = pd.Series(l).rolling(N_MIN).min().values
        # for each row, index of the last fully completed minute (minute < current minute)
        done = np.searchsorted(mids, minute, side="left") - 1
        close = df.close.values[idx]
        pos, stop, trades = 0, 0.0, 0
        for k in np.flatnonzero((t >= START) & (t < FLAT)):
            j = done[k]
            new_min = k == 0 or minute[k] != minute[k - 1]
            if pos:
                # stop judged on completed-minute close: ignores 1s noise spikes
                if new_min and j >= 0 and ((pos > 0 and c[j] <= stop) or (pos < 0 and c[j] >= stop)):
                    pos = 0
            elif trades < MAX_TRADES and t[k] < LAST_ENTRY and new_min:
                if j >= 1 and np.isfinite(tr[j]) and np.isfinite(hh[j - 1]):
                    d = 1 if c[j] > hh[j - 1] else -1 if c[j] < ll[j - 1] else 0
                    if d:
                        dist = max(K_STOP * tr[j] * np.sqrt(N_MIN) / 2, 1.0)
                        pos = d * int(min(50, RISK / (2.0 * dist)))
                        stop = close[k] - d * dist
                        trades += 1
            out[idx[k]] = pos
    return out
