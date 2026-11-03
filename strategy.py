"""gts-2: ensemble of RTH range-breakout lookbacks (one entry per sleeve), vol-scaled stops,
breakeven after 2R, portfolio daily profit lock."""
import numpy as np
import pandas as pd

START = 8 * 3600 + 30 * 60
FLAT = 15 * 3600 + 9 * 60
LOOKBACKS = (10, 15, 20)        # gate 1: breakout lookbacks (minutes), equal-risk sleeves
K_STOP = 0.9                    # gate 2: stop = K * ATR30m * sqrt(N)/2, judged on minute close
RISK = 125.0                    # gate 3: total $ risk across sleeves
LOCK = 1400.0                   # gate 4: bank day at this open+realized $ (best day < 50% of $3K)
ATR_MIN = 30                    # gate 5: ATR window (minutes)
BE = 2                          # gate 6: after +BE*stop-distance move, stop to entry (protect trailing DD)


def signals(df):
    out = np.zeros(len(df), dtype=np.int64)
    sec = df.ct_sec.values
    for _, idx in df.groupby("session", sort=False).indices.items():
        t = sec[idx]
        minute = np.where(t >= 61200, t - 86400, t) // 60
        mb = pd.DataFrame({"m": minute, "h": df.high.values[idx], "l": df.low.values[idx],
                           "c": df.close.values[idx]}).groupby("m", sort=True).agg(h=("h", "max"), l=("l", "min"), c=("c", "last"))
        mids = mb.index.values
        h, l, c = mb.h.values, mb.l.values, mb.c.values
        pc = np.r_[c[0], c[:-1]]
        tr = pd.Series(np.maximum(h, pc) - np.minimum(l, pc)).rolling(ATR_MIN).mean().values
        hh = [pd.Series(h).rolling(n).max().values for n in LOOKBACKS]
        ll = [pd.Series(l).rolling(n).min().values for n in LOOKBACKS]
        done = np.searchsorted(mids, minute, side="left") - 1
        close = df.close.values[idx]
        ns = len(LOOKBACKS)
        frac = np.zeros(ns); ent = np.zeros(ns); dst = np.zeros(ns); stop = np.zeros(ns); trades = np.zeros(ns, int)
        pos, locked = 0, False
        cash = 0.0  # realized $ (close-priced, pre-cost) for the profit lock
        for k in np.flatnonzero((t >= START) & (t < FLAT)):
            j = done[k]
            if locked:
                break
            if k == 0 or minute[k] != minute[k - 1]:
                if j >= 1 and pos and cash + pos * c[j] * 2.0 >= LOCK:
                    locked = True
                    frac[:] = 0
                for s in range(ns):
                    if locked or j < 1:
                        break
                    if frac[s] and BE and (c[j] - ent[s]) * np.sign(frac[s]) >= BE * dst[s]:
                        stop[s] = max(stop[s], ent[s]) if frac[s] > 0 else min(stop[s], ent[s])
                    if frac[s]:
                        if (frac[s] > 0 and c[j] <= stop[s]) or (frac[s] < 0 and c[j] >= stop[s]):
                            frac[s] = 0
                    elif not trades[s] and np.isfinite(tr[j]) and np.isfinite(hh[s][j - 1]):
                        d = 1 if c[j] > hh[s][j - 1] else -1 if c[j] < ll[s][j - 1] else 0
                        if d:
                            dist = max(K_STOP * tr[j] * np.sqrt(LOOKBACKS[s]) / 2, 1.0)
                            frac[s] = d * RISK / ns / (2.0 * dist)
                            stop[s] = close[k] - d * dist
                            ent[s] = close[k]; dst[s] = dist
                            trades[s] = 1
                new = int(np.clip(np.trunc(frac.sum()), -50, 50))
                if new != pos:
                    cash -= (new - pos) * close[k] * 2.0
                    pos = new
            out[idx[k]] = pos
    return out
