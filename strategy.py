"""Cash-open auction continuation after range formation.
Six gates: range duration, stop fraction, reward multiple, risk, cutoff, one entry.
"""
import numpy as np
from numba import njit

RANGE_SECONDS = 1800
STOP_FRACTION = 0.5
REWARD = 1.0
RISK = 300.0
CUTOFF = 54600

@njit
def _targets(session, t, high, low, close):
    out = np.zeros(len(close), np.int64)
    hi, lo, pos, stop, target = -np.inf, np.inf, 0, 0.0, 0.0
    used = False
    for i in range(len(close)):
        if i == 0 or session[i] != session[i - 1]:
            hi, lo, pos, used, escaped = -np.inf, np.inf, 0, False, 0
        if t[i] < 30600:
            continue
        if t[i] < 30600 + RANGE_SECONDS:
            hi, lo = max(hi, high[i]), min(lo, low[i])
            continue
        if t[i] >= CUTOFF:
            pos = 0
        elif pos:
            if (pos > 0 and (close[i] <= stop or close[i] >= target)) or (pos < 0 and (close[i] >= stop or close[i] <= target)):
                pos = 0
        elif not used and np.isfinite(hi) and hi > lo:
            direction = 1 if close[i] > hi else -1 if close[i] < lo else 0
            if direction:
                dist = max(0.25, (hi - lo) * STOP_FRACTION)
                qty = min(50, int(RISK / (2.0 * dist)))
                pos = direction * qty
                stop = close[i] - direction * dist
                target = close[i] + direction * dist * REWARD
                used = True
        out[i] = pos
    return out

def signals(df):
    """Past/current rows only; targets fill next open, never range-future data."""
    return _targets(df.session.factorize(sort=False)[0], df.ct_sec.to_numpy(), df.high.to_numpy(), df.low.to_numpy(), df.close.to_numpy())
