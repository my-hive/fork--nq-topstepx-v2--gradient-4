"""Failed auction rotation; stop beyond rejected excursion, target range value.
Six gates: range duration, buffer, target fraction, risk, cutoff, one entry.
"""
import numpy as np
from numba import njit

RANGE_SECONDS = 1800
BUFFER = 0.1
TARGET_FRACTION = 0.5
RISK = 300.0
CUTOFF = 54600

@njit
def _targets(session, t, high, low, close):
    out = np.zeros(len(close), np.int64)
    hi, lo, pos, stop, target = -np.inf, np.inf, 0, 0.0, 0.0
    used, escaped, extreme = False, 0, 0.0
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
            direction = -escaped if lo < close[i] < hi else 0
            if close[i] > hi:
                extreme = max(extreme, high[i]) if escaped == 1 else high[i]
                escaped = 1
            elif close[i] < lo:
                extreme = min(extreme, low[i]) if escaped == -1 else low[i]
                escaped = -1
            if direction:
                stop = extreme - direction * BUFFER * (hi - lo)
                dist = max(0.25, direction * (close[i] - stop))
                qty = min(50, int(RISK / (2.0 * dist)))
                pos = direction * qty
                target = close[i] + direction * TARGET_FRACTION * ((hi - close[i]) if direction > 0 else (close[i] - lo))
                used = True
        out[i] = pos
    return out

def signals(df):
    """Targets use current and past auction prices; next-open fill handled externally."""
    return _targets(df.session.factorize(sort=False)[0], df.ct_sec.to_numpy(), df.high.to_numpy(), df.low.to_numpy(), df.close.to_numpy())
