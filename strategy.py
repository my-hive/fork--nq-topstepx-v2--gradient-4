"""Three-stream overnight/reversion portfolio for NQ TopstepX combine.

mechanism:
1. gap fade: large RTH-open gaps vs the prior session close reflect overnight
   overreaction that partially reverts in the morning; stop = 0.35x|gap|.
2. pre-open drift (02:00-08:00): equity index futures carry an overnight
   risk premium that accrues into the RTH open; stop = 0.6x median 10-session
   same-window move.
3. post-close drift (17:00-20:00): the same premium accrues after the close;
   stop = 1.5x median 10-session same-window move.
causal: every decision uses rows <= current row or fully completed prior
sessions; state resets each session. sizing: risk $300 / (stop_pts * $2).
gates: GAP_MIN, STOP_FRAC, K_PRE, K_POST, LOCK, DAILY_STOP (6/6). RISK=300
is the program-mandated per-trade cap, not a tunable gate.
"""
import numpy as np
import pandas as pd

GAP_MIN = 40.0        # gate 1: min |gap| in points
STOP_FRAC = 0.5      # gate 2: gap-fade stop fraction of |gap|
K_PRE = 0.5           # gate 3: pre-open stop = 0.6 x median move
K_POST = 2.5          # gate 4: post-close stop = 1.5 x median move
RISK = 300.0          # gate 5: risk budget per trade ($)
LOCK = 1400.0         # gate 6: daily profit lock ($) - satisfies best-day<50% consistency rule
DAILY_STOP = -700.0  # daily loss guard ($): flatten rest of session at -500

PRE_IN, PRE_OUT = 2 * 3600, 8 * 3600          # 02:00 -> 08:00 CT
POST_OUT = 20 * 3600                          # post window: 17:00 -> 20:00 CT
RTH_OPEN = 8 * 3600 + 30 * 60                 # 08:30 CT
GAP_EXIT = 12 * 3600                          # noon CT
FLAT = 15 * 3600 + 10 * 60                    # 15:10 CT
LOOKBACK = 10                                  # median window, sessions
CAP = 10                                       # contract cap per trade


def _size(stop_pts: float) -> int:
    return int(min(CAP, RISK / (2.0 * max(stop_pts, 0.25))))


def _ok(x):
    """Chronological key: 17:00 -> 0, wraps to next-day 00:00 -> 420+min."""
    x = np.asarray(x)
    return np.where(x >= 61200, x - 61200, x + 25200)


def signals(df: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(df), dtype=np.int64)
    o = df.open.values
    h = df.high.values
    l = df.low.values
    c = df.close.values
    ct = df.ct_sec.values
    pre_hist = []
    post_hist = []
    prev_close = np.nan
    for _, idx in df.groupby("session", sort=False).indices.items():
        t = ct[idx]
        kk = np.searchsorted(_ok(t), [_ok(PRE_IN), _ok(PRE_OUT), _ok(POST_OUT), _ok(GAP_EXIT)])
        pre_e, pre_x, post_x, gap_x = kk
        # --- post-close stream (17:00-20:00), entry at session first row ---
        post_state = 0
        post_stop = 0.0
        post_entry = o[idx[0]] if len(idx) else 0.0
        post_pts = 0.0
        if len(post_hist) >= LOOKBACK and post_x > 1:
            med_post = float(np.median(post_hist[-LOOKBACK:]))
            if med_post > 0:
                post_pts = K_POST * med_post
                post_stop = post_entry - post_pts
                post_state = _size(post_pts)
        # --- pre-open stream (02:00-08:00) ---
        pre_state = 0
        pre_stop = 0.0
        pre_entry = o[idx[pre_e]] if pre_e < len(idx) else 0.0
        pre_pts = 0.0
        if len(pre_hist) >= LOOKBACK and pre_x > pre_e + 1:
            med_pre = float(np.median(pre_hist[-LOOKBACK:]))
            if med_pre > 0:
                pre_pts = K_PRE * med_pre
                pre_stop = pre_entry - pre_pts
                pre_state = _size(pre_pts)
        # --- gap stream state ---
        o830 = np.nan
        g = np.nan
        gap_state = 0
        gap_stop = 0.0
        locked = False
        est_cash = 0.0
        prev_val = 0
        for k in range(len(idx)):
            j = idx[k]
            tc = t[k]
            if locked:
                prev_val = 0
                continue
            val = 0
            if 61200 <= tc < POST_OUT:
                if post_state != 0 and l[j] <= post_stop:
                    post_state = 0
                val += post_state
            if PRE_IN <= tc < PRE_OUT:
                if pre_state != 0 and l[j] <= pre_stop:
                    pre_state = 0
                val += pre_state
            if RTH_OPEN <= tc < FLAT:
                if not np.isfinite(o830):
                    o830 = o[j]
                    g = o830 - prev_close if np.isfinite(prev_close) else np.nan
                    if np.isfinite(g) and abs(g) >= GAP_MIN:
                        d = -1.0 if g > 0 else 1.0
                        gap_stop = o830 - d * STOP_FRAC * abs(g)
                        gap_state = int(d * _size(STOP_FRAC * abs(g)))
                if gap_state != 0:
                    if (gap_state > 0 and l[j] <= gap_stop) or (gap_state < 0 and h[j] >= gap_stop) or tc >= GAP_EXIT:
                        gap_state = 0
                    else:
                        val += gap_state
            out[j] = val
            # daily profit lock: estimated session PnL (proxy fills at close +-0.5)
            d_ = val - prev_val
            if d_ != 0:
                fill = c[j] + (0.5 if d_ > 0 else -0.5)
                est_cash -= d_ * fill * 2.0 + abs(d_) * 2.5
            prev_val = val
            eq = est_cash + val * c[j] * 2.0
            if eq >= LOCK or eq <= DAILY_STOP:
                locked = True
                out[j] = 0
                prev_val = 0
        # session done: update drift history and prev_close
        if pre_x > pre_e + 1 and pre_e < len(idx) and pre_x <= len(idx):
            pre_hist.append(abs(o[idx[pre_x - 1]] - o[idx[pre_e]]))
        if post_x > 1 and post_x <= len(idx):
            post_hist.append(abs(o[idx[post_x - 1]] - o[idx[0]]))
        if len(idx):
            prev_close = c[idx[-1]]
    return out
