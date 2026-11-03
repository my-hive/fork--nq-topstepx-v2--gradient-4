"""Causal VWAP-displacement momentum confirmed by aggressive order flow."""
import numpy as np
import pandas as pd

SESSION_START = 8 * 3600 + 30 * 60
EXIT = 14 * 3600 + 30 * 60
LOOKBACK = 1800
ENTRY_Z = 2.0
EFF_MIN = 0.05
DELTA_MIN = 0.05
RISK = 300.0


def signals(df: pd.DataFrame) -> np.ndarray:
    """Follow efficient VWAP extremes only with same-side aggressor flow."""
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
        series = pd.Series(c)
        sd = series.rolling(LOOKBACK // 6, min_periods=LOOKBACK // 6).std().to_numpy()
        path = series.diff().abs().rolling(LOOKBACK, min_periods=LOOKBACK).sum()
        efficiency = (series.diff(LOOKBACK).abs() / path).to_numpy()
        delta = pd.Series(df.buy_vol.values[rows] - df.sell_vol.values[rows])
        delta_sum = delta.rolling(LOOKBACK // 6, min_periods=LOOKBACK // 6).sum()
        volume_sum = pd.Series(v).rolling(LOOKBACK // 6, min_periods=LOOKBACK // 6).sum()
        imbalance = (delta_sum / volume_sum.replace(0, np.nan)).to_numpy()
        pos, stop = 0, 0.0
        for k in range(len(rows)):
            if not np.isfinite(sd[k]) or not np.isfinite(efficiency[k]) or sd[k] <= 0:
                pos = 0
            elif pos != 0:
                if (pos > 0 and c[k] <= stop) or (pos < 0 and c[k] >= stop):
                    pos = 0
            elif efficiency[k] >= EFF_MIN and abs(c[k] - vwap[k]) >= ENTRY_Z * sd[k]:
                direction = 1 if c[k] > vwap[k] else -1
                if direction * imbalance[k] >= DELTA_MIN:
                    dist = ENTRY_Z * sd[k]
                    qty = int(min(50, RISK / (2.0 * max(dist, 0.25))))
                    pos = direction * qty
                    stop = vwap[k]
            out[rows[k]] = pos
    return out
