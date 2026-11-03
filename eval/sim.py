"""TopstepX 50K combine simulator, v2. Do not modify.

strategy.signals(df) -> int array, target MNQ contracts in [-50, 50] per row, decided on that row's close.
Fills at next row's open +/- slippage. Rules: $3,000 target, $1,000 daily loss (flatten, day over),
$2,000 trailing max loss on EOD high (locks at $50,000), flat by 15:10 CT, consistency
(best day < 50% of profit), min 2 days.

v2 scoring (anti-overfit): sessions are cut into contiguous FOLDS; every rolling 30-session
attempt inside every fold is scored (no skipped starts), at STRESSED costs (2 ticks, $1.50/side).
score = min over folds of pass rate + 0.01 * speed. The last fold is the validation fold.
"""
import argparse, os, re, sys
import numpy as np
import pandas as pd
from numba import njit

# ponytail: MNQ priced off NQ 1s data; fees are estimates, confirm vs TopstepX all-in
PT_VALUE, MAX_CT = 2.0, 50
BASE = (0.25, 0.50)       # slippage pts (1 tick), $/contract/side
STRESS = (0.50, 1.50)     # 2 ticks, 3x commission: headline score uses this
START, TARGET, DLL, MLL = 50_000.0, 3_000.0, 1_000.0, 2_000.0
FLAT_SEC, WINDOW = 15 * 3600 + 10 * 60, 30          # 15:10 CT; attempt length in sessions
FOLDS = [(0, 50), (50, 100), (100, 150), (150, None)]   # last = validation (to end of data)
BANNED = re.compile(r"dayofweek|weekday|strftime|toordinal|isocalendar|\.month\b|\.day\b|holiday|20\d\d-\d\d-\d\d",
                    re.I)


@njit(cache=True)
def run_days(sess, ct_sec, o, h, l, c, target, n_sess, slip, comm):
    day_pnl, day_min, trades = np.zeros(n_sess), np.zeros(n_sess), np.zeros(n_sess, np.int64)
    pos, cash, locked = 0, 0.0, False
    for i in range(len(c)):
        s = sess[i]
        if i == 0 or s != sess[i - 1]:
            pos, cash, locked = 0, 0.0, False
        last = i == len(c) - 1 or sess[i + 1] != s
        want = 0 if locked or (ct_sec[i] >= FLAT_SEC and ct_sec[i] < 17 * 3600) else target[i - 1] if i > 0 and sess[i - 1] == s else 0
        want = max(-MAX_CT, min(MAX_CT, want))
        if want != pos:
            d = want - pos
            px = o[i] + slip * (1 if d > 0 else -1)
            cash -= d * px * PT_VALUE + abs(d) * comm
            pos = want
            trades[s] += abs(d)
        worst = cash + pos * (l[i] if pos > 0 else h[i]) * PT_VALUE
        day_min[s] = min(day_min[s], worst)
        if worst <= -DLL and not locked:          # DLL hit: flatten at the limit, day over
            cash = -DLL - abs(pos) * (slip * PT_VALUE + comm)
            trades[s] += abs(pos)
            pos, locked = 0, True
        if last and pos != 0:                     # session end: flatten at close
            cash += pos * (c[i] - slip * np.sign(pos)) * PT_VALUE - abs(pos) * comm
            trades[s] += abs(pos)
            pos = 0
        day_pnl[s] = cash + pos * c[i] * PT_VALUE
    return day_pnl, day_min, trades


def attempts(day_pnl, day_min, lo, hi):
    """Rolling combine attempts starting at each session in [lo, hi - WINDOW]. Returns (passed, days)."""
    res = []
    for st in range(lo, hi - WINDOW + 1):
        bal, peak_eod, best, days, out = START, START, 0.0, 0, (False, WINDOW)
        for s in range(st, st + WINDOW):
            thr = min(peak_eod - MLL, START)
            if bal + day_min[s] <= thr:
                break
            bal += day_pnl[s]
            days += 1
            best, peak_eod = max(best, day_pnl[s]), max(peak_eod, bal)
            profit = bal - START
            if profit >= TARGET and days >= 2 and best < 0.5 * profit:
                out = (True, days)
                break
        res.append(out)
    return res


def folds(n):
    return [(a, n if b is None else min(b, n)) for a, b in FOLDS if a < n]


def score(day_pnl, day_min):
    """min over folds of all-start pass rate, + speed bonus. Returns (score, per-fold rates, pass days)."""
    per = [attempts(day_pnl, day_min, a, b) for a, b in folds(len(day_pnl))]
    rates = [np.mean([p for p, _ in a]) if a else 0.0 for a in per]
    passed = [d for a in per for p, d in a if p]
    speed = 1 - np.mean(passed) / WINDOW if passed else 0.0
    return min(rates) + 0.01 * speed, rates, passed


def lookahead_check(signals, df, rng, n=4):
    """Signals on a prefix must equal signals on full data for that prefix."""
    full = np.asarray(signals(df))
    for cut in rng.integers(len(df) // 10, len(df), n):
        part = np.asarray(signals(df.iloc[:cut].copy()))
        if not np.array_equal(part[:cut - 1], full[:cut - 1]):
            bad = int(np.argmax(part[:cut - 1] != full[:cut - 1]))
            sys.exit(f"[eval] LOOKAHEAD: signal at row {bad} changes when future data after row {cut} is removed")
    return full


def lint(path):
    """Protocol rule: no calendar / date-specific gates."""
    bad = [(i + 1, ln.strip()) for i, ln in enumerate(open(path)) if BANNED.search(ln)]
    if bad:
        sys.exit("[eval] CALENDAR GATE banned by program.md:\n" + "\n".join(f"  line {i}: {t}" for i, t in bad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser(os.environ.get("NQ_DATA", "~/hive-data/nq/train.parquet")))
    a = ap.parse_args()
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    sys.path.insert(0, root)
    lint(os.path.join(root, "strategy.py"))
    from strategy import signals

    df = pd.read_parquet(a.data)
    sess = pd.factorize(df.session, sort=True)[0]
    u = df.session.unique()
    rng = np.random.default_rng(0)
    for lo in (0, 100):   # early and mid windows
        lookahead_check(signals, df[df.session.isin(u[lo:lo + 15])], rng)
    target = np.nan_to_num(np.asarray(signals(df), dtype=float)).astype(np.int64)
    assert len(target) == len(df), "signals() must return one value per row"
    args = (sess, df.ct_sec.values, df.open.values, df.high.values, df.low.values, df.close.values, target, sess.max() + 1)
    dp, dm, tr = run_days(*args, *STRESS)
    bp, bm, _ = run_days(*args, *BASE)
    sc, rates, passed = score(dp, dm)
    _, brates, _ = score(bp, bm)
    fl = folds(len(dp))
    print("---")
    print(f"score:          {sc:.4f}   (stressed costs, min over folds)")
    for k, ((x, y), r, br) in enumerate(zip(fl, rates, brates)):
        tag = "validation" if k == len(fl) - 1 else "dev"
        print(f"fold{k + 1} {tag:10s} sessions {x}-{y}: pass {r:.3f} (base-cost {br:.3f}) pnl {dp[x:y].sum():.0f}")
    print(f"avg_days_pass:  {np.mean(passed) if passed else float('nan'):.1f}")
    print(f"sessions:       {len(dp)}")
    print(f"total_pnl:      {dp.sum():.0f}   (base-cost {bp.sum():.0f})")
    print(f"win_days:       {(dp > 0).mean():.3f}")
    print(f"worst_day:      {dp.min():.0f}")
    print(f"contracts_traded: {tr.sum()}")


if __name__ == "__main__":
    main()
