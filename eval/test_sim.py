"""Self-check for sim.py: python3 eval/test_sim.py"""
import numpy as np
from sim import run_days, attempts, folds, lint, PT_VALUE, BASE
SLIP, COMM = BASE

# one session, 4 bars: go long 1 at bar1 open (100), hold, session-end flatten at close 110
o = np.array([100., 100., 105., 110.]); c = np.array([100., 104., 109., 110.])
h, l = c + 1, c - 1
t = np.array([0, 1, 1, 0])  # target decided on close of row i, filled at row i+1 open
pnl, dmin, tr = run_days(np.zeros(4, np.int64), np.full(4, 9 * 3600), o, h, l, c, t, 1, SLIP, COMM)
exp = (110 - SLIP - (105 + SLIP)) * PT_VALUE - 2 * COMM
assert abs(pnl[0] - exp) < 1e-6, (pnl[0], exp)
assert tr[0] == 2

# costs are parameters: stressed run must earn strictly less
pnl_s, _, _ = run_days(np.zeros(4, np.int64), np.full(4, 9 * 3600), o, h, l, c, t, 1, 0.5, 1.5)
assert pnl_s[0] < pnl[0]

# DLL: long 50 MNQ into a 20-pt drop must lock at about -$1,000
o2 = np.array([100., 100., 80., 80.]); c2 = np.array([100., 99., 80., 80.])
pnl2, _, _ = run_days(np.zeros(4, np.int64), np.full(4, 9 * 3600), o2, c2 + 0.5, c2 - 0.5, c2,
                      np.array([50, 50, 50, 50]), 1, SLIP, COMM)
assert -1_100 < pnl2[0] <= -1_000, pnl2[0]

# combine: +1,000/day passes on day 3; one $3,000 day fails consistency; steady losses bust MLL
day = np.full(40, 1_000.); assert attempts(day, np.zeros(40), 0, 40)[0] == (True, 3)
spike = np.zeros(40); spike[0] = 3_000.; assert attempts(spike, np.zeros(40), 0, 40)[0][0] is False
dd = np.full(40, -700.); assert attempts(dd, dd, 0, 40)[0] == (False, 30)

# folds cover every session once, no gaps (v1 skipped middle starts)
f = folds(215); assert f[0][0] == 0 and f[-1][1] == 215 and all(f[i][1] == f[i + 1][0] for i in range(len(f) - 1))

# calendar-gate lint rejects date/weekday logic
import tempfile, os
for bad in ("x = d.dt.dayofweek", "if s == '2026-01-02':", "m = s.month"):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(bad + "\n")
    try:
        lint(fh.name); raise AssertionError(f"lint missed: {bad}")
    except SystemExit:
        pass
    os.unlink(fh.name)
print("sim self-check ok")
