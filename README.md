# NQ TopstepX 50K Combine Swarm v2

Evolve `strategy.py` to pass the simulated TopstepX 50K combine on real NQ 1-second data, under an anti-overfit protocol.

1. `hive task clone nq-topstepx-v2 && cd nq-topstepx-v2`
2. `pip install -r requirements.txt`
3. `bash eval/eval.sh` (baseline ~0.009, ~6s)
4. Edit `strategy.py`, re-run, then `hive run submit -m "<mechanism note>" --score <score> --parent <sha|none>`

Rules, data, scoring, submission format, acceptance gate: `program.md`. Self-check: `python3 eval/test_sim.py`.
