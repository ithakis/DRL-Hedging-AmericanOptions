# Speed check: E3a, 2% cost, seed 1, 2,000 episodes (val every 500, no test)

Run on the Mac, one core each (OMP/MKL/NUMBA threads = 1), one run at a time, no other jobs.
Base = commit 7644745, speed = this branch (1488cc9 + 6b474af).

| Code | s / 1,000 episodes | train s | env | kernel | learn | validation |
|---|---|---|---|---|---|---|
| base (7644745) | 177.4 | 354.8 | - | - | - | 3.0 |
| speed branch | 173.5 | 347.0 | 0.8 | 15.5 | 330.2 | 2.2 |

Speed-up: 2%. The learner step is 95% of the time, so the faster table lookup and kernel
(2.8x on the kernel alone, see `test_speed_equivalence.py`) change almost nothing overall.

Validation VaR95 / CVaR95 / losses at episodes 512, 1008, 1504, 2000 are bit-identical between base and speed
(VaR95 32.80, 32.83, 33.93, 33.64; max abs diff 0). Files: `timing_*_results.json`, `timing_*_validation.csv`.

## Profile (speed branch, cProfile, `profile_cumulative.txt`, `profile_tottime.txt`)

Of 329 s total, `torch_d4pg/distributional.py:_qh_kernel` (the numba quantile-Huber loss) takes 261 s (80%).
It is called 7,376 times, 35 ms per call. With the 9-node kernel target each call has
B x M = 256 x 9 = 2,304 rows and 99 x 99 quantile pairs per row, about 22.6 M pair terms
(~1.5 ns per pair: the loop is already tight, but the work is O(N^2) per row and runs on one thread).
Next items: network forwards ~26 s (sigmoid 8.7 s, layer_norm 7.7 s), table lookups 15 s, backward 5.5 s.

Possible next speed-ups (not done here):
- Quantile-Huber loss in O(N log N) per row: sort the 99 targets once, then for each online quantile
  the sum over targets is a piecewise quadratic in the gap, so prefix sums plus a binary search give it
  (exact, same numbers up to rounding). Expected about 5-10x on 80% of the time.
- Or compute the loss once per batch row instead of repeating z_pred 9 times (repeat_interleave),
  merging the 9 node targets with their weights.
- Allow more numba threads for this kernel only (it is already `prange` over rows) when cores are free.
