# Dynamic Discount Recommendation - Phase 7 Policy Comparison

## Table 1: Online Mode (Bandits Learning Live Across 20,000 Rounds)
*Note: Evaluated across 10 independent random seeds (20,000 rounds each). All rupee figures represent simulated rupees (Rs).*

| Policy | Total Revenue (Rs) | Total Profit (Rs) | Conversion Rate (%) | Avg Discount Offered (%) | Total Discount Given (Rs) | Cumulative Regret (Rs) |
| --- | --- | --- | --- | --- | --- | --- |
| Random | Rs 232,030.37 +/- 3,354.84 | Rs 78,371.40 +/- 985.88 | 24.05% +/- 0.33% | 8.20% +/- 0.13% | Rs 24,067.26 +/- 747.06 | Rs 1,724.71 +/- 24.09 |
| Fixed 10% | Rs 232,094.81 +/- 5,212.23 | Rs 80,265.41 +/- 1,788.01 | 23.81% +/- 0.44% | 8.09% +/- 0.04% | Rs 20,954.58 +/- 511.11 | Rs 23.97 +/- 1.59 |
| Rule-based | Rs 232,086.22 +/- 3,581.12 | Rs 78,432.33 +/- 1,176.84 | 24.07% +/- 0.37% | 8.19% +/- 0.10% | Rs 24,002.89 +/- 578.23 | Rs 1,529.80 +/- 31.51 |
| LinUCB | Rs 234,276.84 +/- 4,476.98 | Rs 78,924.82 +/- 788.32 | 24.29% +/- 0.58% | 8.45% +/- 0.59% | Rs 24,642.35 +/- 2,794.46 | Rs 1,475.69 +/- 399.77 |
| Thompson Sampling | Rs 232,651.13 +/- 5,585.07 | Rs 78,856.21 +/- 1,027.65 | 24.08% +/- 0.70% | 8.21% +/- 0.71% | Rs 23,673.25 +/- 3,041.12 | Rs 1,489.61 +/- 537.17 |

## Table 2: Frozen Mode (Deployed Policy Quality on 5,000 Held-Out Rounds)
*Note: Evaluated across 10 independent held-out sequences (5,000 rounds each) with learning disabled.*

| Policy | Total Revenue (Rs) | Total Profit (Rs) | Profit / Round (Rs) | Conversion Rate (%) | Avg Discount Offered (%) | Discount Given / Round (Rs) | Total Discount Given (Rs) | Cumulative Regret (Rs) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random | Rs 57,513.17 +/- 2,059.95 | Rs 19,375.57 +/- 650.56 | Rs 3.8751 +/- 0.1301 | 24.11% +/- 0.72% | 8.33% +/- 0.27% | Rs 1.2099 +/- 0.0712 | Rs 6,049.37 +/- 355.92 | Rs 438.64 +/- 12.88 |
| Fixed 10% | Rs 58,298.52 +/- 1,932.34 | Rs 20,136.83 +/- 644.67 | Rs 4.0274 +/- 0.1289 | 23.93% +/- 0.68% | 8.11% +/- 0.11% | Rs 1.0609 +/- 0.0458 | Rs 5,304.56 +/- 228.88 | Rs 5.65 +/- 0.37 |
| Rule-based | Rs 58,313.35 +/- 2,499.43 | Rs 19,667.14 +/- 883.74 | Rs 3.9334 +/- 0.1767 | 24.41% +/- 0.79% | 8.29% +/- 0.20% | Rs 1.2194 +/- 0.0558 | Rs 6,096.96 +/- 279.15 | Rs 387.32 +/- 15.86 |
| LinUCB | Rs 58,464.50 +/- 2,463.67 | Rs 19,736.87 +/- 669.86 | Rs 3.9474 +/- 0.1340 | 24.42% +/- 1.10% | 8.47% +/- 0.68% | Rs 1.2163 +/- 0.1725 | Rs 6,081.46 +/- 862.28 | Rs 346.06 +/- 113.89 |
| Thompson Sampling | Rs 58,158.75 +/- 2,116.35 | Rs 19,729.93 +/- 620.49 | Rs 3.9460 +/- 0.1241 | 24.31% +/- 0.87% | 8.25% +/- 0.70% | Rs 1.1778 +/- 0.1531 | Rs 5,889.20 +/- 765.59 | Rs 337.54 +/- 127.88 |

## Table 3: Profit Lift & Win Counts
*Note: Paired per-seed comparison against Fixed 10% and Rule-based baselines.*

| Mode | Bandit Policy | Comparator | Profit Difference (Rs) | Percentage Lift (%) | Win Count |
| --- | --- | --- | --- | --- | --- |
| ONLINE (20k rounds) | LinUCB | Fixed 10% | Rs -1,340.59 +/- 1,554.26 | -1.67% | 1 of 10 seeds |
| ONLINE (20k rounds) | LinUCB | Rule-based | Rs +492.50 +/- 1,022.62 | +0.63% | 8 of 10 seeds |
| ONLINE (20k rounds) | Thompson Sampling | Fixed 10% | Rs -1,409.20 +/- 1,398.69 | -1.76% | 2 of 10 seeds |
| ONLINE (20k rounds) | Thompson Sampling | Rule-based | Rs +423.88 +/- 937.37 | +0.54% | 6 of 10 seeds |
| FROZEN (5k rounds) | LinUCB | Fixed 10% | Rs -399.96 +/- 476.46 | -1.99% | 1 of 10 seeds |
| FROZEN (5k rounds) | LinUCB | Rule-based | Rs +69.73 +/- 811.53 | +0.35% | 4 of 10 seeds |
| FROZEN (5k rounds) | Thompson Sampling | Fixed 10% | Rs -406.90 +/- 403.13 | -2.02% | 2 of 10 seeds |
| FROZEN (5k rounds) | Thompson Sampling | Rule-based | Rs +62.79 +/- 611.42 | +0.32% | 7 of 10 seeds |
