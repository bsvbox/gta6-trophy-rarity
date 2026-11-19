# bsvbox/gta6-trophy-rarity

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bsvbox/gta6-trophy-rarity/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bsvbox/gta6-trophy-rarity/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, its earn rate on Exophase at 2026-11-26T00:00:00Z (a week after launch), as five quantiles per trophy (5%, 25%, 50%, 75%, 95%), from the house's recorded reads of the trophy page up to the deadline.

The model works in logit space, per trophy: a recency-weighted least-squares line of logit(rate) against ln(1 + hours since 2026-11-19T00:00:00Z), with slopes shrunk toward the tier mean, damped 40% toward the last observed rate, and averaged with a second forecast from the earners/players decomposition (log earners and log players fit on the same clock, rate = earners/players), which anticipates dilution when the tracked-player base grows faster than earners. The spread comes from residual variance, slope uncertainty at the target, and refresh risk, clamped to 0.35–2.5 logit. When the history is flat (no refresh seen yet, as on launch night), the median is persistence minus a 1.2-logit dilution prior with spread 1.0, because Exophase's first refresh grew players tracked 638 to 3,868 while earners lagged. Deterministic, offline, Python standard library only.

Practice score (server, dev phase): measured on submit; the house persistence baseline scores about 1.51.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=4eqvbahr&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
