#!/usr/bin/env python3
"""Forecast GTA VI PS5 trophy earn rates on Exophase at a target instant.

Reads /input/<id>.in (schema makefun/gta6-trophies-case/v1) and writes
/output/<id>.out = {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}.

Model (per trophy, all in logit space, logit(p) = ln(p / (1 - p)) with
p = rate / 100 clamped to [0.01, 99.99]):

  * Dedupe consecutive reads that change nothing for the trophy.
  * With >= 3 distinct reads: recency-weighted least squares of
    logit(rate) on ln(1 + hours since 2026-11-19T00:00:00Z) (12 h
    recency half-life), slopes shrunk toward the tier mean, then
    damped 40% toward the last observed rate. A second forecast from
    the earners/players decomposition (log earners and log players
    each fit on the same clock, rate = earners / players) is averaged
    in, which anticipates dilution when the tracked-player base grows
    faster than earners.
  * With fewer distinct reads (flat history, e.g. the practice cases,
    where Exophase had not refreshed yet): persistence minus a
    1.2-logit dilution prior, spread 1.0 -- the first refresh after
    launch night grew players tracked 638 -> 3,868 while earners
    lagged, so every rate but the one at 100% fell.
  * Spread: residual variance + slope uncertainty at the target +
    refresh risk, clamped to [0.35, 2.5] logit.

Standard library only, deterministic, no network, no clock reads.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Launch reference for the log-time clock, and the standard levels.
T0 = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Flat-history fallback: dilution prior and spread (logit units).
DILUTION_PRIOR = 1.2
FLAT_SPREAD = 1.0
# Fitted branch: damping toward persistence, spread clamp, recency.
DAMP = 0.6
SPREAD_MIN = 0.35
SPREAD_MAX = 2.5
RECENCY_HALFLIFE_H = 12.0
REFRESH_VAR = 0.09  # (0.3 logit)^2 of refresh-timing risk
SLOPE_PRIOR_VAR = 0.09  # (0.3 logit per log-hour)^2 tier-shrink prior


def clamp_rate(rate):
    return min(99.99, max(0.01, rate))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(y):
    if y >= 0:
        return 100.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, as JavaScript Math.round does."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_launch(moment):
    hours = (moment - T0).total_seconds() / 3600.0
    return max(hours, -0.99)


def x_clock(moment):
    return math.log1p(hours_since_launch(moment))


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's approximation)."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285984469534e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def wls_line(xs, ys, ws):
    """Weighted least squares y = a + b*x. Returns (a, b, var_b, resid_var)."""
    sw = sum(ws)
    xm = sum(w * x for w, x in zip(ws, xs)) / sw
    ym = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - xm) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - xm) * (y - ym) for w, x, y in zip(ws, xs, ys))
    if sxx <= 1e-12:
        return ym, 0.0, float("inf"), 0.0
    b = sxy / sxx
    a = ym - b * xm
    n = len(xs)
    if n > 2:
        resid = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
        resid_var = resid / (sw * (n - 2) / n)
    else:
        resid_var = 0.0
    var_b = resid_var / sxx if sxx > 0 else float("inf")
    return a, b, var_b, resid_var


def forecast_trophy(idx, trophy, history, x_target, horizon_days):
    """Return (median_logit, spread_logit) for one trophy."""
    last_rate = None
    for read in reversed(history):
        rate = read["rates"][idx]
        if rate is not None:
            last_rate = rate
            break
    if last_rate is None:
        return logit(1.0), SPREAD_MAX

    # Dedupe: keep a read when this trophy's rate or the player count moves.
    pts = []  # (x, logit_rate, players, earners)
    prev_rate = prev_players = None
    for read in history:
        rate = read["rates"][idx]
        players = read["players_tracked"]
        if rate is None:
            continue
        if rate == prev_rate and players == prev_players:
            continue
        prev_rate, prev_players = rate, players
        pts.append((x_clock(parse_time(read["at"])), logit(rate),
                    players, rate / 100.0 * players))
    last_logit = logit(last_rate)

    # Flat history: no trend to fit, use the dilution prior.
    if len(pts) < 3 or len({p[1] for p in pts}) < 2:
        if last_rate >= 99.99:
            return logit(99.99), SPREAD_MIN
        if last_rate <= 0.01:
            return logit(0.01), FLAT_SPREAD
        spread = min(SPREAD_MAX, max(SPREAD_MIN,
                                     FLAT_SPREAD * math.sqrt(1.0 + horizon_days)))
        return last_logit - DILUTION_PRIOR, spread

    t_last = max(p[0] for p in pts)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    # Recency weight (12 h half-life in real hours) over rounding-noise
    # variance of logit(rate): the rate is printed to 2 decimals.
    ws = []
    hours_last = math.expm1(t_last)
    for (x, y, _players, _e) in pts:
        hours = math.expm1(x)
        recency = 0.5 ** (max(0.0, hours_last - hours) / RECENCY_HALFLIFE_H)
        p = 1.0 / (1.0 + math.exp(-y))
        sigma_round = 5e-5 / max(p * (1.0 - p), 1e-6)
        var = sigma_round * sigma_round + 0.02 * 0.02
        ws.append(recency / var)

    a, b, var_b, resid_var = wls_line(xs, ys, ws)
    return (a, b, var_b, resid_var, xs, ys, ws, pts, last_logit)


def forecast_case(case):
    levels = case.get("levels") or LEVELS
    zs = [norm_ppf(level) for level in levels]
    history = case.get("history") or []
    target = parse_time(case["target"])
    x_target = x_clock(target)
    horizon_days = case.get("horizon_minutes", 0) / 1440.0

    # First pass: per-trophy fits, to pool slopes within tiers.
    fits = {}
    for i, trophy in enumerate(case["trophies"]):
        fits[i] = forecast_trophy(i, trophy, history, x_target, horizon_days)

    tier_slopes = {}
    all_slopes = []
    for i, trophy in enumerate(case["trophies"]):
        f = fits[i]
        if len(f) == 2:
            continue
        _a, b, var_b, _rv, _xs, _ys, _ws, _pts, _ll = f
        tier_slopes.setdefault(trophy["tier"], []).append((b, var_b))
        all_slopes.append((b, var_b))

    def pooled_slope(tier):
        group = tier_slopes.get(tier) or all_slopes
        if not group:
            return 0.0
        num = sum(b / (vb + SLOPE_PRIOR_VAR) for b, vb in group)
        den = sum(1.0 / (vb + SLOPE_PRIOR_VAR) for b, vb in group)
        return num / den if den else 0.0

    global_slope = pooled_slope(None)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        f = fits[i]
        if len(f) == 2:
            med_logit, spread = f
        else:
            a, b, var_b, resid_var, xs, ys, ws, pts, last_logit = f
            pool = pooled_slope(trophy["tier"])
            if len(tier_slopes.get(trophy["tier"], [])) < 3:
                pool = global_slope
            # Shrink the slope toward the tier/global mean.
            w_self = 1.0 / (var_b + 1e-9) if var_b != float("inf") else 0.0
            w_pool = 1.0 / SLOPE_PRIOR_VAR
            b_star = (b * w_self + pool * w_pool) / (w_self + w_pool) \
                if (w_self + w_pool) > 0 else pool
            # Refit the intercept at the shrunk slope.
            sw = sum(ws)
            xm = sum(w * x for w, x in zip(ws, xs)) / sw
            ym = sum(w * y for w, y in zip(ws, ys)) / sw
            a_star = ym - b_star * xm
            y_hat = a_star + b_star * x_target
            med_rate = last_logit + DAMP * (y_hat - last_logit)

            # Earners/players decomposition as a second forecast.
            med_dec = None
            earners = [p[3] for p in pts]
            players = [p[2] for p in pts]
            if min(earners) > 0 and len(set(players)) >= 2:
                log_e = [math.log(e) for e in earners]
                log_p = [math.log(p) for p in players]
                a_e, b_e, _vb, _rv = wls_line(xs, log_e, ws)
                a_p, b_p, _vb2, _rv2 = wls_line(xs, log_p, ws)
                rate2 = 100.0 * math.exp((a_e + b_e * x_target)
                                         - (a_p + b_p * x_target))
                y2 = logit(rate2)
                med_dec = last_logit + DAMP * (y2 - last_logit)
            if med_dec is not None:
                med_logit = 0.5 * (med_rate + med_dec)
            else:
                med_logit = med_rate

            # Spread: residual + slope uncertainty + refresh risk.
            var_med = max(resid_var, 0.01) + REFRESH_VAR
            if var_b != float("inf"):
                var_med += var_b * (x_target - xm) ** 2
            spread = min(SPREAD_MAX, max(SPREAD_MIN, math.sqrt(var_med)))
            # A trophy pinned at 100% stays pinned; one at 0% stays ~0%.
            if pts[-1][1] >= logit(99.99):
                med_logit, spread = logit(99.99), SPREAD_MIN
            med_logit = min(max(med_logit, logit(0.01)), logit(99.99))

        qs = []
        for z in zs:
            q = inv_logit(med_logit + z * spread)
            qs.append(round4(min(99.99, max(0.01, q))))
        # Keep quantiles non-decreasing after rounding.
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
