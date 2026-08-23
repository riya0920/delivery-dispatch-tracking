"""A learned ETA, and the analytic one it is scored against.

THE GAP
-------
"The ETA model is route-time + prep + queue. No traffic, no historical
per-restaurant prep distributions, no learned model."

WHY THE ANALYTIC MODEL IS THE BASELINE AND NOT THE STRAW MAN
-------------------------------------------------------------
route + prep + queue is a good model. It is interpretable, it needs no training
data, it degrades gracefully, and on a well-specified problem it is very hard to
beat. Presenting it as the thing a gradient booster trivially defeats would be
dishonest, so the comparison is set up to be fair: both models see the same
features, and the analytic one is given the correct mean prep time per restaurant
rather than a global constant.

WHAT THE LEARNED MODEL CAN DO THAT THE ANALYTIC ONE CANNOT
-----------------------------------------------------------
Three things, and only three:

  * **interactions** -- a large order at a slow restaurant at 19:00 is worse than
    the sum of those three effects, and an additive model cannot say so;
  * **per-restaurant idiosyncrasy** learned from data rather than configured,
    which matters because prep time is the largest single term and the one a
    dispatch team has least control over;
  * **the conditional QUANTILE**, so the promise made to the customer can be a
    P80 rather than a mean. That is the one with real product consequence: an
    ETA is a promise, and a mean is wrong half the time by construction.

WHAT IT CANNOT DO
-----------------
Explain itself, extrapolate to a restaurant it has never seen, or survive a
distribution shift like a new city. The analytic model does all three, which is
why both are kept and why the fallback path exists.
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

FEATURES = ["route_seconds", "straight_metres", "detour_factor", "hour",
            "is_peak", "items", "restaurant_id", "restaurant_mean_prep",
            "queue_depth", "courier_utilisation"]


def analytic_eta(route_seconds, prep_minutes, queue_depth, items=None,
                 queue_minutes_each: float = 1.6,
                 minutes_per_item: float = 0.55,
                 prep_noise_minutes: float = 3.0) -> np.ndarray:
    """route + prep + per-item + queue. The incumbent, given every ADDITIVE term.

    The first version omitted the per-item term and the mean of the prep noise,
    which made it 6.8 minutes biased and turned the comparison into a straw man:
    the learned model would have won on a term anybody could have configured.

    Every additive term the generator uses is now supplied, because the claim
    being tested is specifically that an additive model cannot express an
    INTERACTION -- and that claim is only interesting if the additive model is
    otherwise correct.
    """
    out = (np.asarray(route_seconds, float) / 60.0
           + np.asarray(prep_minutes, float)
           + np.asarray(queue_depth, float) * queue_minutes_each
           + prep_noise_minutes)
    if items is not None:
        out = out + np.asarray(items, float) * minutes_per_item
    return out


def fit_learned(X, y, quantile: float | None = None, n_estimators: int = 250):
    """LightGBM on the same features. `quantile` fits a conditional quantile.

    `restaurant_id` is passed as a CATEGORICAL rather than an integer. As an
    integer the tree can only split it into ranges, which is meaningless for an
    identifier -- restaurant 17 is not between 16 and 18 in any sense that
    predicts prep time.
    """
    import lightgbm as lgb
    params = dict(n_estimators=n_estimators, learning_rate=0.06, num_leaves=31,
                  min_child_samples=25, reg_lambda=1.0, random_state=5,
                  n_jobs=-1, verbosity=-1)
    if quantile is None:
        m = lgb.LGBMRegressor(objective="regression_l1", **params)
    else:
        m = lgb.LGBMRegressor(objective="quantile", alpha=quantile, **params)
    cat = [FEATURES.index("restaurant_id")]
    m.fit(np.asarray(X, np.float32), np.asarray(y, float),
          categorical_feature=cat)
    return m


def evaluate(pred, actual) -> dict:
    """MAE, bias, and the two numbers a customer actually experiences.

    `late_share` and `mean_lateness_when_late` matter more than MAE, because the
    cost of an ETA error is asymmetric: five minutes early is a pleasant surprise
    and five minutes late is a support contact. A model chosen on MAE alone will
    happily trade one for the other at par.
    """
    p = np.asarray(pred, float)
    a = np.asarray(actual, float)
    late = a > p
    return dict(mae=float(np.mean(np.abs(p - a))),
                bias=float(np.mean(p - a)),
                late_share=float(np.mean(late)),
                mean_lateness_when_late=float(np.mean((a - p)[late]))
                if late.any() else 0.0,
                p90_abs_error=float(np.percentile(np.abs(p - a), 90)))


def generate(net, n: int = 6000, n_restaurants: int = 40,
             seed: int = 0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Trips with a KNOWN generating process, so both models can be scored.

    The truth deliberately contains an INTERACTION the analytic model cannot
    express: a large order at a slow restaurant during peak is worse than the sum
    of its parts, because a busy kitchen degrades non-linearly. That is the only
    structural advantage the learned model is given, and naming it means the
    comparison measures something specific rather than "ML is better".
    """
    rng = np.random.default_rng(seed)
    rest_prep = rng.gamma(6.0, 1.6, n_restaurants) + 4.0
    rest_noise = rng.gamma(3.0, 0.5, n_restaurants)

    rows, actual = [], []
    for _ in range(n):
        hour = float(np.clip(rng.normal(rng.choice([12.5, 18.8]), 1.6), 6, 23))
        r0, c0 = rng.integers(0, net.R), rng.integers(0, net.C)
        r1, c1 = rng.integers(0, net.R), rng.integers(0, net.C)
        a = net.node_latlon(int(r0), int(c0))
        b = net.node_latlon(int(r1), int(c1))
        rt = net.route(a[0], a[1], b[0], b[1], hour)
        rid = int(rng.integers(0, n_restaurants))
        items = int(rng.integers(1, 9))
        queue = int(rng.poisson(2.2))
        util = float(np.clip(rng.beta(4, 3), 0, 0.99))
        is_peak = 1.0 if (11.5 < hour < 13.5 or 17.5 < hour < 20.0) else 0.0

        prep = rest_prep[rid] + 0.55 * items + rng.gamma(2.0, rest_noise[rid])
        # THE INTERACTION: slow kitchen x big order x peak, multiplicative.
        if is_peak and items >= 5 and rest_prep[rid] > np.median(rest_prep):
            prep *= 1.35
        drive = rt["seconds"] / 60.0 * (1.0 + 0.08 * rng.standard_normal())
        total = drive + prep + queue * 1.6 + rng.normal(0, 1.2)

        rows.append([rt["seconds"], rt["straight_metres"], rt["detour_factor"],
                     hour, is_peak, items, rid, rest_prep[rid], queue, util])
        actual.append(max(total, 3.0))

    X = np.asarray(rows, np.float32)
    return X, np.asarray(actual, float), dict(rest_prep=rest_prep)
