"""ETA as a model with an SLO, plus the tracking surface that must not lie.

THE WISMO PRINCIPLE ("where is my order?"): a tracking page that is
CONFIDENTLY WRONG generates the exact support ticket it was built to prevent. A
frozen dot on a map is worse than an honest "we have lost signal", because the
customer believes the frozen dot until they stop believing anything.

So the ETA carries an uncertainty that WIDENS when its inputs go stale, and the
tracking state machine has an explicit degraded mode rather than a last-known
position rendered as if it were live.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geo import haversine

SPEED_KMH = 22.0            # urban courier average
STALE_SECONDS = 45.0        # beyond this, a location is not "live"
LOST_SECONDS = 180.0        # beyond this, say so


def predict_eta(courier_lat, courier_lon, pickup_lat, pickup_lon,
                drop_lat, drop_lon, prep_minutes, queue_depth=0.0,
                speed_kmh=SPEED_KMH):
    """Route time + prep + queue. Deliberately simple; the point is that it is
    EVALUATED, not that it is clever."""
    to_pickup = haversine(courier_lat, courier_lon, pickup_lat, pickup_lon)
    to_drop = haversine(pickup_lat, pickup_lon, drop_lat, drop_lon)
    travel = (to_pickup + to_drop) / speed_kmh * 60.0
    # the courier waits at the restaurant only for whatever prep time REMAINS
    wait = max(0.0, prep_minutes - to_pickup / speed_kmh * 60.0)
    return travel + wait + 1.6 * queue_depth


def eta_interval(point_eta: float, age_seconds: float, base_sigma: float = 4.0):
    """Uncertainty that GROWS with the age of the last location fix.

    This is the whole honesty mechanism. A point ETA computed from a 4-minute-old
    position is not the same claim as one computed from a fresh fix, and a system
    that renders them identically is lying at exactly the moment the customer is
    most likely to be checking.
    """
    inflation = 1.0 + 2.2 * min(age_seconds / LOST_SECONDS, 2.0)
    sigma = base_sigma * inflation
    return point_eta - 1.28 * sigma, point_eta + 1.28 * sigma, sigma


def tracking_state(age_seconds: float) -> str:
    if age_seconds <= STALE_SECONDS:
        return "live"
    if age_seconds <= LOST_SECONDS:
        return "delayed_signal"
    return "signal_lost"


@dataclass
class TrackingView:
    """What the customer's page actually shows."""
    state: str
    lat: float
    lon: float
    eta_low: float
    eta_high: float
    message: str
    show_dot: bool


def render_tracking(last_lat, last_lon, age_seconds, point_eta) -> TrackingView:
    st = tracking_state(age_seconds)
    lo, hi, _ = eta_interval(point_eta, age_seconds)
    if st == "live":
        return TrackingView(st, last_lat, last_lon, lo, hi,
                            "Arriving in %d-%d min" % (round(lo), round(hi)), True)
    if st == "delayed_signal":
        return TrackingView(st, last_lat, last_lon, lo, hi,
                            "Last seen %d min ago - arriving in %d-%d min"
                            % (age_seconds // 60, round(lo), round(hi)), True)
    # SIGNAL LOST: the dot comes OFF the map. Showing a stale position as if it
    # were live is what generates the "my driver hasn't moved in 20 minutes"
    # ticket -- the very ticket the tracking page exists to avoid.
    return TrackingView(st, last_lat, last_lon, lo, hi,
                        "We've lost signal from your courier. Your order is still "
                        "on its way; we'll update as soon as we hear from them.",
                        False)


def smooth_eta(previous: float | None, new: float, alpha: float = 0.45) -> float:
    """Damp ETA updates, but BAD NEWS TRAVELS FAST.

    A number that jumps every few seconds destroys trust as effectively as one
    that is wrong. But smoothing an INCREASE is a lie of omission: if the food
    will be twenty minutes late the customer needs to know now, while they can
    still do something about it. So increases pass through almost undamped and
    decreases are smoothed.
    """
    if previous is None:
        return new
    if new > previous:
        return previous + 0.85 * (new - previous)
    return previous + alpha * (new - previous)


def evaluate_eta(predicted: np.ndarray, actual: np.ndarray,
                 segment: np.ndarray, low: np.ndarray, high: np.ndarray):
    """MAE, bias and interval coverage, segmented. An ETA nobody scores is
    decoration on a map."""
    import pandas as pd
    df = pd.DataFrame(dict(pred=predicted, actual=actual, seg=segment,
                           low=low, high=high))
    df["err"] = df.pred - df.actual
    df["covered"] = ((df.actual >= df.low) & (df.actual <= df.high)).astype(float)
    out = df.groupby("seg").agg(
        n=("err", "size"),
        mae=("err", lambda e: float(np.mean(np.abs(e)))),
        bias=("err", "mean"),
        p90_abs=("err", lambda e: float(np.percentile(np.abs(e), 90))),
        coverage_80=("covered", "mean")).round(3)
    overall = dict(n=len(df), mae=float(np.mean(np.abs(df.err))),
                   bias=float(df.err.mean()),
                   p90_abs=float(np.percentile(np.abs(df.err), 90)),
                   coverage_80=float(df.covered.mean()))
    return out, overall
