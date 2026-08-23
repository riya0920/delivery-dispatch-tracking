"""Courier repositioning, and surge that does not oscillate.

TWO GAPS
--------
  "No courier repositioning. Couriers do not move toward demand between jobs,
   which is the behaviour that makes courier-incentive experiments hard (and the
   carryover mechanism DATA-3 reasons about qualitatively)."
  "Surge is a function of instantaneous utilisation only -- no forecast, no
   hysteresis, so it would oscillate in a real control loop."

WHY THEY BELONG IN ONE FILE
---------------------------
They are the same control problem seen from two sides. Surge is a price signal
that moves couriers; repositioning is couriers moving. A surge model with no
repositioning is a price with no supply response, which is why the previous
pass's surge could only buy acceptance and never buy PRESENCE -- and presence is
what actually fixes a short region.

Running them together also produces the failure that makes this hard: surge
attracts couriers to a region, which lowers its utilisation, which lowers surge,
which sends them away. That is a control loop with a delay in it, and a
controller with no hysteresis and no forecast will oscillate.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# surge
# --------------------------------------------------------------------------
class SurgeController:
    """Utilisation -> multiplier, with hysteresis, a forecast and a rate limit.

    THREE MECHANISMS, EACH FIXING A DIFFERENT FAILURE:

      hysteresis   -- separate switch-ON and switch-OFF thresholds. A single
                      threshold makes the controller chatter across it, and
                      chatter in a price is worse than a wrong price: couriers
                      cannot plan around a multiplier that changes every minute
                      and stop believing it.
      forecast     -- act on where utilisation is GOING, not where it is. Supply
                      responds with a lag, so a controller that waits for the
                      region to be short is already late by the length of that
                      lag.
      rate limit   -- cap the change per step. This is what stops the forecast
                      from turning a noisy derivative into a price spike, and it
                      is the cheapest safety property in the whole loop.
    """

    def __init__(self, on_threshold: float = 0.72, off_threshold: float = 0.62,
                 slope: float = 2.2, cap: float = 2.5,
                 max_step: float = 0.08, forecast_minutes: float = 6.0,
                 ema_alpha: float = 0.25):
        if off_threshold >= on_threshold:
            # Without a gap there is no hysteresis, only a threshold with extra
            # arguments -- and a controller that claims hysteresis and does not
            # have it is worse than one that never claimed it.
            raise ValueError("off_threshold must be BELOW on_threshold")
        self.on, self.off = on_threshold, off_threshold
        self.slope, self.cap = slope, cap
        self.max_step = max_step
        self.forecast_minutes = forecast_minutes
        self.alpha = ema_alpha
        self.multiplier = 1.0
        self.active = False
        self._ema = None
        self._trend = 0.0

    def observe(self, utilisation: float) -> float:
        """Feed one utilisation reading; returns the new multiplier."""
        u = float(np.clip(utilisation, 0.0, 1.0))
        if self._ema is None:
            self._ema = u
        else:
            prev = self._ema
            self._ema = self.alpha * u + (1 - self.alpha) * prev
            # Trend on the SMOOTHED series, not the raw one. Differencing raw
            # utilisation amplifies exactly the minute-to-minute noise the EMA
            # exists to remove, and a forecast built on it is a noise amplifier
            # with a price attached.
            self._trend = self._ema - prev

        projected = float(np.clip(self._ema + self._trend * self.forecast_minutes,
                                  0.0, 1.0))

        if self.active:
            if projected < self.off:
                self.active = False
        else:
            if projected > self.on:
                self.active = True

        target = 1.0
        if self.active:
            over = max(0.0, projected - self.off)
            target = min(self.cap, 1.0 + self.slope * over)

        step = float(np.clip(target - self.multiplier, -self.max_step, self.max_step))
        self.multiplier = float(np.clip(self.multiplier + step, 1.0, self.cap))
        return self.multiplier

    def state(self) -> dict:
        return dict(multiplier=self.multiplier, active=self.active,
                    smoothed_utilisation=self._ema, trend=self._trend)


def naive_surge(utilisation: float, threshold: float = 0.6, slope: float = 2.2,
                cap: float = 2.5) -> float:
    """The previous pass's controller: instantaneous, no memory, no limit.

    Kept so the oscillation comparison has a baseline rather than an assertion.
    """
    if utilisation <= threshold:
        return 1.0
    return float(min(cap, 1.0 + slope * (utilisation - threshold)))


def oscillation(series: list[float]) -> dict:
    """How much a multiplier series churns.

    `reversals` counts direction changes, which is the number a courier
    experiences: a price that goes up, down, up, down within an hour is not a
    signal, it is noise with a dollar sign.
    """
    a = np.asarray(series, float)
    if len(a) < 3:
        return dict(reversals=0, mean_abs_step=0.0, range=0.0)
    d = np.diff(a)
    sign = np.sign(d)
    nz = sign[sign != 0]
    reversals = int(np.sum(nz[1:] != nz[:-1])) if len(nz) > 1 else 0
    return dict(reversals=reversals, mean_abs_step=float(np.mean(np.abs(d))),
                range=float(a.max() - a.min()))


# --------------------------------------------------------------------------
# repositioning
# --------------------------------------------------------------------------
def reposition(courier_zone: np.ndarray, idle: np.ndarray,
               demand_forecast: np.ndarray, supply: np.ndarray,
               travel_minutes: np.ndarray, surge: np.ndarray,
               rng: np.random.Generator,
               move_threshold: float = 1.25,
               compliance: float = 0.55) -> np.ndarray:
    """Idle couriers choose a zone to drift toward. Returns the new zone per courier.

    The decision is a RATIO of expected earnings, not a difference: a courier
    compares what they would make where they are against what they would make
    somewhere else, net of the time it takes to get there. Using a difference
    would make every high-value zone attract every courier regardless of
    distance, which is the behaviour that produces the classic thundering-herd
    artifact in these simulations.

    COMPLIANCE IS NOT 1.0, and that is the most important parameter here. A
    dispatch team can recommend a zone; they cannot move anybody. Modelling
    perfect compliance turns a marketplace into a fleet, and every conclusion
    about incentives drawn from a perfect-compliance simulation is a conclusion
    about a business somebody else is in.
    """
    n_zones = len(demand_forecast)
    out = courier_zone.copy()
    value = surge * demand_forecast / np.maximum(supply, 1.0)

    for i in np.flatnonzero(idle):
        here = courier_zone[i]
        # value net of travel: 12 minutes of driving is 12 minutes not earning
        net = value / (1.0 + travel_minutes[here] / 12.0)
        best = int(np.argmax(net))
        if best == here:
            continue
        if net[best] > move_threshold * net[here] and rng.random() < compliance:
            out[i] = best
    return out


def herding_index(zone_counts: np.ndarray) -> float:
    """Concentration of couriers across zones, 0 = even, 1 = all in one place.

    The number that says whether a repositioning policy has produced a
    thundering herd. A policy that raises earnings while driving this toward 1
    has moved the shortage rather than fixed it.
    """
    x = np.asarray(zone_counts, float)
    tot = x.sum()
    if tot <= 0:
        return 0.0
    p = x / tot
    n = len(p)
    if n <= 1:
        return 0.0
    hhi = float((p ** 2).sum())
    return (hhi - 1.0 / n) / (1.0 - 1.0 / n)
