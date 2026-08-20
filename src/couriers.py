"""Courier agents that accept, decline and go offline -- and the re-offer cascade.

THE FIDELITY GAP THIS CLOSES
----------------------------
The first pass called couriers "a capacity pool with a service-time
distribution" and named that as its biggest fidelity gap. It was: the offer flow
was exercised by test threads that always accepted, so offer-accept rate, time to
assign and re-offer depth -- the three numbers a dispatch team actually watches --
could not be measured at all.

Couriers here are agents with preferences. They decline offers that are too far
for too little, they get more selective as their shift goes on, and they go
offline. None of those behaviours is exotic; all three are what makes dispatch
hard, because the matcher proposes and the courier disposes.

WHY THE ACCEPTANCE MODEL IS LOGISTIC AND NOT A THRESHOLD
--------------------------------------------------------
A threshold ("decline if further than 3km") makes the whole system deterministic
and the re-offer cascade trivial -- you can compute exactly who will accept. Real
acceptance is probabilistic: the same courier takes a 4km trip when they are cold
and refuses it when they are busy. The probabilistic version is what makes the
cascade depth a random variable worth measuring.
"""
from __future__ import annotations

import numpy as np

from .geo import haversine


class CourierAgent:
    """One courier's decision rule.

    `pickiness` is the courier's own bar, drawn once per courier: some take
    everything, some cherry-pick. That heterogeneity matters because the marginal
    offer goes to the pickiest courier still available, so a fleet's effective
    supply is smaller than its headcount.
    """

    def __init__(self, courier_id: int, pickiness: float, shift_length: float,
                 rng: np.random.Generator):
        self.id = courier_id
        self.pickiness = pickiness
        self.shift_length = shift_length
        self.rng = rng
        self.offers_seen = 0
        self.offers_accepted = 0
        self.online = True
        self.minutes_worked = 0.0

    # Fixed overhead on every job regardless of distance: waiting for the order
    # at the counter, parking, handoff. It is what makes a very short trip a bad
    # deal rather than a free one.
    OVERHEAD_MINUTES = 11.0
    SPEED_KMH = 20.0

    def accept_probability(self, distance_km: float, payout: float,
                           surge: float = 1.0) -> float:
        """Logistic in the thing couriers actually trade off: money per HOUR.

        Not per km. A first version keyed on $/km and every courier accepted
        everything, because in a dense metro the marginal offer is 200 metres
        away and $6/0.2km is an absurd rate -- the metric said the job was
        wonderful when it was eleven minutes of waiting for six dollars.

        Effective time is overhead + travel, so a short trip carries the same
        fixed cost as a long one. That is why couriers dislike tiny orders and
        why per-drop pay without a distance term produces exactly the wrong
        selection.
        """
        eff_payout = payout * surge
        hours = (self.OVERHEAD_MINUTES + 60.0 * distance_km / self.SPEED_KMH) / 60.0
        value = eff_payout / max(hours, 1e-6)
        # fatigue: the bar rises through the shift
        fatigue = 1.0 + 0.8 * min(self.minutes_worked / max(self.shift_length, 1.0), 1.0)
        z = (value - self.pickiness * fatigue) * 0.22
        return float(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))))

    def decide(self, distance_km: float, payout: float, surge: float = 1.0) -> bool:
        self.offers_seen += 1
        p = self.accept_probability(distance_km, payout, surge)
        took = bool(self.rng.random() < p)
        self.offers_accepted += took
        return took

    def maybe_go_offline(self, p_offline: float = 0.02) -> bool:
        """Couriers leave mid-shift, and they leave more as the shift wears on.

        A dispatch system that assumes a stable fleet is fine until 21:00, when
        supply falls off a cliff and every model tuned at 19:00 is wrong.
        """
        if not self.online:
            return False
        rate = p_offline * (1.0 + 2.0 * min(
            self.minutes_worked / max(self.shift_length, 1.0), 1.0))
        if self.rng.random() < rate:
            self.online = False
            return True
        return False

    @property
    def accept_rate(self) -> float:
        return self.offers_accepted / self.offers_seen if self.offers_seen else float("nan")


def make_fleet(n: int, rng: np.random.Generator) -> dict[int, CourierAgent]:
    # $/hour bar. Mean ~= $24/h, which is where the acceptance curve actually
    # bites given the payout schedule below -- calibrated so the fleet is
    # selective rather than saturated at either end.
    pick = rng.gamma(9.0, 2.7, n)
    shift = rng.uniform(240, 600, n)       # 4-10 hour shifts
    return {i: CourierAgent(i, float(pick[i]), float(shift[i]),
                            np.random.default_rng(rng.integers(1 << 31)))
            for i in range(n)}


# --------------------------------------------------------------------------
# the re-offer cascade
# --------------------------------------------------------------------------
def cascade_assign(order, candidates: list[int], fleet: dict, lat, lon,
                   base_payout: float = 6.0, per_km: float = 1.1,
                   max_depth: int = 8, offer_ttl_s: float = 25.0,
                   surge: float = 1.0) -> dict:
    """Offer the order down a ranked candidate list until someone accepts.

    Each declined offer costs its TTL in wall-clock, so cascade depth converts
    directly into time-to-assign -- which is the customer-visible cost of a fleet
    that declines a lot. That conversion is the reason depth is worth measuring
    rather than just accept rate: a 60% accept rate sounds fine and means the
    median order waits through nearly one full offer cycle.

    Returns the assignment, the depth reached, and the elapsed offer time.
    """
    for depth, cid in enumerate(candidates[:max_depth], start=1):
        agent = fleet.get(cid)
        if agent is None or not agent.online:
            continue
        d = float(haversine(lat[cid], lon[cid], order.lat, order.lon))
        payout = base_payout + per_km * d
        if agent.decide(d, payout, surge):
            return dict(assigned_to=cid, depth=depth,
                        offer_seconds=(depth - 1) * offer_ttl_s,
                        distance_km=d, payout=payout)
    return dict(assigned_to=None, depth=min(len(candidates), max_depth),
                offer_seconds=min(len(candidates), max_depth) * offer_ttl_s,
                distance_km=float("nan"), payout=float("nan"))


def surge_for(utilisation: float, base: float = 1.0, cap: float = 2.5) -> float:
    """Raise payout when supply is tight.

    Included because the cascade makes its absence visible: without a surge term,
    a tight market produces long cascades and unassigned orders, and the only
    lever left is routing -- which cannot conjure couriers. Surge is the lever
    that actually moves acceptance, and it is a COST, so it belongs in the same
    table as the assignment quality it buys.
    """
    if utilisation <= 0.6:
        return base
    # NOTE: at this slope the cap does NOT bind -- utilisation 1.0 gives 1.88,
    # below the 2.5 ceiling. It is a guard against a future steeper slope, not a
    # constraint that shapes any number in this report, and saying so is better
    # than leaving a reader to assume the surge is being clipped somewhere.
    return float(min(cap, base * (1.0 + 2.2 * (utilisation - 0.6))))
