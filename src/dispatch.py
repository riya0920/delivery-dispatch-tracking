"""Offer-based dispatch: exactly-one-assignment, greedy vs batched, ETA.

The offer flow is how real platforms work and it is where the concurrency bug
lives: the matcher PROPOSES, the courier ACCEPTS or DECLINES within a TTL, and
two couriers can accept the same offer milliseconds apart. Exactly one must win,
cleanly, and the loser must be told why rather than silently dropped.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geo import haversine


@dataclass
class Offer:
    offer_id: str
    order_id: int
    courier_id: int
    expires_at: float
    state: str = "open"          # open | accepted | declined | expired | superseded


@dataclass
class Order:
    order_id: int
    lat: float
    lon: float
    created_at: float
    prep_minutes: float
    assigned_to: int | None = None
    assigned_at: float | None = None
    offers: list = field(default_factory=list)


class AssignmentRace(Exception):
    pass


class Dispatcher:
    """Holds the exactly-one-assignment invariant.

    The invariant is enforced by a single guarded compare-and-set on the order's
    `assigned_to` field, under one lock. It is deliberately NOT enforced by
    checking-then-writing, because that is the race it exists to prevent -- the
    same defect as decrement-and-hope inventory in SE-1, in a different costume.
    """

    def __init__(self):
        self.orders: dict[int, Order] = {}
        self.offers: dict[str, Offer] = {}
        self._lock = threading.Lock()
        # A courier can carry one order at a time in this model. Without this
        # set, nothing stopped one courier winning two different orders
        # concurrently -- the invariant checker caught it, which is what
        # invariant checkers are for.
        self.busy_couriers: set[int] = set()
        self.accept_races = 0          # lost because the ORDER was taken
        self.courier_busy_rejects = 0  # lost because the COURIER was taken

    def create_order(self, order: Order):
        with self._lock:
            self.orders[order.order_id] = order

    def make_offer(self, order_id: int, courier_id: int, ttl: float,
                   now: float | None = None) -> Offer:
        now = time.time() if now is None else now
        oid = "%d:%d:%.6f" % (order_id, courier_id, now)
        off = Offer(oid, order_id, courier_id, now + ttl)
        with self._lock:
            self.offers[oid] = off
            self.orders[order_id].offers.append(oid)
        return off

    def accept(self, offer_id: str, now: float | None = None) -> bool:
        """Returns True if this courier won the order. Exactly one caller can."""
        now = time.time() if now is None else now
        with self._lock:
            off = self.offers.get(offer_id)
            if off is None:
                return False
            if off.state != "open":
                # already superseded by the winner's sweep -- still a contested
                # offer, and counting it is the difference between "0 races
                # happened" and "0 races were RECORDED"
                if off.state == "superseded":
                    self.accept_races += 1
                return False
            if now > off.expires_at:
                off.state = "expired"
                return False
            order = self.orders[off.order_id]
            if order.assigned_to is not None:
                # someone else got here first: clean loss, recorded, not silent
                off.state = "superseded"
                self.accept_races += 1
                return False
            if off.courier_id in self.busy_couriers:
                # this courier already won a different order in the same instant
                off.state = "superseded"
                self.courier_busy_rejects += 1
                return False
            order.assigned_to = off.courier_id
            self.busy_couriers.add(off.courier_id)
            order.assigned_at = now
            off.state = "accepted"
            # every other open offer on this order is now void
            for other_id in order.offers:
                other = self.offers[other_id]
                if other.offer_id != offer_id and other.state == "open":
                    other.state = "superseded"
            return True

    def decline(self, offer_id: str):
        with self._lock:
            off = self.offers.get(offer_id)
            if off and off.state == "open":
                off.state = "declined"

    def check_invariants(self) -> list[str]:
        problems = []
        for o in self.orders.values():
            accepted = [self.offers[i] for i in o.offers
                        if self.offers[i].state == "accepted"]
            if len(accepted) > 1:
                problems.append("order %d has %d accepted offers"
                                % (o.order_id, len(accepted)))
            if accepted and o.assigned_to != accepted[0].courier_id:
                problems.append("order %d assignment disagrees with its offer"
                                % o.order_id)
            if o.assigned_to is not None and not accepted:
                problems.append("order %d assigned with no accepted offer" % o.order_id)
        # no courier may hold two orders at once
        seen = {}
        for o in self.orders.values():
            if o.assigned_to is None:
                continue
            if o.assigned_to in seen:
                problems.append("courier %d assigned to orders %d and %d"
                                % (o.assigned_to, seen[o.assigned_to], o.order_id))
            seen[o.assigned_to] = o.order_id
        return problems


# --------------------------------------------------------------------------
# matching policies
# --------------------------------------------------------------------------
def greedy_match(orders, courier_lat, courier_lon, available):
    """Nearest available courier, order by order, first come first served.

    The baseline. Its failure mode is myopia: an order that arrives first takes
    the courier that a later, much closer order needed.
    """
    assign = {}
    taken = set()
    for o in orders:
        d = haversine(courier_lat, courier_lon, o.lat, o.lon)
        d = np.where(available, d, np.inf)
        for c in np.argsort(d):
            if int(c) not in taken and np.isfinite(d[c]):
                assign[o.order_id] = int(c)
                taken.add(int(c))
                break
    return assign


def batched_match(orders, courier_lat, courier_lon, available, max_km=8.0):
    """Accumulate a window of orders and solve the assignment problem jointly.

    Hungarian / min-cost, via scipy's linear_sum_assignment. The win over greedy
    is global: it will give courier A to the second order if that frees a much
    better courier for the first. The cost is that everyone waits for the window
    to close.
    """
    if not orders:
        return {}
    cand = np.flatnonzero(available)
    if len(cand) == 0:
        return {}
    C = np.zeros((len(orders), len(cand)))
    for i, o in enumerate(orders):
        C[i] = haversine(courier_lat[cand], courier_lon[cand], o.lat, o.lon)
    C = np.where(C > max_km, 1e6, C)
    rows, cols = linear_sum_assignment(C)
    return {orders[r].order_id: int(cand[c])
            for r, c in zip(rows, cols) if C[r, c] < 1e6}
