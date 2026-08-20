"""Guards on the geo indexes, the assignment invariant, and the tracking surface."""
from __future__ import annotations

import os
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import dispatch as D  # noqa: E402
from src import eta as E  # noqa: E402
from src import geo as G  # noqa: E402


@pytest.fixture(scope="module")
def fleet():
    rng = np.random.default_rng(0)
    n = 3000
    lat = 37.76 + rng.normal(0, 0.05, n)
    lon = -122.44 + rng.normal(0, 0.05, n)
    available = rng.random(n) < 0.4
    return lat, lon, available


# --------------------------------------------------------------------------
# geo
# --------------------------------------------------------------------------
def test_haversine_against_known_distance():
    # one degree of latitude is ~111.19 km
    d = G.haversine(np.array([0.0]), np.array([0.0]),
                    np.array([1.0]), np.array([0.0]))[0]
    assert 110.0 < d < 112.0


def test_every_index_agrees_with_the_exact_scan(fleet):
    """The indexes exist to be faster, not different. Any structure that returns
    a different answer from the exact scan on a static fleet is broken, and the
    only way to know is to check it against the scan."""
    lat, lon, av = fleet
    truth = G.LinearScan(lat.copy(), lon.copy(), av)
    rng = np.random.default_rng(1)
    for cls in (G.KDTreeIndex, G.GridHash, G.HexIndex):
        idx = cls(lat.copy(), lon.copy(), av)
        for _ in range(40):
            qlat = 37.76 + rng.normal(0, 0.05)
            qlon = -122.44 + rng.normal(0, 0.05)
            want = truth.query(qlat, qlon, k=5, radius_km=3.0)
            got = idx.query(qlat, qlon, k=5, radius_km=3.0)
            assert set(got) == set(want), "%s disagrees with exact scan" % cls.name


def test_indexes_never_return_unavailable_couriers(fleet):
    lat, lon, av = fleet
    for cls in G.INDEXES:
        idx = cls(lat.copy(), lon.copy(), av)
        got = idx.query(37.76, -122.44, k=10, radius_km=5.0)
        assert all(av[i] for i in got), cls.name


def test_indexes_respect_the_radius(fleet):
    lat, lon, av = fleet
    for cls in G.INDEXES:
        idx = cls(lat.copy(), lon.copy(), av)
        got = idx.query(37.76, -122.44, k=10, radius_km=1.0)
        for i in got:
            assert G.haversine(lat[i], lon[i], 37.76, -122.44) <= 1.0 + 1e-9


def test_moving_a_courier_moves_it_between_cells(fleet):
    lat, lon, av = fleet
    for cls in (G.GridHash, G.HexIndex):
        idx = cls(lat.copy(), lon.copy(), av.copy())
        i = int(np.flatnonzero(av)[0])
        idx.available[i] = True
        idx.update(i, 37.90, -122.10)          # far away
        near_old = idx.query(lat[i], lon[i], k=50, radius_km=0.5)
        assert i not in near_old, cls.name
        near_new = idx.query(37.90, -122.10, k=50, radius_km=0.5)
        assert i in near_new, cls.name


# --------------------------------------------------------------------------
# assignment invariant -- the load-bearing test
# --------------------------------------------------------------------------
def test_exactly_one_courier_wins_each_order():
    disp = D.Dispatcher()
    n_orders, offers_each = 200, 8
    for i in range(n_orders):
        disp.create_order(D.Order(i, 37.76, -122.44, 0.0, 8.0))
    ids = []
    for i in range(n_orders):
        for c in range(offers_each):
            ids.append(disp.make_offer(i, i * offers_each + c, ttl=60.0,
                                       now=0.0).offer_id)

    wins, lock = [], threading.Lock()
    barrier = threading.Barrier(16)

    def worker(chunk):
        got = 0
        barrier.wait()
        for oid in chunk:
            if disp.accept(oid, now=1.0):
                got += 1
        with lock:
            wins.append(got)

    ts = [threading.Thread(target=worker, args=(ids[i::16],)) for i in range(16)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    assert sum(wins) == n_orders
    assert disp.check_invariants() == []
    for o in disp.orders.values():
        accepted = [disp.offers[i] for i in o.offers
                    if disp.offers[i].state == "accepted"]
        assert len(accepted) == 1
        assert o.assigned_to == accepted[0].courier_id


def test_a_courier_cannot_hold_two_orders():
    """The bug the invariant checker caught: nothing stopped one courier winning
    two different orders concurrently."""
    disp = D.Dispatcher()
    disp.create_order(D.Order(1, 37.76, -122.44, 0.0, 8.0))
    disp.create_order(D.Order(2, 37.76, -122.44, 0.0, 8.0))
    a = disp.make_offer(1, courier_id=99, ttl=60.0, now=0.0)
    b = disp.make_offer(2, courier_id=99, ttl=60.0, now=0.0)
    assert disp.accept(a.offer_id, now=1.0) is True
    assert disp.accept(b.offer_id, now=1.0) is False
    assert disp.check_invariants() == []


def test_expired_offers_cannot_be_accepted():
    disp = D.Dispatcher()
    disp.create_order(D.Order(1, 37.76, -122.44, 0.0, 8.0))
    off = disp.make_offer(1, 5, ttl=10.0, now=0.0)
    assert disp.accept(off.offer_id, now=99.0) is False
    assert disp.orders[1].assigned_to is None


def test_declined_offers_cannot_be_accepted():
    disp = D.Dispatcher()
    disp.create_order(D.Order(1, 37.76, -122.44, 0.0, 8.0))
    off = disp.make_offer(1, 5, ttl=60.0, now=0.0)
    disp.decline(off.offer_id)
    assert disp.accept(off.offer_id, now=1.0) is False


# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------
def test_batched_matching_is_never_worse_in_total_distance():
    """The Hungarian solution minimises TOTAL cost by construction, so it cannot
    lose to greedy on that objective. If it does, the cost matrix is wrong."""
    rng = np.random.default_rng(2)
    for trial in range(25):
        n_c, n_o = 40, 6
        lat = 37.76 + rng.normal(0, 0.03, n_c)
        lon = -122.44 + rng.normal(0, 0.03, n_c)
        av = np.ones(n_c, bool)
        orders = [D.Order(i, 37.76 + rng.normal(0, 0.03),
                          -122.44 + rng.normal(0, 0.03), 0.0, 8.0)
                  for i in range(n_o)]
        g = D.greedy_match(orders, lat, lon, av)
        b = D.batched_match(orders, lat, lon, av)

        def total(assign):
            return sum(G.haversine(lat[c], lon[c], o.lat, o.lon)
                       for o in orders if (c := assign.get(o.order_id)) is not None)
        assert total(b) <= total(g) + 1e-9


def test_no_courier_is_assigned_twice_by_either_matcher():
    rng = np.random.default_rng(3)
    lat = 37.76 + rng.normal(0, 0.03, 30)
    lon = -122.44 + rng.normal(0, 0.03, 30)
    av = np.ones(30, bool)
    orders = [D.Order(i, 37.76, -122.44, 0.0, 8.0) for i in range(8)]
    for assign in (D.greedy_match(orders, lat, lon, av),
                   D.batched_match(orders, lat, lon, av)):
        vals = list(assign.values())
        assert len(vals) == len(set(vals))


# --------------------------------------------------------------------------
# ETA and the tracking surface
# --------------------------------------------------------------------------
def test_eta_interval_widens_with_staleness():
    narrow = E.eta_interval(20.0, age_seconds=0.0)
    wide = E.eta_interval(20.0, age_seconds=300.0)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_tracking_states_progress_with_age():
    assert E.tracking_state(5) == "live"
    assert E.tracking_state(90) == "delayed_signal"
    assert E.tracking_state(600) == "signal_lost"


def test_the_dot_comes_off_the_map_when_signal_is_lost():
    """The WISMO principle, asserted. A frozen dot rendered as live is the
    failure this whole surface exists to prevent."""
    live = E.render_tracking(37.7, -122.4, 5, 18.0)
    lost = E.render_tracking(37.7, -122.4, 600, 18.0)
    assert live.show_dot is True
    assert lost.show_dot is False
    assert "lost signal" in lost.message.lower()


def test_bad_news_travels_faster_than_good_news():
    """An ETA jumping UP must pass through nearly undamped; drifting DOWN is
    smoothed. Damping a delay is a lie of omission."""
    up = E.smooth_eta(20.0, 40.0)
    down = E.smooth_eta(40.0, 20.0)
    assert up > 35.0, "a 20-minute delay must reach the customer immediately"
    assert down > 25.0, "improvements should be damped, not snapped"


def test_eta_grows_with_distance():
    near = E.predict_eta(37.760, -122.440, 37.762, -122.442, 37.764, -122.444, 5.0)
    far = E.predict_eta(37.700, -122.500, 37.762, -122.442, 37.764, -122.444, 5.0)
    assert far > near


def test_eta_evaluation_reports_coverage():
    pred = np.array([20.0, 25.0, 30.0])
    actual = np.array([22.0, 40.0, 29.0])
    low, high = pred - 5, pred + 5
    _tab, overall = E.evaluate_eta(pred, actual, np.array(["a", "a", "a"]),
                                   low, high)
    assert overall["coverage_80"] == pytest.approx(2 / 3)
    assert overall["mae"] == pytest.approx((2 + 15 + 1) / 3)
