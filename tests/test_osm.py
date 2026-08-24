"""A real street network, against the synthetic grid that stood in for it.

Skips when the extract is absent, so the suite passes without it.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import osmnet as OS  # noqa: E402


needs_osm = pytest.mark.skipif(not OS.available(),
                               reason="no OSM extract at %s" % OS.DATA)


@pytest.fixture(scope="module")
def net():
    if not OS.available():
        pytest.skip("no OSM extract")
    return OS.OSMNetwork()


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def test_maxspeed_parsing_handles_what_osm_actually_contains():
    """`maxspeed` is free text written by volunteers."""
    assert OS._parse_maxspeed("30") == pytest.approx(30.0)
    assert OS._parse_maxspeed("30 mph") == pytest.approx(48.28, abs=0.05)
    assert OS._parse_maxspeed("DE:urban") is None
    assert OS._parse_maxspeed("walk") is None
    assert OS._parse_maxspeed(None) is None
    assert OS._parse_maxspeed("") is None


@needs_osm
def test_the_network_is_real_and_substantially_one_way(net):
    assert net.n_ways > 1000
    assert 0.2 < net.n_oneway / net.n_ways < 0.9


@needs_osm
def test_routing_is_restricted_to_the_largest_strongly_connected_component(net):
    """A bounding box cuts ways at its edge, leaving stubs a courier can enter
    and not leave. 11.7% of random pairs were unreachable before this."""
    assert len(net.node_ids) < len(net.all_node_ids)
    assert len(net.node_ids) / len(net.all_node_ids) > 0.8


@needs_osm
def test_every_pair_inside_the_component_is_reachable(net):
    """That is what 'strongly connected' means, and it is worth asserting
    because getting the one-way direction backwards would still leave a large
    component -- just the wrong one."""
    prof = OS.detour_profile(net, n=60, seed=3)
    assert prof["unreachable"] == 0
    assert net.fallbacks == 0


# --------------------------------------------------------------------------
# the geometry findings
# --------------------------------------------------------------------------
@needs_osm
def test_a_route_is_never_shorter_than_the_straight_line(net):
    prof = OS.detour_profile(net, n=60, seed=4)
    assert prof["median"] >= 1.0
    assert prof["p99"] >= prof["median"]


@needs_osm
def test_the_real_detour_depends_on_trip_length_and_the_grid_does_not():
    """The headline. A barrier or a one-way pair costs a fixed number of metres,
    which is a bigger fraction of a short trip -- and deliveries are short."""
    from src import roadnet as RN
    net = OS.OSMNetwork()
    grid = RN.RoadNetwork(n_rows=40, n_cols=40, seed=1)

    def osm_sample(rng):
        a, b = rng.choice(net.node_ids), rng.choice(net.node_ids)
        if a == b:
            return None, None, 0.0
        return a, b, OS._haversine_m(net.nodes[a], net.nodes[b])

    def grid_sample(rng):
        a = grid.node_latlon(rng.randrange(grid.R), rng.randrange(grid.C))
        b = grid.node_latlon(rng.randrange(grid.R), rng.randrange(grid.C))
        return a, b, OS._haversine_m(a, b)

    real = OS.detour_by_distance(lambda a, b: net.route(a, b), osm_sample,
                                 n=300, seed=5)
    synth = OS.detour_by_distance(
        lambda a, b: grid.route(a[0], a[1], b[0], b[1]), grid_sample,
        n=300, seed=5)
    assert real[0]["median"] > real[-1]["median"] * 1.15, real
    assert abs(synth[0]["median"] - synth[-1]["median"]) < 0.15, synth
    assert real[0]["median"] > synth[0]["median"] * 1.2


@needs_osm
def test_one_way_asymmetry_is_the_common_case_not_an_edge_case(net):
    """The grid measured 7.7% of routes asymmetric. A scorer caching 'the
    distance between a and b' is wrong far more often than that here."""
    a = OS.asymmetry(net, n=60, seed=1)
    assert a["share_differing"] > 0.5
    assert a["mean_gap"] > 0.01


@needs_osm
def test_a_route_to_itself_is_free(net):
    nid = net.node_ids[0]
    r = net.route(nid, nid)
    assert r["seconds"] == 0.0
    assert r["detour_factor"] == 1.0


@needs_osm
def test_slowing_every_edge_slows_the_route_proportionally(net):
    a, b = net.node_ids[0], net.node_ids[len(net.node_ids) // 2]
    fast = net.route(a, b, speed_factor=1.0)
    slow = net.route(a, b, speed_factor=0.5)
    if fast["unreachable"]:
        pytest.skip("pair not connected")
    assert slow["seconds"] == pytest.approx(2 * fast["seconds"], rel=1e-6)
    assert slow["metres"] == pytest.approx(fast["metres"], rel=1e-6)


@needs_osm
def test_short_pairs_are_excluded_from_the_profile(net):
    """Under a few hundred metres the ratio is dominated by which side of the
    block you started on, which is not a fact about the network."""
    loose = OS.detour_profile(net, n=40, seed=7, min_straight_m=50.0)
    tight = OS.detour_profile(net, n=40, seed=7, min_straight_m=1500.0)
    assert tight["mean"] < loose["mean"]
