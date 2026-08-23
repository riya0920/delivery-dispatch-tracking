"""A road network with one-ways, turn penalties and time-varying traffic.

THE GAP, AND WHAT THIS IS AND IS NOT
------------------------------------
"No road network. No OSM extract, no OSRM -- distances are haversine, so every
ETA is optimistic by whatever the local street grid costs."

This is not OSM. There is no extract to download here and no routing server to
run, and calling a synthetic grid "a road network" without saying so would be the
same overclaim the haversine version was honest about. What it IS:

  * a directed graph with **one-way streets**, so the route out is not the route
    back and the distance function stops being symmetric;
  * **turn penalties**, because a left across traffic costs seconds that no
    distance metric contains;
  * **time-varying edge speeds**, so the same route takes different times at
    08:00 and 14:00.

WHY THOSE THREE AND NOT OTHERS
------------------------------
Each one breaks a property that haversine silently assumes and that dispatch code
silently relies on:

  symmetry      -- haversine says d(a,b) == d(b,a). With one-ways it does not,
                   and an assignment scorer that caches "distance between a and
                   b" is now wrong half the time.
  triangle ineq -- still holds on a graph, but the SHORTCUT haversine implies
                   ("these are 300 m apart so it is a one-minute walk") does not:
                   300 m apart across a river is a two-kilometre drive.
  time-invariance -- haversine has no clock. Every ETA built on it is the same at
                   rush hour as at midnight.

THE NUMBER THIS PRODUCES
------------------------
A **detour factor**: route distance divided by straight-line distance. It is the
correction every haversine-based ETA in the previous pass was missing, and
measuring it is more useful than any single routed ETA, because it can be applied
as a multiplier by anyone who cannot run a router.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

# A metro grid. Blocks are ~110 m, which is a normal city block.
BLOCK_M = 110.0
DEG_PER_M_LAT = 1.0 / 111_320.0


class RoadNetwork:
    """A directed grid with one-ways, turn costs and a traffic profile."""

    def __init__(self, n_rows: int = 40, n_cols: int = 40,
                 origin_lat: float = 37.74, origin_lon: float = -122.46,
                 one_way_share: float = 0.35, turn_penalty_s: float = 12.0,
                 seed: int = 0):
        self.R, self.C = n_rows, n_cols
        self.origin_lat, self.origin_lon = origin_lat, origin_lon
        self.turn_penalty_s = turn_penalty_s
        rng = np.random.default_rng(seed)

        # Every other avenue is one-way, alternating direction -- the Manhattan
        # pattern. Randomising which ones would be more general and less
        # realistic: real one-way systems alternate on purpose.
        self.one_way_col = np.zeros(n_cols, dtype=np.int8)
        for c in range(n_cols):
            if rng.random() < one_way_share:
                self.one_way_col[c] = 1 if c % 2 == 0 else -1
        self.one_way_row = np.zeros(n_rows, dtype=np.int8)
        for r in range(n_rows):
            if rng.random() < one_way_share:
                self.one_way_row[r] = 1 if r % 2 == 0 else -1

        # Arterials are faster and are every fifth street, which is what makes a
        # detour worthwhile: without a speed hierarchy the shortest path is
        # always the straightest one and routing adds nothing over haversine.
        self.arterial_row = np.array([r % 5 == 0 for r in range(n_rows)])
        self.arterial_col = np.array([c % 5 == 0 for c in range(n_cols)])

    # ------------------------------------------------------------------
    def node_latlon(self, r: int, c: int) -> tuple[float, float]:
        lat = self.origin_lat + r * BLOCK_M * DEG_PER_M_LAT
        lon = self.origin_lon + c * BLOCK_M * DEG_PER_M_LAT / math.cos(
            math.radians(self.origin_lat))
        return lat, lon

    def nearest_node(self, lat: float, lon: float) -> tuple[int, int]:
        r = int(round((lat - self.origin_lat) / (BLOCK_M * DEG_PER_M_LAT)))
        c = int(round((lon - self.origin_lon) * math.cos(math.radians(self.origin_lat))
                      / (BLOCK_M * DEG_PER_M_LAT)))
        return max(0, min(self.R - 1, r)), max(0, min(self.C - 1, c))

    def speed_kmh(self, r: int, c: int, horizontal: bool, hour: float) -> float:
        """Edge speed, by road class and time of day.

        The rush-hour profile is two humps. Speeds are per EDGE rather than
        global because congestion is not uniform: arterials slow down more in
        absolute terms and still stay faster than side streets, which is why a
        router keeps choosing them at 08:00 even though they are worse than at
        midnight.
        """
        arterial = self.arterial_row[r] if horizontal else self.arterial_col[c]
        base = 42.0 if arterial else 22.0
        am = math.exp(-0.5 * ((hour - 8.5) / 1.2) ** 2)
        pm = math.exp(-0.5 * ((hour - 17.8) / 1.4) ** 2)
        congestion = 1.0 - 0.45 * max(am, pm)
        return base * congestion

    def _neighbours(self, r: int, c: int):
        """Edges out of a node, respecting one-ways.

        A one-way street restricts travel ALONG it, not across it. Moving
        north/south is travel along COLUMN c, so it is governed by
        `one_way_col[c]`; moving east/west is travel along ROW r, governed by
        `one_way_row[r]`.

        The first version had these swapped -- it treated "row r is one-way
        southbound" as "you may not move north FROM row r", which restricts
        movement across the street rather than along it. The effect was that
        large parts of the grid became unreachable, A* returned nothing, and
        every route silently fell through to the haversine fallback -- so the
        router reported detour factors of exactly 1.00 and identical times at
        03:00 and 08:30. A routing bug that returns haversine looks precisely
        like a working router on a featureless grid, which is why the fallback
        counts itself.
        """
        if r + 1 < self.R and self.one_way_col[c] >= 0:
            yield (r + 1, c), False
        if r - 1 >= 0 and self.one_way_col[c] <= 0:
            yield (r - 1, c), False
        if c + 1 < self.C and self.one_way_row[r] >= 0:
            yield (r, c + 1), True
        if c - 1 >= 0 and self.one_way_row[r] <= 0:
            yield (r, c - 1), True

    def route(self, a_lat: float, a_lon: float, b_lat: float, b_lon: float,
              hour: float = 12.0) -> dict:
        """A* over the grid. Returns seconds, metres, and the detour factor.

        The heuristic is straight-line distance at the FASTEST speed in the
        network, which is admissible (it can never overestimate) and therefore
        keeps A* exact. An inadmissible heuristic would be faster and would
        silently return non-optimal routes, which is the wrong trade in a
        function whose whole purpose is to be more accurate than haversine.
        """
        start = self.nearest_node(a_lat, a_lon)
        goal = self.nearest_node(b_lat, b_lon)
        if start == goal:
            return dict(seconds=0.0, metres=0.0, straight_metres=0.0,
                        detour_factor=1.0, nodes=1)

        def h(n):
            dr, dc = abs(n[0] - goal[0]), abs(n[1] - goal[1])
            metres = math.hypot(dr, dc) * BLOCK_M
            return metres / (42.0 * 1000 / 3600)

        dist = {(start, None): 0.0}
        pq = [(h(start), 0.0, start, None)]
        came = {}
        best = None
        while pq:
            _f, g, node, heading = heapq.heappop(pq)
            if node == goal:
                best = (g, (node, heading))
                break
            if g > dist.get((node, heading), math.inf) + 1e-9:
                continue
            r, c = node
            for nxt, horizontal in self._neighbours(r, c):
                length = BLOCK_M
                spd = self.speed_kmh(r, c, horizontal, hour) * 1000 / 3600
                cost = length / max(spd, 1e-6)
                if heading is not None and heading != horizontal:
                    cost += self.turn_penalty_s
                key = (nxt, horizontal)
                ng = g + cost
                if ng < dist.get(key, math.inf) - 1e-9:
                    dist[key] = ng
                    came[key] = (node, heading)
                    heapq.heappush(pq, (ng + h(nxt), ng, nxt, horizontal))
        if best is None:
            # One-ways can strand a node. Falling back to haversine and SAYING
            # SO is better than returning infinity into a dispatch loop, and the
            # flag lets the caller count how often it happens.
            straight = haversine_m(a_lat, a_lon, b_lat, b_lon)
            return dict(seconds=straight / (22.0 * 1000 / 3600), metres=straight,
                        straight_metres=straight, detour_factor=1.0, nodes=0,
                        unreachable=True)

        seconds, key = best
        n_nodes = 0
        while key in came:
            key = came[key]
            n_nodes += 1
        metres = n_nodes * BLOCK_M
        straight = haversine_m(a_lat, a_lon, b_lat, b_lon)
        return dict(seconds=seconds, metres=metres, straight_metres=straight,
                    detour_factor=metres / straight if straight > 0 else 1.0,
                    nodes=n_nodes)


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def detour_profile(net: RoadNetwork, n: int = 300, hour: float = 12.0,
                   seed: int = 0) -> dict:
    """Sample routes and summarise the correction haversine was missing.

    The DISTRIBUTION matters more than the mean. A mean detour factor can be
    applied as a multiplier; a heavy right tail cannot, because the trips that
    blow the ETA are the ones in the tail and multiplying them by the mean
    leaves them just as wrong.
    """
    rng = np.random.default_rng(seed)
    factors, asym = [], []
    for _ in range(n):
        r1, c1 = rng.integers(0, net.R), rng.integers(0, net.C)
        r2, c2 = rng.integers(0, net.R), rng.integers(0, net.C)
        a = net.node_latlon(int(r1), int(c1))
        b = net.node_latlon(int(r2), int(c2))
        fwd = net.route(a[0], a[1], b[0], b[1], hour)
        rev = net.route(b[0], b[1], a[0], a[1], hour)
        if fwd["straight_metres"] < BLOCK_M:
            continue
        factors.append(fwd["detour_factor"])
        if rev["seconds"] > 0:
            asym.append(abs(fwd["seconds"] - rev["seconds"]) / max(fwd["seconds"], 1e-9))
    f = np.array(factors)
    return dict(n=len(f), mean=float(f.mean()), median=float(np.median(f)),
                p90=float(np.percentile(f, 90)), p99=float(np.percentile(f, 99)),
                max=float(f.max()),
                mean_asymmetry=float(np.mean(asym)) if asym else 0.0)
