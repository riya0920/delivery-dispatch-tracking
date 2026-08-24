"""A real street network from OpenStreetMap — the caveat this project kept.

WHAT THIS PROJECT SAID, TWICE
-----------------------------
"Still not OSM. A synthetic grid with one-ways and traffic is a better model than
haversine and is not a map. Real street networks have rivers, bridges, dead ends
and turn restrictions that a grid cannot express, and the detour factor on a real
metro is a different number."

That last clause is a prediction, and this module is what tests it. The synthetic
grid reports a mean detour factor of 1.329 and a p99 of 1.718. If a real city
comes out close to that, the grid was a good enough stand-in and the caveat was
conservative. If it comes out far away, every ETA and every allocation decision
tuned on the grid inherits the gap.

HOW THE DATA GETS HERE, AND WHY IT IS NOT A PBF
-----------------------------------------------
`pyosmium` installs and then fails to import: its compiled extension is blocked
by this machine's Application Control policy. That is a real block and not an
inherited assumption -- the wheel downloads, the DLL is refused.

So the data comes from the Overpass API as JSON, which needs no compiled parser
at all. It is a bounding box rather than a whole-planet extract, which is a
smaller claim and an honest one: this is one city's road graph, not a routing
service.

WHAT IS REAL HERE AND WHAT IS NOT
---------------------------------
Real: the geometry, the topology, one-way directions, road classes, and any
`maxspeed` the mappers recorded.

Not real: traffic. OSM has no idea what time it is. The time-of-day multipliers
are SE-3's own, carried over from the synthetic grid, so the RUSH HOUR numbers
are a property of this project's assumption and not of Wilmington. Only the
geometry results below should be read as measurements of a real city.

Also not modelled: turn restrictions (`type=restriction` relations), which is the
one structural feature of real networks this still cannot express.
"""
from __future__ import annotations

import heapq
import json
import math
import os

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    os.pardir, ".vendor", "wilmington.json")

# km/h by road class, used only where the mappers recorded no maxspeed
CLASS_SPEED = {
    "motorway": 100.0, "motorway_link": 60.0,
    "trunk": 80.0, "trunk_link": 50.0,
    "primary": 60.0, "primary_link": 40.0,
    "secondary": 50.0, "secondary_link": 35.0,
    "tertiary": 40.0, "tertiary_link": 30.0,
    "unclassified": 35.0, "residential": 30.0, "living_street": 15.0,
}
DEFAULT_SPEED = 30.0


def available(path: str = DATA) -> bool:
    return os.path.exists(path)


def _haversine_m(a, b) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    h = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * R * math.asin(min(1.0, math.sqrt(h)))


def _parse_maxspeed(value) -> float | None:
    """OSM maxspeed is free text: '30', '30 mph', 'DE:urban', 'walk'."""
    if not value:
        return None
    v = str(value).strip().lower()
    try:
        if v.endswith("mph"):
            return float(v.replace("mph", "").strip()) * 1.609344
        return float(v)
    except ValueError:
        return None


class OSMNetwork:
    """A directed road graph built from Overpass JSON.

    One-ways are honoured in the direction the way is drawn, including the
    `oneway=-1` case, which reverses it. Getting that backwards is the exact bug
    the synthetic grid shipped in its first version -- there it stranded the
    graph and every route fell through to haversine. Here it would silently make
    the network *more* connected than reality rather than less, which is harder
    to notice, so `reachable_share` is measured rather than assumed.
    """

    def __init__(self, path: str = DATA):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        self.nodes: dict[int, tuple[float, float]] = {}
        ways = []
        for el in raw["elements"]:
            if el["type"] == "node":
                self.nodes[el["id"]] = (el["lat"], el["lon"])
            elif el["type"] == "way":
                ways.append(el)

        self.adj: dict[int, list[tuple[int, float, float]]] = {}
        self.n_ways = 0
        self.n_oneway = 0
        used = set()
        for w in ways:
            tags = w.get("tags", {})
            hw = tags.get("highway")
            if hw is None:
                continue
            speed = (_parse_maxspeed(tags.get("maxspeed"))
                     or CLASS_SPEED.get(hw, DEFAULT_SPEED))
            speed = max(speed, 5.0)
            ow = str(tags.get("oneway", "no")).lower()
            roundabout = str(tags.get("junction", "")).lower() in (
                "roundabout", "circular")
            forward_only = ow in ("yes", "true", "1") or roundabout
            backward_only = ow == "-1"
            if forward_only or backward_only:
                self.n_oneway += 1
            self.n_ways += 1
            refs = [r for r in w["nodes"] if r in self.nodes]
            for a, b in zip(refs, refs[1:]):
                d = _haversine_m(self.nodes[a], self.nodes[b])
                if d <= 0:
                    continue
                t = d / (speed / 3.6)
                if not backward_only:
                    self.adj.setdefault(a, []).append((b, d, t))
                if not forward_only:
                    self.adj.setdefault(b, []).append((a, d, t))
                used.add(a)
                used.add(b)
        self.all_node_ids = sorted(used)
        self.node_ids = self._largest_scc()
        self.fallbacks = 0

    def _largest_scc(self) -> list[int]:
        """Restrict routing to the largest strongly-connected component.

        A bounding box cuts ways at its edge, so the raw graph contains stubs
        that can be entered and not left -- 11.7% of random pairs were
        unreachable before this, and almost none of that is real: it is the box.
        Keeping them would report the extract's border as a property of
        Wilmington's streets.

        STRONGLY connected, not weakly: with one-ways, "there is a path between
        them if you ignore direction" is not the question a courier asks.
        Iterative Kosaraju, because a recursive Tarjan overflows the stack on
        16k nodes.
        """
        order, seen = [], set()
        for start in self.all_node_ids:
            if start in seen:
                continue
            stack = [(start, iter(self.adj.get(start, ())))]
            seen.add(start)
            while stack:
                node, it = stack[-1]
                advanced = False
                for nb, _d, _t in it:
                    if nb not in seen:
                        seen.add(nb)
                        stack.append((nb, iter(self.adj.get(nb, ()))))
                        advanced = True
                        break
                if not advanced:
                    order.append(stack.pop()[0])
        rev: dict[int, list[int]] = {}
        for a, edges in self.adj.items():
            for b, _d, _t in edges:
                rev.setdefault(b, []).append(a)
        assigned, best = set(), []
        for node in reversed(order):
            if node in assigned:
                continue
            comp, stack = [], [node]
            assigned.add(node)
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for prev in rev.get(cur, ()):
                    if prev not in assigned:
                        assigned.add(prev)
                        stack.append(prev)
            if len(comp) > len(best):
                best = comp
        return sorted(best)

    # ------------------------------------------------------------------
    def nearest(self, lat: float, lon: float) -> int:
        best, bd = None, float("inf")
        for nid in self.node_ids:
            d = _haversine_m(self.nodes[nid], (lat, lon))
            if d < bd:
                best, bd = nid, d
        return best

    def route(self, src: int, dst: int, speed_factor: float = 1.0) -> dict:
        """A* on travel time. `speed_factor` scales every edge's speed.

        The heuristic is straight-line distance divided by the fastest speed in
        the graph, which is admissible: no edge can beat it.
        """
        if src == dst:
            return dict(seconds=0.0, metres=0.0, straight_metres=0.0,
                        detour_factor=1.0, hops=0, unreachable=False)
        goal = self.nodes[dst]
        fastest = 120.0 / 3.6

        def h(n):
            return _haversine_m(self.nodes[n], goal) / fastest

        dist = {src: 0.0}
        metres = {src: 0.0}
        hops = {src: 0}
        seen = set()
        pq = [(h(src), 0.0, src)]
        while pq:
            _, g, n = heapq.heappop(pq)
            if n in seen:
                continue
            seen.add(n)
            if n == dst:
                straight = _haversine_m(self.nodes[src], goal)
                return dict(seconds=g, metres=metres[n],
                            straight_metres=straight,
                            detour_factor=(metres[n] / straight
                                           if straight > 1 else 1.0),
                            hops=hops[n], unreachable=False)
            for nb, d, t in self.adj.get(n, ()):
                ng = g + t / max(speed_factor, 1e-6)
                if ng < dist.get(nb, float("inf")):
                    dist[nb] = ng
                    metres[nb] = metres[n] + d
                    hops[nb] = hops[n] + 1
                    heapq.heappush(pq, (ng + h(nb), ng, nb))
        self.fallbacks += 1
        straight = _haversine_m(self.nodes[src], goal)
        return dict(seconds=straight / (30 / 3.6), metres=straight,
                    straight_metres=straight, detour_factor=1.0, hops=0,
                    unreachable=True)


def detour_profile(net: OSMNetwork, n: int = 200, seed: int = 0,
                   min_straight_m: float = 500.0) -> dict:
    """Route ÷ straight line over random node pairs.

    Pairs closer than `min_straight_m` are skipped: over 200 m the ratio is
    dominated by which side of the block you started on, and averaging those in
    would make the network look worse than it is for reasons that are not about
    the network.
    """
    import random

    rng = random.Random(seed)
    ratios, unreachable, times = [], 0, []
    ids = net.node_ids
    tries = 0
    while len(ratios) + unreachable < n and tries < n * 40:
        tries += 1
        a, b = rng.choice(ids), rng.choice(ids)
        if a == b:
            continue
        if _haversine_m(net.nodes[a], net.nodes[b]) < min_straight_m:
            continue
        r = net.route(a, b)
        if r["unreachable"]:
            unreachable += 1
            continue
        ratios.append(r["detour_factor"])
        times.append(r["seconds"])
    ratios.sort()

    def q(p):
        if not ratios:
            return float("nan")
        return ratios[min(len(ratios) - 1, int(p * len(ratios)))]

    return dict(n=len(ratios), unreachable=unreachable,
                unreachable_share=unreachable / max(len(ratios) + unreachable, 1),
                mean=sum(ratios) / max(len(ratios), 1),
                median=q(0.50), p90=q(0.90), p99=q(0.99),
                max=ratios[-1] if ratios else float("nan"),
                mean_seconds=sum(times) / max(len(times), 1))


def asymmetry(net: OSMNetwork, n: int = 150, seed: int = 1,
              min_straight_m: float = 500.0) -> dict:
    """How often does A->B differ from B->A, and by how much?

    haversine guarantees they are equal. One-ways are why they are not, and a
    scorer that caches "the distance between a and b" is wrong whenever they
    differ.
    """
    import random

    rng = random.Random(seed)
    diffs, both = [], 0
    ids = net.node_ids
    tries = 0
    while both < n and tries < n * 40:
        tries += 1
        a, b = rng.choice(ids), rng.choice(ids)
        if a == b or _haversine_m(net.nodes[a], net.nodes[b]) < min_straight_m:
            continue
        f = net.route(a, b)
        r = net.route(b, a)
        if f["unreachable"] or r["unreachable"]:
            continue
        both += 1
        m = max(f["seconds"], r["seconds"])
        if m > 0:
            diffs.append(abs(f["seconds"] - r["seconds"]) / m)
    diffs.sort()
    return dict(pairs=both,
                share_differing=sum(1 for d in diffs if d > 0.01) / max(len(diffs), 1),
                mean_gap=sum(diffs) / max(len(diffs), 1),
                p90_gap=diffs[int(0.9 * len(diffs))] if diffs else float("nan"))


def detour_by_distance(route_fn, sample_fn, buckets_m=(500, 1000, 2000, 3000,
                                                       4000, 6000, 8000),
                       n: int = 500, seed: int = 5) -> list[dict]:
    """Detour factor as a function of how far the trip actually is.

    THE MEASUREMENT THAT MATTERS FOR A DELIVERY MARKETPLACE. A single mean detour
    multiplier assumes the ratio is a constant of the network. On a real street
    graph it is not: short trips detour much worse, because a barrier or a
    one-way pair costs a fixed number of metres and that is a larger fraction of
    a short trip. Deliveries are short.

    `route_fn(a, b) -> dict` and `sample_fn(rng) -> (a, b, straight_metres)` keep
    this usable for both the OSM graph and the synthetic grid, so the comparison
    is one function applied twice rather than two functions that might differ.
    """
    import random
    import statistics

    rng = random.Random(seed)
    rows: dict[int, list[float]] = {}
    tries = 0
    got = 0
    while got < n and tries < n * 60:
        tries += 1
        a, b, straight = sample_fn(rng)
        if a is None or straight < buckets_m[0]:
            continue
        r = route_fn(a, b)
        if r.get("unreachable"):
            continue
        edge = max([m for m in buckets_m if m <= straight], default=None)
        if edge is None:
            continue
        rows.setdefault(edge, []).append(r["detour_factor"])
        got += 1
    out = []
    for edge in sorted(rows):
        v = sorted(rows[edge])
        if len(v) < 8:
            continue
        out.append(dict(from_m=edge, n=len(v),
                        median=statistics.median(v),
                        p90=v[int(0.9 * len(v))], max=v[-1]))
    return out
