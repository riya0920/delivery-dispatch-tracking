"""Four ways to answer "nearest available couriers to point X", benchmarked.

SUBSTITUTIONS, STATED FIRST: PostGIS, Redis and the h3 library are not available
in this offline environment. What is benchmarked is the ALGORITHMIC SHAPE of each
approach, in-process, on the same data:

  linear_scan   the naive full-table scan -- the thing a design review is
                supposed to reject, included because "we benchmarked it" is a
                much stronger claim than "everyone knows it is slow"
  kdtree        scipy cKDTree, standing in for a PostGIS GiST index: a spatial
                tree, rebuilt on write, excellent for reads
  grid_hash     uniform lat/lon grid buckets, standing in for Redis GEO: O(1)
                updates, query touches a ring of neighbouring cells
  hex_h3        axial hexagonal binning with a resolution parameter, standing in
                for H3: same idea as the grid with uniform neighbour distance
                and no cell-corner pathology

WHAT THE SUBSTITUTION COSTS, so nobody quotes these as database numbers: no
network hop, no serialisation, no concurrency control, no durability. A real
PostGIS query pays milliseconds of round-trip that dwarf the microseconds
measured here. What DOES transfer is the shape -- how each structure's read and
write costs scale as the fleet grows, and the staleness each one forces you to
accept.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.spatial import cKDTree

EARTH_KM = 6371.0


def haversine(lat1, lon1, lat2, lon2):
    p = math.pi / 180.0
    a = (0.5 - np.cos((lat2 - lat1) * p) / 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2)
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class LinearScan:
    """The baseline that exists to be beaten, and to be beaten with a number."""
    name = "linear_scan"

    def __init__(self, lat, lon, available):
        self.lat, self.lon, self.available = lat, lon, available

    def update(self, idx, lat, lon):
        self.lat[idx] = lat
        self.lon[idx] = lon

    def query(self, lat, lon, k=5, radius_km=5.0):
        d = haversine(self.lat, self.lon, lat, lon)
        d = np.where(self.available, d, np.inf)
        order = np.argpartition(d, min(k, len(d) - 1))[:k]
        order = order[np.argsort(d[order])]
        return [int(i) for i in order if d[i] <= radius_km]


class KDTreeIndex:
    """Spatial tree. Reads are excellent; the tree must be REBUILT on write,
    which is the whole story for a fleet that moves every second."""
    name = "kdtree"

    def __init__(self, lat, lon, available, rebuild_every: int = 2000):
        self.lat, self.lon, self.available = lat, lon, available
        self.rebuild_every = rebuild_every
        self.dirty = 0
        self._build()

    def _build(self):
        self.xyz = np.column_stack(_to_xyz(self.lat, self.lon))
        self.tree = cKDTree(self.xyz)
        self.dirty = 0

    def update(self, idx, lat, lon):
        self.lat[idx] = lat
        self.lon[idx] = lon
        self.dirty += 1
        # AMORTISED REBUILD is the honest trade: rebuilding on every ping is
        # unaffordable, so the index is allowed to go stale between rebuilds and
        # the staleness is MEASURED rather than hidden.
        if self.dirty >= self.rebuild_every:
            self._build()

    def query(self, lat, lon, k=5, radius_km=5.0):
        pt = np.array(_to_xyz(np.array([lat]), np.array([lon]))).ravel()
        kk = min(k * 8, len(self.lat))
        _d, idx = self.tree.query(pt, k=kk)
        out = []
        for i in np.atleast_1d(idx):
            i = int(i)
            if not self.available[i]:
                continue
            if haversine(self.lat[i], self.lon[i], lat, lon) <= radius_km:
                out.append(i)
            if len(out) >= k:
                break
        return out


def _to_xyz(lat, lon):
    p = math.pi / 180.0
    return (EARTH_KM * np.cos(lat * p) * np.cos(lon * p),
            EARTH_KM * np.cos(lat * p) * np.sin(lon * p),
            EARTH_KM * np.sin(lat * p))


class GridHash:
    """Uniform lat/lon cells. O(1) update, query scans a ring of cells.

    This is the Redis GEO shape: writes are cheap and always fresh, reads pay for
    scanning neighbours. Note the flaw the hex index fixes -- lat/lon cells are
    not square, and a courier just across a cell CORNER is further away in cells
    than one across an edge, so a 1-ring query has a direction-dependent radius.
    """
    name = "grid_hash"

    def __init__(self, lat, lon, available, cell_km: float = 1.0):
        self.lat, self.lon, self.available = lat, lon, available
        self.cell_deg = cell_km / 111.0
        self.cells: dict[tuple[int, int], set[int]] = {}
        self.where: dict[int, tuple[int, int]] = {}
        for i in range(len(lat)):
            c = self._cell(lat[i], lon[i])
            self.cells.setdefault(c, set()).add(i)
            self.where[i] = c

    def _cell(self, lat, lon):
        return (int(math.floor(lat / self.cell_deg)),
                int(math.floor(lon / self.cell_deg)))

    def update(self, idx, lat, lon):
        self.lat[idx] = lat
        self.lon[idx] = lon
        c = self._cell(lat, lon)
        old = self.where.get(idx)
        if old != c:
            if old is not None:
                self.cells[old].discard(idx)
            self.cells.setdefault(c, set()).add(idx)
            self.where[idx] = c

    def query(self, lat, lon, k=5, radius_km=5.0):
        r = max(1, int(math.ceil(radius_km / (self.cell_deg * 111.0))))
        cy, cx = self._cell(lat, lon)
        cand = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                cand.extend(self.cells.get((cy + dy, cx + dx), ()))
        if not cand:
            return []
        cand = np.array(cand)
        cand = cand[self.available[cand]]
        if not len(cand):
            return []
        d = haversine(self.lat[cand], self.lon[cand], lat, lon)
        keep = d <= radius_km
        cand, d = cand[keep], d[keep]
        return [int(i) for i in cand[np.argsort(d)][:k]]


class HexIndex:
    """Axial hexagonal binning -- the H3 shape without the H3 dependency.

    Hexagons have six equidistant neighbours, so a k-ring is a genuine radius
    rather than a square whose corners are 1.41x further than its edges. That
    matters for dispatch: a square-cell ring quietly biases which couriers get
    considered depending on where in the cell the restaurant sits.
    """
    name = "hex_h3"

    def __init__(self, lat, lon, available, cell_km: float = 1.0):
        self.lat, self.lon, self.available = lat, lon, available
        self.size = cell_km / 111.0
        self.cells: dict[tuple[int, int], set[int]] = {}
        self.where: dict[int, tuple[int, int]] = {}
        for i in range(len(lat)):
            c = self._cell(lat[i], lon[i])
            self.cells.setdefault(c, set()).add(i)
            self.where[i] = c

    def _cell(self, lat, lon):
        # pointy-top axial coordinates
        q = (math.sqrt(3) / 3 * lon - 1.0 / 3 * lat) / self.size
        r = (2.0 / 3 * lat) / self.size
        return _hex_round(q, r)

    def update(self, idx, lat, lon):
        self.lat[idx] = lat
        self.lon[idx] = lon
        c = self._cell(lat, lon)
        old = self.where.get(idx)
        if old != c:
            if old is not None:
                self.cells[old].discard(idx)
            self.cells.setdefault(c, set()).add(idx)
            self.where[idx] = c

    def query(self, lat, lon, k=5, radius_km=5.0):
        ring = max(1, int(math.ceil(radius_km / (self.size * 111.0))))
        q0, r0 = self._cell(lat, lon)
        cand = []
        for dq in range(-ring, ring + 1):
            lo = max(-ring, -dq - ring)
            hi = min(ring, -dq + ring)
            for dr in range(lo, hi + 1):
                cand.extend(self.cells.get((q0 + dq, r0 + dr), ()))
        if not cand:
            return []
        cand = np.array(cand)
        cand = cand[self.available[cand]]
        if not len(cand):
            return []
        d = haversine(self.lat[cand], self.lon[cand], lat, lon)
        keep = d <= radius_km
        cand, d = cand[keep], d[keep]
        return [int(i) for i in cand[np.argsort(d)][:k]]


def _hex_round(q, r):
    x, z = q, r
    y = -x - z
    rx, ry, rz = round(x), round(y), round(z)
    dx, dy, dz = abs(rx - x), abs(ry - y), abs(rz - z)
    if dx > dy and dx > dz:
        rx = -ry - rz
    elif dy > dz:
        ry = -rx - rz
    else:
        rz = -rx - ry
    return (int(rx), int(rz))


INDEXES = [LinearScan, KDTreeIndex, GridHash, HexIndex]


def benchmark(index_cls, lat, lon, available, queries, updates_per_query: int,
              rng, k=5, radius_km=3.0):
    """Interleave writes and reads the way a live fleet does.

    Benchmarking reads alone would flatter the tree index enormously, which is
    exactly the mistake that gets made: couriers ping every 1-5 seconds, so the
    write path is the dominant load and any structure that pays to absorb a write
    pays it 10,000 times a second.
    """
    lat, lon = lat.copy(), lon.copy()
    idx = index_cls(lat, lon, available)
    n = len(lat)

    t_up = 0.0
    t_q = 0.0
    n_up = 0
    recall_hits, recall_total = 0, 0
    truth_index = LinearScan(lat.copy(), lon.copy(), available)

    lat_q = rng.random(len(queries)) * 0 + queries[:, 0]
    lon_q = queries[:, 1]

    for qi in range(len(queries)):
        movers = rng.integers(0, n, updates_per_query)
        dlat = rng.normal(0, 0.0008, updates_per_query)
        dlon = rng.normal(0, 0.0008, updates_per_query)
        t0 = time.perf_counter()
        for j, m in enumerate(movers):
            idx.update(int(m), lat[m] + dlat[j], lon[m] + dlon[j])
        t_up += time.perf_counter() - t0
        n_up += updates_per_query
        for j, m in enumerate(movers):
            truth_index.update(int(m), truth_index.lat[m] + dlat[j],
                               truth_index.lon[m] + dlon[j])

        t0 = time.perf_counter()
        got = idx.query(lat_q[qi], lon_q[qi], k=k, radius_km=radius_km)
        t_q += time.perf_counter() - t0

        want = truth_index.query(lat_q[qi], lon_q[qi], k=k, radius_km=radius_km)
        if want:
            recall_hits += len(set(got) & set(want))
            recall_total += len(want)

    return dict(
        index=index_cls.name,
        update_us=1e6 * t_up / max(n_up, 1),
        query_ms=1e3 * t_q / len(queries),
        query_qps=len(queries) / t_q if t_q > 0 else float("inf"),
        recall_vs_exact=recall_hits / recall_total if recall_total else float("nan"))
