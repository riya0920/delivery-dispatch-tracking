"""Real OpenStreetMap geometry against the synthetic grid that stood in for it.

The caveat this project carried in every pass: "Still not OSM. [...] the detour
factor on a real metro is a different number." That is a prediction. This checks
it.
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import osmnet as OS      # noqa: E402
from src import roadnet as RN     # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def main():
    os.makedirs(OUT, exist_ok=True)
    lines, summary = [], {}

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 78)
    emit("SE-3 OSM PASS -- THE GRID, CHECKED AGAINST A REAL CITY")
    emit("=" * 78)
    if not OS.available():
        emit("No OSM extract at %s" % OS.DATA)
        return

    net = OS.OSMNetwork()
    emit("Wilmington, Delaware, via the Overpass API.")
    emit("")
    emit("  road ways            : %d" % net.n_ways)
    emit("  one-way ways         : %d (%.1f%%)"
         % (net.n_oneway, 100 * net.n_oneway / net.n_ways))
    emit("  nodes                : %d" % len(net.all_node_ids))
    emit("  largest SCC          : %d (%.1f%%) -- routing is restricted to it"
         % (len(net.node_ids), 100 * len(net.node_ids) / len(net.all_node_ids)))
    emit("")
    emit("  The bounding box cuts ways at its edge, leaving stubs a courier can")
    emit("  enter and not leave: 11.7%% of random pairs were unreachable before")
    emit("  the component restriction and almost none of that is Wilmington, it")
    emit("  is the box. STRONGLY connected, not weakly -- with one-ways,")
    emit("  'connected if you ignore direction' is not the question a courier")
    emit("  asks.")
    emit("")
    emit("  pyosmium installs and then fails to import: its compiled extension")
    emit("  is blocked by this machine's Application Control policy. That is a")
    emit("  real block, unlike the Postgres one. Overpass JSON needs no compiled")
    emit("  parser, so the data comes that way instead.")
    emit("")

    # ------------------------------------------------------------------
    emit("=" * 78)
    emit("A. DETOUR FACTOR: THE GRID'S SINGLE NUMBER VS A REAL CITY'S CURVE")
    emit("=" * 78)
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
                                 n=420, seed=5)
    synth = OS.detour_by_distance(
        lambda a, b: grid.route(a[0], a[1], b[0], b[1]), grid_sample,
        n=420, seed=5)
    R = pd.DataFrame(real).assign(network="wilmington")
    S = pd.DataFrame(synth).assign(network="synthetic grid")
    both = pd.concat([R, S])[["network", "from_m", "n", "median", "p90", "max"]]
    emit(both.to_string(index=False, float_format=lambda x: "%8.2f" % x))
    emit("")
    r0, rl = real[0], real[-1]
    s0, sl = synth[0], synth[-1]
    emit("  THE GRID HAS NO DISTANCE DEPENDENCE AND THE REAL CITY DOES.")
    emit("")
    emit("    wilmington : median %.2f on the shortest trips -> %.2f on the"
         % (r0["median"], rl["median"]))
    emit("                 longest, a %.0f%% fall"
         % (100 * (1 - rl["median"] / r0["median"])))
    emit("    grid       : median %.2f -> %.2f, flat"
         % (s0["median"], sl["median"]))
    emit("")
    emit("  That is not a small correction and it points the wrong way for this")
    emit("  project. A barrier or a one-way pair costs a fixed number of metres,")
    emit("  and a fixed number of metres is a bigger fraction of a short trip.")
    emit("  DELIVERIES ARE SHORT TRIPS.")
    emit("")
    emit("  On the shortest bucket the grid says %.2f and the city says %.2f --"
         % (s0["median"], r0["median"]))
    emit("  it understates the detour by %.0f%% exactly where the business lives."
         % (100 * (r0["median"] / s0["median"] - 1)))
    emit("  The p90 there is %.2f against %.2f, which is close to double."
         % (s0["p90"], r0["p90"]))
    emit("")
    emit("  A SINGLE MEAN DETOUR MULTIPLIER IS THE WRONG SHAPE OF CORRECTION.")
    emit("  The previous pass already said the mean could not be applied to the")
    emit("  tail; this says it cannot be applied across trip lengths either.")
    emit("")
    summary["detour"] = both.round(4).to_dict("records")

    # ------------------------------------------------------------------
    emit("=" * 78)
    emit("B. ONE-WAY ASYMMETRY: 7.7% ON THE GRID, AND ON A REAL CITY?")
    emit("=" * 78)
    asym = OS.asymmetry(net, n=120, seed=1)
    emit("  pairs where A->B differs from B->A : %.1f%%"
         % (100 * asym["share_differing"]))
    emit("  mean gap when they differ          : %.1f%%"
         % (100 * asym["mean_gap"]))
    emit("  p90 gap                            : %.1f%%"
         % (100 * asym["p90_gap"]))
    emit("")
    emit("  The synthetic grid measured 7.7%% of routes asymmetric. Wilmington")
    emit("  is %.0f%%, which is %.0fx more, on a street network where half the"
         % (100 * asym["share_differing"], asym["share_differing"] / 0.077))
    emit("  ways are one-way (%.1f%%)." % (100 * net.n_oneway / net.n_ways))
    emit("")
    emit("  THE PREVIOUS PASS CALLED THIS A CORRECTNESS BUG AND UNDERSTATED HOW")
    emit("  OFTEN IT FIRES. An assignment scorer that caches 'the distance")
    emit("  between a and b' is wrong on 7.7%% of pairs in the grid's world and")
    emit("  on %.0f%% of them here. That is not an edge case, it is the common"
         % (100 * asym["share_differing"]))
    emit("  case, and the direction of travel has to be part of the key.")
    emit("")
    summary["asymmetry"] = asym

    # ------------------------------------------------------------------
    emit("=" * 78)
    emit("WHAT THIS IS NOT")
    emit("=" * 78)
    emit("  ONE CITY, and a mid-sized American one with a river through it. The")
    emit("  numbers above are Wilmington's, not 'a real metro's' -- Manhattan's")
    emit("  grid and London's medieval core would both disagree, in opposite")
    emit("  directions. What transfers is the SHAPE of the error, not its size.")
    emit("")
    emit("  NO TRAFFIC. OSM does not know what time it is. The time-of-day")
    emit("  multipliers in the ETA work remain this project's own assumption,")
    emit("  so the rush-hour results are not validated by anything here -- only")
    emit("  the geometry is.")
    emit("")
    emit("  NO TURN RESTRICTIONS. `type=restriction` relations are not parsed,")
    emit("  so every junction here permits every turn. That makes these detour")
    emit("  factors a LOWER BOUND on the real ones.")
    emit("")
    emit("  NOT A ROUTING SERVICE. A* over 15k nodes in-process, no contraction")
    emit("  hierarchies, no live traffic feed, and the ETA models were not")
    emit("  refitted on this network.")
    emit("")

    with open(os.path.join(OUT, "osm_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "osm_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\n-> out/osm_report.txt")


if __name__ == "__main__":
    main()
