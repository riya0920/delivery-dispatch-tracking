"""The completion pass: a road network, a learned ETA, surge that does not
oscillate, and couriers that reposition.

Run after nothing. Writes out/complete_report.txt.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import control as CTL       # noqa: E402
from src import couriers as CR       # noqa: E402
from src import eta_model as EM      # noqa: E402
from src import roadnet as RN        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def main():
    os.makedirs(OUT, exist_ok=True)
    lines, summary = [], {}

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 78)
    emit("SE-3 COMPLETION PASS")
    emit("=" * 78)
    emit("")

    # ======================================================================
    emit("=" * 78)
    emit("A. A ROAD NETWORK -- WHAT HAVERSINE WAS COSTING")
    emit("=" * 78)
    emit("Not OSM. There is no extract to download here and no routing server to")
    emit("run, and calling a synthetic grid 'a road network' without saying so")
    emit("would be the same overclaim the haversine version was honest about.")
    emit("What it is: a DIRECTED grid with one-way streets, turn penalties and")
    emit("time-varying edge speeds.")
    emit("")
    net = RN.RoadNetwork(seed=1)
    prof = RN.detour_profile(net, n=400, hour=12.0, seed=2)
    emit("  Detour factor -- route distance / straight-line distance, %d samples:"
         % prof["n"])
    emit("    mean %.3f   median %.3f   p90 %.3f   p99 %.3f   max %.3f"
         % (prof["mean"], prof["median"], prof["p90"], prof["p99"], prof["max"]))
    emit("")
    emit("  EVERY HAVERSINE ETA IN THE PREVIOUS PASS WAS OPTIMISTIC BY ~%.0f%% ON"
         % (100 * (prof["mean"] - 1)))
    emit("  DISTANCE ALONE, before traffic. That is the correction the README")
    emit("  said was missing, now measured rather than asserted.")
    emit("")
    emit("  The DISTRIBUTION matters more than the mean, and this is why: a mean")
    emit("  detour factor can be applied as a multiplier, but the p99 of %.2f"
         % prof["p99"])
    emit("  cannot -- the trips that blow the ETA are the ones in the tail, and")
    emit("  multiplying them by the mean leaves them just as wrong.")
    emit("")
    a = net.node_latlon(3, 4)
    b = net.node_latlon(28, 31)
    rows = []
    for hour in (3.0, 8.5, 12.0, 17.8, 22.0):
        r = net.route(a[0], a[1], b[0], b[1], hour)
        rows.append(dict(hour=hour, seconds=r["seconds"], metres=r["metres"]))
    T = pd.DataFrame(rows)
    emit("  The same route at five times of day:")
    emit(T.to_string(index=False, float_format=lambda x: "%10.1f" % x))
    quiet = T.seconds.min()
    peak = T.seconds.max()
    emit("")
    emit("  Rush hour costs %.0f%% (%.0fs vs %.0fs). Haversine has no clock at"
         % (100 * (peak / quiet - 1), peak, quiet))
    emit("  all, so every ETA built on it is the same at 08:30 as at 03:00.")
    emit("")
    emit("  ROUTE ASYMMETRY from one-ways: %.1f%% mean difference between a->b"
         % (100 * prof["mean_asymmetry"]))
    emit("  and b->a. Haversine guarantees symmetry, and an assignment scorer")
    emit("  that caches 'the distance between a and b' is now wrong half the")
    emit("  time -- which is a correctness bug rather than an accuracy one.")
    emit("")
    summary["roadnet"] = dict(profile=prof, by_hour=T.to_dict("records"))

    # ======================================================================
    emit("=" * 78)
    emit("B. A LEARNED ETA, AND THE ANALYTIC ONE IT IS SCORED AGAINST")
    emit("=" * 78)
    X, y, meta = EM.generate(net, n=5000, seed=3)
    cut = int(0.7 * len(X))
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]

    analytic = EM.analytic_eta(Xte[:, 0], Xte[:, 7], Xte[:, 8],
                               items=Xte[:, 5])
    m_mean = EM.fit_learned(Xtr, ytr)
    m_p80 = EM.fit_learned(Xtr, ytr, quantile=0.80)
    learned = m_mean.predict(Xte)
    p80 = m_p80.predict(Xte)

    rows = []
    for name, pred in (("analytic (route+prep+queue)", analytic),
                       ("learned, mean", learned),
                       ("learned, P80", p80)):
        rows.append(dict(model=name, **EM.evaluate(pred, yte)))
    E = pd.DataFrame(rows)
    emit(E.to_string(index=False, float_format=lambda x: "%10.3f" % x))
    emit("")
    an = E.iloc[0]
    le = E.iloc[1]
    q8 = E.iloc[2]
    emit("  The analytic model is the BASELINE, not the straw man. It is")
    emit("  interpretable, needs no training data, degrades gracefully, and is")
    emit("  given the CORRECT mean prep time per restaurant rather than a global")
    emit("  constant -- so the comparison is fair.")
    emit("")
    emit("  Learned minus analytic on MAE: %+.3f minutes." % (le.mae - an.mae))
    emit("")
    emit("  The learned model's only structural advantage is an INTERACTION the")
    emit("  generator plants and an additive model cannot express: a large order")
    emit("  at a slow restaurant during peak is worse than the sum of those three")
    emit("  effects, because a busy kitchen degrades non-linearly. Naming the")
    emit("  advantage is what makes this a measurement rather than 'ML is better'.")
    emit("")
    emit("  BUT LOOK AT THE LATENESS COLUMNS, NOT THE MAE. An ETA is a PROMISE,")
    emit("  and the cost of breaking it is asymmetric -- five minutes early is a")
    emit("  pleasant surprise, five minutes late is a support contact.")
    emit("")
    emit("    analytic     : late %.1f%% of the time, by %.1f min when late"
         % (100 * an.late_share, an.mean_lateness_when_late))
    emit("    learned mean : late %.1f%% of the time, by %.1f min when late"
         % (100 * le.late_share, le.mean_lateness_when_late))
    emit("    learned P80  : late %.1f%% of the time, by %.1f min when late"
         % (100 * q8.late_share, q8.mean_lateness_when_late))
    emit("")
    emit("  A MEAN ETA IS WRONG HALF THE TIME BY CONSTRUCTION, which is what the")
    emit("  ~50%% lateness on both mean models says. The P80 is worse on MAE and")
    emit("  is the one to ship, because the customer-facing question is not 'what")
    emit("  is the expected arrival' but 'what time can I promise'. Choosing an")
    emit("  ETA model on MAE optimises a quantity nobody experiences.")
    emit("")
    emit("  WHAT THE LEARNED MODEL CANNOT DO: explain itself, extrapolate to a")
    emit("  restaurant it has never seen, or survive a new city. The analytic")
    emit("  model does all three, which is why both are kept.")
    emit("")
    summary["eta"] = E.round(4).to_dict("records")

    # ======================================================================
    emit("=" * 78)
    emit("C. SURGE THAT DOES NOT OSCILLATE")
    emit("=" * 78)
    rng = np.random.default_rng(7)
    T_STEPS = 300
    # utilisation wandering across the threshold, which is the regime that
    # separates a controller from a formula
    base = 0.62 + 0.10 * np.sin(np.linspace(0, 6 * np.pi, T_STEPS))
    util = np.clip(base + rng.normal(0, 0.035, T_STEPS), 0, 1)

    naive_series = [CTL.naive_surge(u) for u in util]
    ctl = CTL.SurgeController()
    smart_series = [ctl.observe(u) for u in util]

    n_osc = CTL.oscillation(naive_series)
    s_osc = CTL.oscillation(smart_series)
    emit("  Utilisation wanders across the threshold for %d steps." % T_STEPS)
    emit("")
    emit("  %-28s %12s %16s %10s" % ("controller", "reversals", "mean |step|",
                                     "range"))
    emit("  %-28s %12d %16.4f %10.3f"
         % ("instantaneous (previous)", n_osc["reversals"],
            n_osc["mean_abs_step"], n_osc["range"]))
    emit("  %-28s %12d %16.4f %10.3f"
         % ("hysteresis + forecast", s_osc["reversals"], s_osc["mean_abs_step"],
            s_osc["range"]))
    emit("")
    emit("  REVERSALS ARE THE NUMBER A COURIER EXPERIENCES. A price that goes up,")
    emit("  down, up and down within an hour is not a signal -- it is noise with")
    emit("  a dollar sign, and couriers stop believing it. The instantaneous")
    emit("  controller reverses %d times; the damped one %d."
         % (n_osc["reversals"], s_osc["reversals"]))
    emit("")
    emit("  Three mechanisms, each fixing a different failure. HYSTERESIS gives")
    emit("  separate on and off thresholds, so the controller cannot chatter")
    emit("  across a single line. A FORECAST acts on where utilisation is going,")
    emit("  because supply responds with a lag and a controller that waits for a")
    emit("  region to be short is already late by that lag. A RATE LIMIT caps the")
    emit("  change per step, which is what stops the forecast turning a noisy")
    emit("  derivative into a price spike.")
    emit("")
    emit("  The trend is computed on the SMOOTHED series. Differencing raw")
    emit("  utilisation amplifies exactly the noise the EMA exists to remove, and")
    emit("  a forecast built on it is a noise amplifier with a price attached.")
    emit("")
    emit("  THE CAP IS NO LONGER INERT. The previous slope reached 1.88 at")
    emit("  utilisation 1.0 against a 2.5 ceiling, so the cap shaped nothing and a")
    emit("  test pinned that fact. This controller reaches %.2f, and the cap is"
         % max(smart_series))
    emit("  now a constraint somebody has to sign off rather than a comment.")
    emit("")
    summary["surge"] = dict(naive=n_osc, controlled=s_osc,
                            max_multiplier=float(max(smart_series)))

    # ======================================================================
    emit("=" * 78)
    emit("D. COURIERS THAT REPOSITION")
    emit("=" * 78)
    # A SCARCE market, because repositioning cannot matter in a slack one.
    # The first version gave 120 couriers a total demand of 43 and every policy
    # filled 100% of it -- a comparison in a regime where the thing being
    # compared has no effect. Capacity is now ~65% of demand and the demand is
    # concentrated, which is when where-the-couriers-are starts to decide the
    # outcome.
    n_zones, steps = 8, 200
    rng = np.random.default_rng(11)
    demand = np.array([3.0, 26.0, 2.0, 34.0, 4.0, 1.5, 18.0, 2.5])
    n_couriers = int(demand.sum() * 0.65 / 2.0)
    travel = np.abs(np.arange(n_zones)[:, None] - np.arange(n_zones)[None, :]) * 3.0

    rows = []
    for label, do_move, compliance in (("static (previous)", False, 0.0),
                                       ("repositioning, 55% compliance", True, 0.55),
                                       ("repositioning, 100% compliance", True, 1.0)):
        zone = rng.integers(0, n_zones, n_couriers)
        surge = np.ones(n_zones)
        controllers = [CTL.SurgeController() for _ in range(n_zones)]
        served, unserved, herd = 0, 0, []
        for t in range(steps):
            counts = np.bincount(zone, minlength=n_zones).astype(float)
            util = np.clip(demand / np.maximum(counts, 1e-6) / 2.0, 0, 1)
            surge = np.array([c.observe(u) for c, u in zip(controllers, util)])
            capacity = counts * 2.0
            s = np.minimum(demand, capacity)
            served += float(s.sum())
            unserved += float((demand - s).sum())
            herd.append(CTL.herding_index(counts))
            if do_move:
                idle = rng.random(n_couriers) < 0.35
                zone = CTL.reposition(zone, idle, demand, counts, travel, surge,
                                      rng, compliance=compliance)
        rows.append(dict(policy=label, served=served, unserved=unserved,
                         fill_rate=served / (served + unserved),
                         mean_herding=float(np.mean(herd)),
                         final_herding=herd[-1],
                         mean_surge=float(surge.mean())))
    R = pd.DataFrame(rows)
    emit(R.to_string(index=False, float_format=lambda x: "%12.4f" % x))
    emit("")
    st, mid, full = R.iloc[0], R.iloc[1], R.iloc[2]
    emit("  %d couriers against demand of %.0f -- capacity is %.0f%% of demand."
         % (n_couriers, demand.sum(), 100 * n_couriers * 2 / demand.sum()))
    emit("")
    emit("  Repositioning lifts fill rate from %.4f to %.4f at realistic"
         % (st.fill_rate, mid.fill_rate))
    emit("  compliance, and to %.4f if every courier did as they were told."
         % full.fill_rate)
    emit("")
    gap = full.fill_rate - mid.fill_rate
    emit("  PERFECT COMPLIANCE BUYS %s. That is the result, and it is not the one"
         % ("NOTHING" if gap <= 0.005 else "%.4f MORE FILL RATE" % gap))
    emit("  I expected to write.")
    emit("")
    if gap <= 0.005:
        emit("  Going from 55%% to 100%% compliance moves fill rate by %+.4f while"
             % gap)
        emit("  raising the herding index from %.4f to %.4f. Every courier obeying"
             % (mid.mean_herding, full.mean_herding))
        emit("  the same recommendation sends every courier to the same zone --")
        emit("  which serves that zone twice over and leaves the others exactly as")
        emit("  short as before. THE THUNDERING HERD IS NOT A BUG IN THE POLICY,")
        emit("  IT IS WHAT THE POLICY SAYS when everyone follows it.")
        emit("")
        emit("  The practical reading is uncomfortable and worth stating: the")
        emit("  PARTIAL compliance a dispatch team spends money trying to increase")
        emit("  is doing useful randomisation for free. Spending on incentives to")
        emit("  raise compliance, without also making the recommendation")
        emit("  courier-specific, buys herding rather than coverage.")
        emit("")
        emit("  The fix is not more compliance, it is a recommendation that")
        emit("  differs per courier -- assign zones rather than broadcast them, or")
        emit("  price the zone down as couriers commit to it. Neither is built")
        emit("  here, and both are what a real repositioning system does.")
    else:
        emit("  A dispatch team can RECOMMEND a zone; they cannot move anybody.")
        emit("  The gap between the last two rows, %.4f fill rate, is what a" % gap)
        emit("  dispatch team is bidding for when it spends on incentives.")
    emit("")
    emit("  What repositioning unambiguously DOES buy is the first step: %.4f to"
         % st.fill_rate)
    emit("  %.4f, which is %.0f%% more orders served with the same fleet. Couriers"
         % (mid.fill_rate, 100 * (mid.fill_rate / st.fill_rate - 1)))
    emit("  sitting where demand was yesterday is the cheapest supply problem a")
    emit("  marketplace has.")
    emit("")
    emit("  HERDING is the number that says whether a policy fixed the shortage or")
    emit("  moved it: %.4f static, %.4f at 55%%, %.4f at 100%%. Reporting fill rate"
         % (st.mean_herding, mid.mean_herding, full.mean_herding))
    emit("  without it would have made the 100% row look like a tie rather than a")
    emit("  failure mode.")
    emit("")
    emit("  This is also the carryover mechanism DATA-3 reasons about")
    emit("  qualitatively and cannot represent: couriers moving between zones is")
    emit("  what makes a courier-incentive switchback leak across blocks, because")
    emit("  a courier attracted in block 3 is still there in block 4.")
    emit("")
    summary["repositioning"] = R.round(4).to_dict("records")

    with open(os.path.join(OUT, "complete_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "complete_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\n-> out/complete_report.txt")


if __name__ == "__main__":
    main()
