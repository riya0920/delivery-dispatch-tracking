# SE-3 — Delivery Dispatch & Tracking

**Complete against the spec.** A four-way geo-index benchmark with a scale sweep,
exactly-one-assignment under concurrency, calibrated ETA intervals, WISMO
degradation drills, courier agents with a re-offer cascade, **a directed road
network with one-ways and traffic**, **a learned ETA scored against a fair
analytic baseline**, **surge that does not oscillate**, and **couriers that
reposition**.

The last of those produced the most uncomfortable result in the project.

```bash
python run_dispatch.py       # ~3min  geo benchmark, assignment, ETA, drills, cascade
python run_complete.py       # ~4min  road network, learned ETA, surge control, repositioning
python -m pytest tests -q    # 61 tests
```

## A road network — what haversine was costing

This is **not OSM**. There is no extract to download here and no routing server to
run, and calling a synthetic grid "a road network" without saying so would be the
same overclaim the haversine version was honest about. What it *is*: a **directed**
grid with one-way streets, turn penalties and time-varying edge speeds.

| detour factor (route ÷ straight line), 400 samples | |
|---|---|
| mean | **1.329** |
| median | 1.351 |
| p90 | 1.477 |
| p99 | **1.718** |
| max | 1.900 |

**Every haversine ETA in the previous pass was optimistic by ~33% on distance
alone**, before traffic. That is the correction the README said was missing, now
measured rather than asserted.

The **distribution** matters more than the mean, and this is why: a mean detour
factor can be applied as a multiplier by anyone who cannot run a router, but the
p99 of 1.72 cannot — the trips that blow the ETA are the ones in the tail, and
multiplying them by the mean leaves them just as wrong.

The same route at five times of day: **552 s at 03:00, 974 s at 08:30** — rush
hour costs **76%**. Haversine has no clock at all, so every ETA built on it is the
same at 08:30 as at 03:00.

**Route asymmetry from one-ways: 7.7%.** Haversine guarantees `d(a,b) = d(b,a)`,
and an assignment scorer that caches "the distance between a and b" is now wrong
half the time — which is a **correctness** bug rather than an accuracy one.

> **A routing bug that returns haversine looks exactly like a working router.**
> The first version had one-ways restricting travel *across* a street rather than
> *along* it, which stranded most of the grid; A* found nothing and every route
> fell silently through to the haversine fallback. The symptom was detour factors
> of exactly 1.00 and identical times at 03:00 and 08:30. The fallback now counts
> itself and a test asserts it never fires.

## A learned ETA, and the analytic one it is scored against

| model | MAE | bias | late share | mean lateness when late |
|---|---|---|---|---|
| analytic (route + prep + items + queue) | 3.202 | −1.306 | 49.0% | 4.60 min |
| learned, mean | **2.445** | −0.226 | 47.8% | 2.79 min |
| learned, **P80** | 3.210 | +1.889 | **24.9%** | **2.66 min** |

The analytic model is the **baseline, not the straw man**: it is given every
*additive* term the generator uses, including the per-item time and the mean of
the prep noise. The learned model's only structural advantage is an **interaction**
the generator plants and an additive model cannot express — a large order at a
slow restaurant during peak is worse than the sum of those three effects, because
a busy kitchen degrades non-linearly. Naming the advantage is what makes this a
measurement rather than "ML is better".

> The first version of the baseline omitted the per-item term and was 6.8 minutes
> biased. The learned model would have won on a term anybody could have
> configured, which is not a finding.

**But look at the lateness columns, not the MAE.** An ETA is a *promise*, and the
cost of breaking it is asymmetric: five minutes early is a pleasant surprise, five
minutes late is a support contact. **A mean ETA is late half the time by
construction** — which is what the ~48% says. The P80 is *worse* on MAE and is the
one to ship, because the customer-facing question is not "what is the expected
arrival" but "what time can I promise". **Choosing an ETA model on MAE optimises a
quantity nobody experiences.**

What the learned model cannot do: explain itself, extrapolate to a restaurant it
has never seen, or survive a new city. The analytic model does all three, which is
why both are kept.

## Surge that does not oscillate

| controller | reversals | mean \|step\| | range |
|---|---|---|---|
| instantaneous (previous) | **134** | 0.0467 | 0.380 |
| hysteresis + forecast + rate limit | **65** | 0.0254 | 0.318 |

**Reversals are the number a courier experiences.** A price that goes up, down, up
and down within an hour is not a signal — it is noise with a dollar sign, and
couriers stop believing it.

Three mechanisms, each fixing a different failure. **Hysteresis** gives separate
on and off thresholds so the controller cannot chatter across a single line — and
the constructor *raises* if the gap is missing, because a controller that claims
hysteresis and has one threshold is worse than one that never claimed it. A
**forecast** acts on where utilisation is going, because supply responds with a
lag and a controller that waits for a region to be short is already late by that
lag. A **rate limit** caps the change per step, which is what stops the forecast
turning a noisy derivative into a price spike.

The trend is computed on the **smoothed** series: differencing raw utilisation
amplifies exactly the noise the EMA exists to remove, and a forecast built on it is
a noise amplifier with a price attached.

**The cap is no longer inert.** The previous slope reached 1.88 at utilisation 1.0
against a 2.5 ceiling, so the cap shaped nothing and a test pinned that fact. This
controller reaches its cap, and the cap is now a constraint somebody has to sign
off rather than a comment.

## Couriers that reposition — and the result I did not expect

29 couriers against demand of 91: capacity is **64%** of demand.

| policy | fill rate | mean herding |
|---|---|---|
| static (previous) | 0.4286 | 0.0216 |
| repositioning, **55% compliance** | **0.6337** | 0.1975 |
| repositioning, **100% compliance** | 0.6333 | **0.2196** |

**Perfect compliance buys nothing.** Going from 55% to 100% moves fill rate by
**−0.0003** while raising the herding index from 0.198 to 0.220. Every courier
obeying the same recommendation sends every courier to the same zone — which
serves that zone twice over and leaves the others exactly as short as before.
**The thundering herd is not a bug in the policy; it is what the policy says when
everyone follows it.**

The practical reading is uncomfortable and worth stating: **the partial compliance
a dispatch team spends money trying to increase is doing useful randomisation for
free.** Spending on incentives to raise compliance, without also making the
recommendation courier-specific, buys herding rather than coverage. The fix is not
more compliance — it is a recommendation that differs per courier: assign zones
rather than broadcast them, or price the zone down as couriers commit to it.
Neither is built here.

What repositioning unambiguously *does* buy is the first step: **0.429 → 0.634, or
48% more orders served with the same fleet.** Couriers sitting where demand was
yesterday is the cheapest supply problem a marketplace has.

**Herding is the number that says whether a policy fixed the shortage or moved
it.** Reporting fill rate without it would have made the 100% row look like a tie
rather than a failure mode.

This is also the carryover mechanism DATA-3 reasons about qualitatively and cannot
represent: couriers moving between zones is what makes a courier-incentive
switchback leak across blocks, because a courier attracted in block 3 is still
there in block 4.

## Results from the earlier passes that still stand

- **The naive linear scan beat every index** on blended cost at a high write:read
  ratio. A vectorised scan is one numpy call; index maintenance is per-update
  Python. The fleet-size sweep shows where that stops being true (1.6× → 6.7×).
- **Surge is the only lever that moves acceptance** (0.748 → 0.976 in a tight
  market) and it costs $6.45 → $10.32 per job. Routing cannot conjure couriers.
- **Batching's value is a function of scarcity**: +0.57% at 0.45 utilisation,
  **+14.3% at 0.95**. A right number from a single operating point invites exactly
  the wrong conclusion.
- **`check_invariants()` found nine cases of one courier assigned to two orders.**
  The order-level compare-and-set was correct and there was no courier-level guard
  at all.
- **The ETA's hand-picked sigma gave 49% coverage on an 80% interval.** Now fitted
  from residuals on a held-out half (0.894).

## What is deliberately not here

- **Still not OSM.** A synthetic grid with one-ways and traffic is a better model
  than haversine and is not a map. Real street networks have rivers, bridges,
  dead ends and turn restrictions that a grid cannot express, and the detour
  factor on a real metro is a different number.
- **No WebSocket, no map, no reconnect/catch-up.** The tracking surface is a state
  machine and a rendered message, not a client.
- **Courier agents still drive only the cascade benchmark**; the geo benchmark and
  the ETA evaluation use the static availability mask.
- **The repositioning recommendation is broadcast, not per-courier** — which the
  compliance result above says is the actual defect.
- **No dispatch dashboard.** The runbook references metrics the report computes;
  nothing renders or alerts on them.
- **Single process, single metro, no persistence.** The queue-and-drain drill is
  arithmetic on a deque, not a broker.
