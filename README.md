# SE-3 — Real-Time Delivery Tracking & Dispatch

**Roughly 50% of the spec.** A defended geo-index choice, the assignment
invariant under concurrency, a calibrated ETA and the degraded-mode behaviour -
plus the three things the first pass named as missing: **courier agents that
decline** (its own "biggest fidelity gap"), the batching sweep it called its
missing experiment, and an on-call runbook. No map, no WebSocket, no road
network; what remains is named at the bottom.

```bash
python run_dispatch.py      # ~2min
python -m pytest tests -q   # 38 tests
```

## Geo index, benchmarked and defended

10,000 couriers, writes and reads **interleaved** — benchmarking reads alone
would flatter the tree index enormously, and that's the mistake that gets made:
couriers ping every 1–5 seconds, so writes dominate.

| updates/query | index | update µs | query ms | recall vs exact |
|---|---|---|---|---|
| 50 | linear_scan | 4.2 | 1.725 | 1.000 |
| 50 | kdtree | 11.4 | 0.684 | **0.984** |
| 50 | grid_hash | 13.7 | **0.538** | 1.000 |
| 50 | hex_h3 | 44.5 | 1.229 | 1.000 |

`kdtree` has the fastest queries and the worst updates — it rebuilds — and its
recall is **below 1.0** because between rebuilds it answers from stale positions.
That's amortisation showing up as a *correctness* cost, which is the honest place
to put it.

**The result I didn't expect:** at 200 updates/query the **naive linear scan wins
on blended cost**. A vectorised haversine over 10,000 points is one numpy call,
while every index pays per-update Python overhead. At a high enough write:read
ratio, maintaining an index costs more than not having one.

That's true and it doesn't generalise, which is why the sweep exists:

| fleet | grid_hash query ms | linear_scan query ms | **ratio** |
|---|---|---|---|
| 2,000 | 0.370 | 0.600 | 1.6× |
| 10,000 | 0.804 | 1.922 | 2.4× |
| 50,000 | 1.833 | 10.10 | 5.5× |
| 200,000 | 10.49 | 70.50 | **6.7×** |

The scan is linear in fleet size; the grid touches a fixed ring of cells whose
occupancy grows only with *density*. Benchmarking at the size you have and
deploying at the size you'll have is how teams end up rewriting dispatch.

**Substitutions:** PostGIS, Redis and `h3` aren't installed, so these are the
*algorithmic shapes* in-process — no network hop, no serialisation, no
concurrency control, no durability. A real PostGIS query pays milliseconds of
round-trip that dwarf everything here. What transfers is how read and write costs
scale and what staleness each structure forces. At 500K couriers across 50 metros
this becomes a sharding problem, and the **hex index is the one that survives**
because the cell id *is* the shard key and neighbour lookup is arithmetic rather
than a range scan.

Hexagons also fix a real flaw: a square grid's corners are 1.41× further than its
edges, so which couriers get considered depends on where in the cell the
restaurant sits.

## Exactly-one-assignment

500 orders, 3,000 offers, 24 threads accepting concurrently:

| | |
|---|---|
| offers accepted | **500** |
| orders assigned | 500 / 500 |
| offers that raced and lost cleanly | 2,490 |
| **invariant violations** | **none** |

Enforced by a single guarded compare-and-set under one lock — not check-then-write,
which is the race it exists to prevent. Same defect as decrement-and-hope
inventory in SE-1, different costume.

**A bug my own invariant checker caught:** the first version let one courier win
*two* orders concurrently — the order-level CAS was correct and there was no
courier-level guard at all. `check_invariants()` reported nine violations, which
is exactly what invariant checkers are for. Both the guard and a regression test
now exist.

## Batched vs greedy — and an honest small number

| window | orders/batch | greedy km | batched km | improvement | first-offer delay |
|---|---|---|---|---|---|
| 0s | 1 | 0.1993 | 0.1993 | 0.00% | 0s |
| 30s | 36 | 0.2035 | 0.2021 | 0.67% | 15s |
| 60s | 72 | 0.2070 | 0.2042 | **1.34%** | 30s |

**1.34% is small, and reporting it as small is the honest read.** With 3,490
available couriers in this metro the nearest one is already ~0.2 km away, so
there is very little room for a smarter assignment. Batching pays when supply is
*tight* and couriers are far apart — exactly the regime this fleet is not in. A
benchmark run only in the easy regime would conclude batching is worthless; the
correct conclusion is that its value is a function of utilisation — which the
second pass sweeps and confirms below.

On "product says couriers churn when offers are slow": that cost isn't in this
table, which measures assignment quality and not retention. The exchange rate
between them is a business input. What the curves *do* support: pick the shortest
window whose gain is still material, and make it **adaptive** — short when supply
is plentiful, longer only when utilisation makes the assignment matter.

## ETA — a failing number, then fixed

| | as shipped | calibrated |
|---|---|---|
| base sigma | 4.00 min (hand-picked) | **12.41 min** (fitted) |
| 80% interval coverage | **0.494** | **0.894** |

MAE 10.06 min, bias +3.00, p90 absolute error 21.4 min.

An interval covering 49% of outcomes while *claiming* 80% is not conservative,
it's **wrong**, and it fails in the direction that costs support tickets — the
customer is told 25–31 minutes and the order arrives at 44. The base sigma was
picked by hand, which is exactly how this happens. Sigma is now fitted on one
half of the evaluation set and scored on the other; quoting coverage on the data
you tuned against is the interval equivalent of reporting training accuracy.

**Staleness policy:** rows for predictions made from a fix older than 45s
*over-cover* (0.99–1.00) rather than collapsing, because the interval widens with
the age of the fix. Over-covering is the correct direction to be wrong when you
don't know where the courier is.

## Degradation drills

**Matcher outage** — killed 120s at 1.2 orders/s: 144 orders queued, **0 lost**,
backlog cleared in **52s**, steady state 1 order. Dispatch is allowed to be down;
it is not allowed to lose an order.

*(An earlier version looped until the queue emptied and reported a 10,001-second
drain — which is what an unbounded steady state looks like when you mistake it
for a backlog. The stopping condition is now "backlog cleared", not "queue
empty".)*

**GPS blackout** — 20% of couriers: mean 80% interval widens 34.5 → 47.0 min
(+36%), tracking states split live / delayed_signal / signal_lost.

**The WISMO surface**, second by second:

```
t+ 10s  state=live            dot=True   Arriving in 4-32 min
t+ 60s  state=delayed_signal  dot=True   Last seen 1 min ago - arriving in ...
t+150s  state=delayed_signal  dot=True   Last seen 2 min ago - ...
t+400s  state=signal_lost     dot=False  We've lost signal from your courier...
```

At 180s **the dot comes off the map**. A frozen dot is worse than no dot, because
the customer believes it until they stop believing anything you tell them — a
confidently-wrong tracking page generates the exact ticket it was built to
prevent. And `smooth_eta` damps *decreases* while letting *increases* through
almost undamped: a customer facing 20 extra minutes needs to know now, while they
can still act. Good news can wait; bad news cannot.

## Second pass: three gaps the first pass named

### Couriers that decline - the fidelity gap, closed

The first pass called couriers "a capacity pool with a service-time distribution"
and named that as its biggest fidelity gap. It was: the offer flow was exercised
by test threads that always accepted, so **offer-accept rate, time-to-assign and
re-offer depth** - the three numbers a dispatch team watches - could not be
measured at all.

Couriers are now agents. They decline offers that are too far for too little, get
pickier through the shift, and go offline. A re-offer cascade walks the ranked
candidate list until someone accepts, and **each decline costs its TTL in wall
clock** - so cascade depth *is* time-to-assign.

| surge | market | accept rate | mean depth | mean wait | mean payout |
|---|---|---|---|---|---|
| off | slack | 0.780 | 1.28 | 7.1s | — |
| off | normal | 0.765 | 1.31 | 7.7s | — |
| off | **tight** | **0.748** | 1.34 | **8.4s** | $6.45 |
| on | **tight** | **0.976** | 1.02 | **0.6s** | **$10.32** |

**Surge is the only lever that moves acceptance.** Routing cannot conjure
couriers - a better index finds the nearest available one faster and it is still
the same courier saying no. Surge buys acceptance and **the price is on the same
table**: a dispatch team reporting accept rate without payout is reporting half a
metric.

*A calibration bug worth keeping:* acceptance first keyed on **$/km**, and every
courier accepted everything - because in a dense metro the marginal offer is 200
metres away and $6/0.2km is an absurd rate. The metric said the job was wonderful
when it was eleven minutes of waiting for six dollars. It now keys on **$/hour**
with a fixed per-job overhead, which is why couriers dislike tiny orders and why
per-drop pay without a distance term selects exactly wrong.

### Batching swept over utilisation - the missing experiment

Section 3 measured batching at one supply level, found 1.3%, and said plainly the
number was small *because the fleet was slack* - and that its value is a function
of utilisation, which the run did not sweep. It called that the missing
experiment. At a 60-second window:

| market | utilisation | mean improvement | worst-case improvement |
|---|---|---|---|
| slack | 0.45 | +0.57% | +0.82% |
| normal | 0.70 | +1.47% | +2.65% |
| tight | 0.88 | +4.57% | +5.86% |
| **very tight** | **0.95** | **+14.28%** | **+24.58%** |

**Batching's value is a function of scarcity**, now measured rather than
asserted. The first pass's 1.3% was measured in the slack regime and correctly
reported as small - and reading it as "batching isn't worth it" would have been
the wrong conclusion from a right number, which is exactly what a
single-operating-point benchmark invites.

The worst-case column moves more than the mean throughout. **Batching's real
product is tail control**: greedy's failure mode is that an early order takes the
courier a later, closer order needed, and that shows up as one very long pickup
rather than a slightly worse average.

### On-call runbook

Prose, deliberately - a runbook is read at 3am by someone who did not write the
system. Three sections matching the three drills: **matching service down**
(confirm orders are queuing not erroring; the queue is the design), **GPS
blackout** (check the ingest partition before blaming 20% of phones; do not
"fix" the map by rendering last-known as live), and **accept-rate collapse**
(this is a supply problem and routing cannot fix it; the lever is surge, which is
a business decision with a number attached, not an incident action).

## The other ~50% - what is still NOT here

- **No road network.** No OSM extract, no OSRM — distances are haversine, so
  every ETA is optimistic by whatever the local street grid costs.
- **No WebSocket, no map, no reconnect/catch-up.** The tracking surface is a
  state machine and a rendered message, not a client.
- **Courier agents are not wired into the main simulation loop** - they drive
  the cascade benchmark in section 6, but sections 1-5 still use the static
  availability mask, so the geo benchmark and the ETA evaluation do not see
  declines or offline couriers.
- **No dispatch dashboard.** The runbook references metrics the report computes;
  nothing renders or alerts on them.
- **No courier repositioning.** Couriers do not move toward demand between jobs,
  which is the behaviour that makes courier-incentive experiments hard (and the
  carryover mechanism DATA-3 reasons about qualitatively).
- **Surge is a function of instantaneous utilisation only** - no forecast, no
  hysteresis, so it would oscillate in a real control loop.
- **The surge cap is inert** at the configured slope (utilisation 1.0 gives
  1.88 against a 2.5 ceiling); a test pins that so a future steeper slope makes
  it visible rather than silent.
- **The ETA model is route-time + prep + queue.** No traffic, no historical
  per-restaurant prep distributions, no learned model.
- **Single process, single metro, no persistence.** The queue-and-drain drill is
  arithmetic on a deque, not a broker.
