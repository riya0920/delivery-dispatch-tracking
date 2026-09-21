# Delivery Dispatch & Tracking

## What it is

The software behind a food delivery app's dispatch desk. When an order comes in,
it finds nearby couriers, offers the job, makes sure exactly one courier gets it,
and tells the customer when the food will arrive.

It covers the hard parts of that loop: finding the nearest couriers fast while
thousands of them move, giving an arrival time (ETA) the customer can trust,
setting surge pay that does not jump around, and telling idle couriers where to
wait for the next order.

Everything runs in one Python process on a simulated city. At the end we checked
the simulated road grid against a real city map (Wilmington, Delaware) to see
where it is wrong.

## What we did

1. **Benchmarked four ways to find nearby couriers**: a plain scan, a k-d tree,
   a square grid, and a hexagon grid (standing in for PostGIS, Redis and H3).
2. **Made assignment safe under concurrency**: one order goes to one courier, and
   one courier never gets two orders at once.
3. **Built an ETA with an honest range**, fitted on held-out data, plus
   "where is my order?" drills for when data goes stale.
4. **Added courier agents** that accept, decline or go offline, with a re-offer
   cascade.
5. **Built a road network**: a directed grid with one-way streets, turn penalties
   and traffic that changes by time of day.
6. **Trained a learned ETA** and scored it against a fair formula-based one.
7. **Rebuilt surge pricing** so it stops flipping up and down.
8. **Added courier repositioning**, then per-courier recommendations, measured
   against the best placement any policy could reach.
9. **Checked the grid against real OpenStreetMap data** for Wilmington.

Work was done in several passes. Later passes re-tested earlier claims, and one
overturned an earlier conclusion (see Results).

## Results

**Finding nearby couriers (10,000 couriers)**

- At 5 and 50 location updates per query, the square grid wins.
- At 200 updates per query, the **plain scan wins**. It is one numpy call, while
  each index pays to update itself.
- This does not hold at every size: the scan's cost grows faster than the grid's
  as the fleet grows (1.6x to 6.7x in the fleet-size sweep).

**Road network (simulated grid)**

| detour factor (route length / straight line), 400 samples | |
|---|---|
| mean | 1.329 |
| p99 | 1.718 |

- So ETAs based on straight-line distance were about **33% too short** before
  traffic.
- The same route takes **552 s at 03:00 and 974 s at 08:30** (rush hour costs 76%).
- 7.7% of routes differ by direction (A to B vs B to A) because of one-way streets.

**ETA models**

| model | MAE (min) | late share | avg lateness when late |
|---|---|---|---|
| formula (route + prep + items + queue) | 3.202 | 49.0% | 4.60 min |
| learned, mean | **2.445** | 47.8% | 2.79 min |
| learned, 80th percentile | 3.210 | **24.9%** | **2.66 min** |

The 80th-percentile model has worse average error but is late half as often.
That is the one to show customers.

**Surge pricing**

| controller | price reversals |
|---|---|
| old (reacts instantly) | 134 |
| new (thresholds + forecast + rate limit) | **65** |

**Courier repositioning (29 couriers, capacity 64% of demand)**

| policy | fill rate | herding |
|---|---|---|
| couriers stay put | 0.4286 | 0.0216 |
| reposition, 55% follow advice | **0.6337** | 0.1975 |
| reposition, 100% follow advice | 0.6333 | 0.2196 |

- Repositioning serves **48% more orders** with the same fleet.
- "Herding" measures couriers piling into the same zone. Full compliance added
  herding and no fill rate.

**Per-courier recommendations vs the best possible**

| capacity vs demand | best possible | send everyone | targeted | gap closed |
|---|---|---|---|---|
| 65% | 0.6374 | 0.6359 | 0.6358 | n/a |
| 120% | 1.0000 | 0.9469 | **0.9984** | **0.97** |

In the scarce market, sending everyone was already at the ceiling, so the
herding we saw there was the best answer, not a flaw. Once supply covers demand,
targeting recovers 97% of the lost fill with 33% less herding. **This corrected
our own earlier conclusion**, which had called the broadcast "the defect" from one
scarce-market run.

**Real city check (Wilmington, 3,136 roads, 50.6% one-way)**

| straight-line distance | Wilmington median detour | grid median detour |
|---|---|---|
| 500 m and up | **1.75** | 1.30 |
| 4,000 m and up | 1.34 | 1.39 |

- The real detour falls 24% as trips get longer; the grid's is flat. Delivery
  trips are short, which is where the grid is most wrong.
- Routes that differ by direction: **85.0%** in Wilmington vs 7.7% on the grid.

**Other results**

- Surge is the only lever that raised acceptance (0.748 to 0.976 in a tight
  market), at a cost of $6.45 to $10.32 per job.
- Batching orders gains +0.57% at 0.45 utilisation and **+14.3% at 0.95**.

**Bugs found by testing (and fixed)**

- The invariant check found **nine cases of one courier assigned to two orders**.
  There was an order-level guard but no courier-level one.
- The hand-picked ETA range gave **49% coverage** on an 80% interval. Fitted on
  held-out data it gives 0.894.
- One-way streets first blocked travel across a street instead of along it.
  Routing silently fell back to straight lines (detour exactly 1.00). The
  fallback now counts itself and a test checks it never fires.
- The first formula ETA left out the per-item time and was 6.8 minutes biased,
  which would have handed the learned model an unfair win.
- 11.7% of random pairs on the real map were unreachable, because the map's box
  cut roads at its edge. Routing now uses only the largest strongly-connected
  part (93% of nodes).

## Key decisions and why

**Score the learned ETA against a fair formula baseline.**
The formula gets every additive term the data generator uses. The learned model's
only edge is one named interaction (big order, slow kitchen, peak hour), so the
gain is a measurement, not "ML is better".

**Pick the ETA by how often it is late, not by average error.**
An ETA is a promise. Five minutes late costs a support contact, five minutes
early costs nothing. A mean ETA is late about half the time by design.

**Give surge separate on and off thresholds.**
One threshold lets the price chatter across a single line. The code refuses to
build a controller without the gap. The trend is computed on smoothed data so the
forecast does not amplify noise.

**Always report herding next to fill rate.**
Without it, 100% compliance looks like a tie. With it, you see couriers just
moving the shortage around.

**Measure the ceiling before judging a policy.**
Without the "best possible" number, "the smart policy did not help" and "there
was nothing left to win" look the same, and they call for opposite decisions.

**A declined offer does not use up the zone's slot.**
Otherwise low compliance would look worse for a modelling reason, and the result
would be built into the setup.

**Use strongly-connected, not weakly-connected, roads.**
With one-way streets, "connected if you ignore direction" is not the question a
courier asks.

**Get the real map from Overpass, not pyosmium.**
pyosmium installs but is blocked by this machine's security policy. Overpass JSON
needs no compiled parser, so we used a city bounding box instead of a full map.

## Limits

- **The real map is measured, not used.** ETA, assignment and repositioning still
  run on the synthetic grid.
- The real map has **no traffic and no turn restrictions**, so its detour factors
  are a lower bound. Rush-hour effects are our own assumption.
- One city only. What transfers is the shape of the error, not its size.
- The geo benchmark uses in-process stand-ins, not real PostGIS, Redis or H3. No
  network hop, so do not quote the times as database numbers.
- Zone demand is given as known, not forecast. This flatters every policy, and
  targeting most.
- The repositioning advice is not priced (surge does not react to it).
- Greedy assignment, not the optimal one.
- No WebSocket, map, dashboard or alerting. Tracking is a state machine and a
  message.
- Courier agents only drive the cascade benchmark.
- Single process, single metro, no persistence. The queue drill is a deque, not a
  real message broker.

## How to run

```bash
pip install -r requirements.txt
python run_dispatch.py       # geo benchmark, assignment, ETA, drills, cascade (~3 min)
python run_complete.py       # road network, learned ETA, surge, repositioning (~4 min)
python run_osm.py            # real-map check (fetches Wilmington from Overpass)
python -m pytest tests -q    # 83 tests
```

Full reports: [dispatch_report](out/dispatch_report.txt) ·
[complete_report](out/complete_report.txt) · [osm_report](out/osm_report.txt)
(metrics as JSON in `out/`).

## Layout

```
src/geo.py         four nearest-courier indexes, benchmarked
src/dispatch.py    offers, exactly-one assignment, batching
src/eta.py         ETA with fitted range, "where is my order?" tracking
src/couriers.py    courier agents and the re-offer cascade
src/roadnet.py     directed grid: one-ways, turn penalties, traffic
src/eta_model.py   learned ETA and the formula baseline
src/control.py     surge controller and courier repositioning
src/osmnet.py      real OpenStreetMap network for Wilmington
```
