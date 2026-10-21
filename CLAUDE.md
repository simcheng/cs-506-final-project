# CLAUDE.md

CS 506 final project: predict which B line trains run express between Kenmore and Harvard Ave
(the BU stops), using MBTA headway and travel-time data, Jan–Sep 2026. See README.md for setup.

## Pipeline

Run from `src/` with `../.venv/bin/python`:

1. `extract.py`: raw CSVs in `datasets/` → `data_processed/data_cleaned/headways_bline.parquet`,
   `traveltimes_bline.parquet` (Green-B only, all B line stops).
2. `process.py`: labels express runs → `express_trains.parquet`, `GeneralStats.json`,
   `data_quality.json`, plus `data_cleaned/stop_events|trips|current_stops.parquet`.
3. `visualize.py`: `figures/fig1–fig7` + `dwellTime.csv` + `figure_data/figN_*.csv`.

All constants (stop ids, 25 s threshold, 6:00–22:00 window) live in `src/common.py`.

## Definitions (agreed with the user; do not change without asking)

- **Corridor**: Kenmore, Blandford St, BU East, BU Central, Amory St, Babcock St, Packard's Corner,
  Harvard Ave, both directions. direction_id 0 = West, 1 = East.
- **Window**: keep trips whose whole corridor passage is between 6:00 and 22:00. Fig 5 is the
  only figure that shows all hours (to justify the cut).
- **Rolled through** a stop: dwell = departure − arrival < 25 s.
- **Express run**: a train stops at a corridor stop (src stop), then rolls through the next stop.
  **No minimum** number of skipped stops. The run continues until the train stops again (dst stop),
  which can be beyond Harvard Ave.
- **Current stop** = the src stop. Every corridor stop where a train stopped is a candidate.
- **Express %** at a stop = express departures from that stop / departures from that stop.
- **Train in front / behind**: the previous / next B departure at the same stop, same day and
  direction, ordered by time. Never order by `trip_id`, which has no pattern.
  - minutes in front = (our departure − its departure from the current stop) / 60
  - minutes behind = (its arrival at the current stop − our departure) / 60
  - stops = where that train last departed on the B line at the moment we depart (between Kenmore
    and Blandford counts as "at Kenmore"). A train that hasn't left its terminal counts from
    Government Center (WB) / Boston College (EB).

## Data facts

- Express runs do NOT leave missing stop rows. The feed logs a departure at every stop passed.
  Detect skips by dwell time instead.
- Timestamps end in `Z` but are local Boston time. `*_sec` = seconds after midnight of the
  service day, and can exceed 86400 after midnight.
- `headway_trunk_seconds` = gap to any Green Line branch; `headway_branch_seconds` = B only.
- Arrivals come from the LR travel-time files (`to_stop_arrival_sec`); departures from headways.
- About 19 days had the corridor replaced by shuttles; those (date, direction) pairs are dropped.

## Working with this user

- Do only what is asked. Don't add fields, files, figures, panels or stats that weren't requested.
- No confidence intervals or error bars on figures.
- Every figure gets a CSV of its exact plotted numbers. Every data table must be produced by
  the scripts, never written by hand.
- Data the user asked for goes in `data_processed/`; intermediate tables go in
  `data_processed/data_cleaned/` (gitignored).
- When a definition (express, current stop, a denominator) is ambiguous, ask in plain words with
  a concrete example before building on it. Don't invent thresholds.
- Figure titles must stay true to the current data. Re-check them after any definition change.
- The user writes the check-in write-up themselves.
