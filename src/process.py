"""Step 2: find express runs and measure the trains around them at the current stop.

Express run: a train stops at a corridor stop (src stop), then rolls through the next stop
(dwell < PASS_THRESHOLD_SEC) without opening its doors. Any number of skipped stops counts. The
run ends at the first stop where it stops again (dst stop), which can be beyond Harvard Ave.
We cannot use "missing stop rows" because the MBTA feed records a departure event at every stop a
train passes, stopped or not.

Current stop: every corridor stop where a train stopped is a possible src stop. The train counts
as express from that stop if it rolls through the next stop. Express % at a stop =
express runs starting there / trains that stopped and departed there.

Outputs:
  data_processed/express_trains.parquet  one row per express run (= one express departure), 6:00-22:00
  data_processed/GeneralStats.json       trip, departure and express departure counts, 6:00-22:00
  data_processed/data_cleaned/
    stop_events.parquet     one row per trip x B line stop: arrival, departure, dwell, passed flag
    trips.parquet           one row per trip with all 8 corridor stops, all hours (fig5, fig6, fig7)
    current_stops.parquet   one row per trip x corridor stop it stopped at, 6:00-22:00 (fig2-fig4)
  data_quality.json         cleaning counts quoted in docs/october_checkin.md
"""
import json

import duckdb
import numpy as np
import pandas as pd

from common import (BLINE, CORRIDOR, CORRIDOR_NAMES, DAY_END_HOUR, DAY_START_HOUR, PASS_THRESHOLD_SEC,
                    CLEANED, PROCESSED)

MIN_DEPARTURES_PER_STOP_DAY = 20  # below this, the corridor was bustituted / shut that day


def load_stop_events(con):
    """B line departures joined with arrivals from the travel-time feed."""
    hw = con.sql(f"SELECT * FROM '{CLEANED}/headways_bline.parquet'").df()
    # Keep each direction's own platforms; a few raw rows carry the opposite platform's stop id.
    hw = hw[[s in BLINE[d] for d, s in zip(hw.direction_id, hw.stop_id)]]
    # Arrival at a stop is the same no matter which origin row it comes from (verified: 0 conflicts).
    arr = con.sql(f"""
        SELECT service_date, trip_id, to_stop_id AS stop_id, min(to_arr_sec) AS arr_sec
        FROM '{CLEANED}/traveltimes_bline.parquet' GROUP BY ALL
    """).df()
    ev = hw.merge(arr, on=["service_date", "trip_id", "stop_id"], how="left")
    ev["dwell"] = ev.dep_sec - ev.arr_sec
    ev["passed"] = ev.dwell < PASS_THRESHOLD_SEC
    return ev


def full_service_days(ev):
    """(service_date, direction) pairs where every corridor stop saw regular service."""
    counts = ev.groupby(["service_date", "direction_id", "stop_id"]).size().unstack("stop_id")
    keep = []
    for d, stops in CORRIDOR.items():
        c = counts.xs(d, level="direction_id").reindex(columns=stops)
        keep += [(day, d) for day in c.index[(c.fillna(0) >= MIN_DEPARTURES_PER_STOP_DAY).all(axis=1)]]
    return pd.MultiIndex.from_tuples(keep, names=["service_date", "direction_id"])


def presence_pattern(stop_ids, d):
    present = set(stop_ids)
    return "".join("X" if s in present else "." for s in CORRIDOR[d])


def stop_name(d, stop_id):
    if stop_id in CORRIDOR[d]:
        return CORRIDOR_NAMES[d][CORRIDOR[d].index(stop_id)]
    return STOP_NAMES.get(stop_id)


def label_trips(ev):
    """One row per trip that has a departure and a dwell at all 8 corridor stops, plus its express runs."""
    trips, runs = [], []
    for (d, day, trip), g in ev.groupby(["direction_id", "service_date", "trip_id"], sort=False):
        dwell = dict(zip(g.stop_id, g.dwell))
        dep = dict(zip(g.stop_id, g.dep_sec))
        if not all(s in dwell and pd.notna(dwell[s]) for s in CORRIDOR[d]):
            continue  # cannot tell stopped from rolled through at every corridor stop
        passed = [dwell[s] < PASS_THRESHOLD_SEC for s in CORRIDOR[d]]
        n_runs = 0
        for k in range(len(CORRIDOR[d]) - 1):
            if passed[k] or not passed[k + 1]:
                continue
            # Stopped at k, rolled through k + 1: walk down the B line until it stops again.
            j = BLINE[d].index(CORRIDOR[d][k]) + 1
            skipped = 0
            while j < len(BLINE[d]) and pd.notna(dwell.get(BLINE[d][j])) and dwell[BLINE[d][j]] < PASS_THRESHOLD_SEC:
                skipped += 1
                j += 1
            ends = j < len(BLINE[d]) and pd.notna(dwell.get(BLINE[d][j]))
            runs.append({"direction_id": d, "service_date": day, "trip_id": trip, "corridor_idx": k,
                         "dst_stop": stop_name(d, BLINE[d][j]) if ends else None, "skipped_number": skipped})
            n_runs += 1
        trips.append({
            "direction_id": d, "service_date": day, "trip_id": trip,
            "first_dep_sec": dep[CORRIDOR[d][0]], "last_dep_sec": dep[CORRIDOR[d][-1]],
            "n_express_runs": n_runs, "express": n_runs > 0,
            **{f"passed_{i}": p for i, p in enumerate(passed)},
        })
    return pd.DataFrame(trips), pd.DataFrame(runs)


def current_stop_candidates(trips, ev):
    """One row per trip x corridor stop where it stopped (not the last stop): a possible src stop."""
    dep = ev.set_index(["service_date", "direction_id", "trip_id", "stop_id"]).dep_sec
    rows = []
    for t in trips.itertuples():
        passed = [getattr(t, f"passed_{i}") for i in range(len(CORRIDOR[t.direction_id]))]
        for k in range(len(passed) - 1):
            if not passed[k]:
                rows.append((t.service_date, t.direction_id, t.trip_id, k, CORRIDOR[t.direction_id][k],
                             bool(passed[k + 1])))
    c = pd.DataFrame(rows, columns=["service_date", "direction_id", "trip_id", "corridor_idx", "stop_id",
                                    "goes_express"])
    c["dep_sec"] = dep.reindex(pd.MultiIndex.from_frame(c[["service_date", "direction_id", "trip_id",
                                                            "stop_id"]])).values
    c["stop_name"] = [CORRIDOR_NAMES[d][k] for d, k in zip(c.direction_id, c.corridor_idx)]
    return c


def add_neighbors(c, ev, hw_all):
    """Find the B trains in front of and behind each candidate at its current stop.

    In front = the B train that departed the current stop just before us (same day, direction).
    Behind   = the B train that departs the current stop just after us.
    minutes:  in front = (our departure - its departure) / 60
              behind   = (its arrival at the current stop - our departure) / 60
    stops:    where each train was at the moment we departed, counted as its most recent
              departure anywhere on the B line (a train between Kenmore and Blandford is "at Kenmore").
              in front = its stop index - current stop index; behind = current - its stop index.
    """
    order = ev.sort_values(["service_date", "direction_id", "stop_id", "dep_sec"])[
        ["service_date", "direction_id", "stop_id", "trip_id", "dep_sec", "arr_sec"]].copy()
    g = order.groupby(["service_date", "direction_id", "stop_id"])
    order["front_trip"], order["front_dep"] = g.trip_id.shift(1), g.dep_sec.shift(1)
    order["behind_trip"], order["behind_arr"] = g.trip_id.shift(-1), g.arr_sec.shift(-1)
    c = c.merge(order.drop(columns=["dep_sec", "arr_sec"]),
                on=["service_date", "direction_id", "stop_id", "trip_id"], how="left")
    c["front_minutes"] = (c.dep_sec - c.front_dep) / 60
    c["behind_minutes"] = (c.behind_arr - c.dep_sec) / 60

    pos = {d: {s: i for i, s in enumerate(stops)} for d, stops in BLINE.items()}
    right = hw_all[["service_date", "direction_id", "trip_id", "dep_sec", "stop_id"]].sort_values("dep_sec")
    cur_pos = pd.Series([pos[d][s] for d, s in zip(c.direction_id, c.stop_id)], index=c.index)
    for side, sign in (("front", 1), ("behind", -1)):
        left = c[["service_date", "direction_id", f"{side}_trip", "dep_sec"]].reset_index()
        left = left.dropna(subset=[f"{side}_trip"]).sort_values("dep_sec")
        left["direction_id"] = left.direction_id.astype(right.direction_id.dtype)
        r = right.rename(columns={"trip_id": f"{side}_trip", "dep_sec": "their_dep", "stop_id": "their_stop"})
        m = pd.merge_asof(left, r.sort_values("their_dep"), left_on="dep_sec", right_on="their_dep",
                          by=["service_date", "direction_id", f"{side}_trip"],
                          direction="backward").set_index("index")
        their = pd.Series([pos[d].get(s, np.nan) if isinstance(s, str) else np.nan
                           for d, s in zip(m.direction_id, m.their_stop)], index=m.index)
        # No departure yet = that train is still at its terminal (Government Center westbound,
        # Boston College eastbound), one stop before the first stop the feed records.
        their = their.where(m.their_stop.notna(), -1)
        stops = sign * (their - cur_pos.loc[m.index])
        # Negative = the trip id was reused earlier that day for another run; not a real position.
        c[f"{side}_stops"] = stops.where(stops >= 0) + 0.0  # + 0.0 turns -0.0 into 0.0
    c["hour"] = (c.dep_sec // 3600 % 24).astype(int)
    c["weekday"] = pd.to_datetime(c.service_date).dt.dayofweek  # 0 = Monday
    return c


def corridor_travel_time(con):
    pairs = " OR ".join(f"(from_stop_id='{c[0]}' AND to_stop_id='{c[-1]}')" for c in CORRIDOR.values())
    return con.sql(f"""
        SELECT service_date, trip_id, direction_id, travel_time_sec AS corridor_travel_sec
        FROM '{CLEANED}/traveltimes_bline.parquet' WHERE {pairs}
    """).df()


def fmt_clock(sec):
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


STOP_NAMES = {}


def main():
    con = duckdb.connect()
    hw_all = con.sql(f"SELECT * FROM '{CLEANED}/headways_bline.parquet'").df()
    STOP_NAMES.update(hw_all.drop_duplicates("stop_id").set_index("stop_id").stop_name.to_dict())
    ev_all = load_stop_events(con)

    days = full_service_days(ev_all)
    ev = ev_all.set_index(["service_date", "direction_id"]).loc[lambda x: x.index.isin(days)].reset_index()
    ev.to_parquet(CLEANED / "stop_events.parquet", index=False)

    # How often does a trip have a hole in its stop sequence? (the "missing rows" idea)
    corridor_ev = ev[[s in CORRIDOR[d] for d, s in zip(ev.direction_id, ev.stop_id)]]
    patterns = corridor_ev.groupby(["direction_id", "service_date", "trip_id"]).stop_id.agg(list)
    patterns = pd.Series([presence_pattern(s, d) for (d, _, _), s in patterns.items()], index=patterns.index)

    trips, runs = label_trips(ev)
    trips = trips.merge(corridor_travel_time(con), on=["service_date", "trip_id", "direction_id"], how="left")
    trips["hour"] = (trips.first_dep_sec // 3600 % 24).astype(int)
    trips["weekday"] = pd.to_datetime(trips.service_date).dt.dayofweek
    # In the analysis window = the whole corridor passage happens between 6:00 and 22:00.
    trips["in_window"] = (trips.first_dep_sec >= DAY_START_HOUR * 3600) & (trips.last_dep_sec < DAY_END_HOUR * 3600)
    trips.to_parquet(CLEANED / "trips.parquet", index=False)

    day_trips = trips[trips.in_window]
    cand = add_neighbors(current_stop_candidates(day_trips, ev), ev, hw_all)
    cand.to_parquet(CLEANED / "current_stops.parquet", index=False)

    ex = cand[cand.goes_express].merge(runs, on=["service_date", "direction_id", "trip_id", "corridor_idx"])
    ex = ex.sort_values(["service_date", "dep_sec"])
    express_trains = pd.DataFrame({
        "date": pd.to_datetime(ex.service_date).dt.date,
        "time": ex.dep_sec.map(fmt_clock),
        "direction": ex.direction_id.map({0: "West", 1: "East"}),
        "trip_id": ex.trip_id,
        "src_stop": ex.stop_name,
        "dst_stop": ex.dst_stop,
        "skipped_number": ex.skipped_number.astype(int),
        "trip_behind": ex.behind_trip,
        "trip_behind_minutes": ex.behind_minutes.round(2),
        "trip_behind_stopNum": ex.behind_stops.astype("Int64"),
        "trip_inFront": ex.front_trip,
        "trip_inFront_minutes": ex.front_minutes.round(2),
        "trip_inFront_stopNum": ex.front_stops.astype("Int64"),
    })
    express_trains.to_parquet(PROCESSED / "express_trains.parquet", index=False)

    day_ev = corridor_ev.merge(day_trips[["service_date", "direction_id", "trip_id"]],
                               on=["service_date", "direction_id", "trip_id"])
    stats = {
        "west_trip_num": int((day_trips.direction_id == 0).sum()),
        "east_trip_num": int((day_trips.direction_id == 1).sum()),
        "west_departure_num": int((day_ev.direction_id == 0).sum()),
        "east_departure_num": int((day_ev.direction_id == 1).sum()),
        "west_express_departure_num": int((express_trains.direction == "West").sum()),
        "east_express_departure_num": int((express_trains.direction == "East").sum()),
    }
    (PROCESSED / "GeneralStats.json").write_text(json.dumps(stats, indent=2))

    quality = {
        "service_days_in_raw": int(ev_all.service_date.nunique()),
        "direction_days_kept": int(len(days)),
        "trips_on_kept_days": int(len(patterns)),
        "trips_all_8_stops": int((patterns == "XXXXXXXX").sum()),
        "trips_interior_gap": int(patterns.str.contains(r"X\.+X").sum()),
        "trips_interior_gap_3plus": int(patterns.str.contains(r"X\.{3,}X").sum()),
        "trips_labeled_all_hours": int(len(trips)),
        "trips_in_window": int(len(day_trips)),
        "express_runs_by_skipped_number": express_trains.skipped_number.value_counts().sort_index().to_dict(),
        "express_runs_without_known_dst": int(express_trains.dst_stop.isna().sum()),
    }
    (PROCESSED / "data_quality.json").write_text(json.dumps(quality, indent=2, default=int))
    print(json.dumps(stats, indent=2))
    print(json.dumps(quality, indent=2, default=int))


if __name__ == "__main__":
    main()
