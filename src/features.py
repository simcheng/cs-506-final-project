"""Step 4: the model table. One row per current-stop candidate from process.py (a train that stopped
at a corridor stop with 3 corridor stops still ahead). Label: goes_express (the dwell-based express
run label from process.py).

Prediction time = the moment train t ARRIVES at the current stop s1 (the README question: "will a
train arriving at s1 go express?"). That is when a rider on the platform decides whether to board.
Every feature uses only what is known at that moment.

Features
  front_gap_min    minutes since the previous B train departed s1 (0 = it was still at the
                   platform when we arrived)                                             (README feature 1)
  behind_eta_min   estimated minutes until the next B train reaches s1: historical travel time
                   (training months) from where it was when we arrived, minus the time it has
                   already spent on that leg; lower bound if it is still at its terminal (README feature 2)
  sched_dev_min    actual minus scheduled arrival at s1, signed (+ = late). NaN for ADDED
                   trips, whose schedule match is a guess                                (README feature 3)
  delay_alert      1 if a Green-B DELAY alert (LAMP_RT_ALERTS) was active
  hour, weekend, stop (direction + current stop)

Not features:
  dwell_sec               how long t stood at s1. Only known when it leaves, and partly caused by
                          the express decision itself (announcement, riders getting off). Kept to
                          show that effect in the ablation.
  behind_minutes_actual   when the next train really arrived at s1 (future). Kept to check the ETA.

Split by date: train Jan-Jun, validation Jul, test Aug-Sep 2026.
Output: data_processed/data_cleaned/model_table.parquet
"""
import duckdb
import numpy as np
import pandas as pd

from common import BLINE, CLEANED, RAW_ALERTS

SPLIT_BY_MONTH = {**{m: "train" for m in range(1, 7)}, 7: "val", 8: "test", 9: "test"}


def historical_travel(ev):
    """tt[d][p, s] = median seconds from departing B line stop p to arriving at stop s."""
    tt = {}
    for d, stops in BLINE.items():
        e = ev[(ev.direction_id == d) & ev.stop_id.isin(stops)]
        wide = e.groupby(["service_date", "trip_id", "stop_id"])[["dep_sec", "arr_sec"]].first().unstack("stop_id")
        dep = wide["dep_sec"].reindex(columns=stops).to_numpy(float)
        arr = wide["arr_sec"].reindex(columns=stops).to_numpy(float)
        n = len(stops)
        tt[d] = np.full((n, n), np.nan)
        for p in range(n):
            for s in range(p + 1, n):
                leg = arr[:, s] - dep[:, p]
                if np.isfinite(leg).any():
                    tt[d][p, s] = np.nanmedian(leg)
    return tt


def train_behind(c, ev, tt):
    """ETA of the next B train to s1, from where it was when we arrived (its last B line departure)."""
    right = (ev.dropna(subset=["dep_sec"])[["service_date", "direction_id", "trip_id", "stop_id", "dep_sec"]]
             .rename(columns={"trip_id": "behind_trip", "stop_id": "their_stop", "dep_sec": "their_dep"}))
    right["their_dep"] = right.their_dep.astype("int64")
    left = c[["service_date", "direction_id", "behind_trip", "now_sec"]].dropna(subset=["behind_trip"]).reset_index()
    left["now_sec"] = left.now_sec.astype("int64")
    m = pd.merge_asof(left.sort_values("now_sec"), right.sort_values("their_dep"), left_on="now_sec",
                      right_on="their_dep", by=["service_date", "direction_id", "behind_trip"],
                      direction="backward").set_index("index")
    pos = {d: {s: i for i, s in enumerate(stops)} for d, stops in BLINE.items()}
    eta = pd.Series(np.nan, index=c.index)
    for i, d, here, there, now, t_them in zip(m.index, m.direction_id, c.loc[m.index, "stop_id"], m.their_stop,
                                             m.now_sec, m.their_dep):
        s = pos[d][here]
        if not isinstance(there, str):
            # Still at its terminal (Government Center WB / Boston College EB), one stop before the
            # first stop the feed records: at least a full trip from that first stop away.
            eta[i] = tt[d][0, s] / 60
            continue
        p = pos[d].get(there)
        if p is None or p >= s:
            continue  # trip id reused earlier that day; not a real position
        eta[i] = (tt[d][p, s] - (now - t_them)) / 60
    return eta


def delay_alert_active(c):
    """1 if any Green-B DELAY alert was active at the prediction time.

    The alert archive stores every update of an alert; open-ended alerts get their end time pushed
    ~2 h past each update. So an alert is active from its first start until it was closed, or until
    its last update if it was never marked closed.
    """
    a = duckdb.sql(f"""
        SELECT id, min("active_period.start_datetime") AS start_t,
               coalesce(max(closed_datetime), max(last_modified_datetime)) AS end_t
        FROM '{RAW_ALERTS}'
        WHERE "informed_entity.route_id" = 'Green-B' AND effect_detail = 'DELAY'
          AND created_datetime BETWEEN '2025-12-01' AND '2026-10-02'
        GROUP BY id
    """).df()
    t = (c.service_date + pd.to_timedelta(c.now_sec, unit="s")).to_numpy()
    active = np.zeros(len(c), dtype=bool)
    for s, e in zip(a.start_t.to_numpy(), a.end_t.to_numpy()):
        active |= (t >= s) & (t <= e)
    return active.astype(int), len(a)


def main():
    c = pd.read_parquet(CLEANED / "current_stops.parquet")
    ev = pd.read_parquet(CLEANED / "stop_events.parquet")
    sched = pd.read_parquet(CLEANED / "schedule_bline.parquet")
    for df in (ev, sched):
        df["service_date"] = pd.to_datetime(df.service_date).astype(c.service_date.dtype)
        df["direction_id"] = df.direction_id.astype(c.direction_id.dtype)

    keys = ["service_date", "direction_id", "trip_id", "stop_id"]
    c = c.merge(ev[keys + ["arr_sec", "dwell"]].drop_duplicates(keys), on=keys, how="left")
    c["now_sec"] = c.arr_sec  # every candidate stopped at s1, so it has an arrival time
    c["split"] = pd.to_datetime(c.service_date).dt.month.map(SPLIT_BY_MONTH)

    train_days = c.loc[c.split == "train", "service_date"].unique()
    tt = historical_travel(ev[ev.service_date.isin(train_days)])
    c["behind_eta_min"] = train_behind(c, ev, tt)

    c = c.merge(sched.drop_duplicates(keys), on=keys, how="left")
    added = c.trip_id.str.startswith("ADDED")
    c.loc[added, "sched_arr_sec"] = np.nan
    c["delay_alert"], n_alerts = delay_alert_active(c)

    # If the train in front is still at the platform when we arrive, when it will leave is future
    # information: all we know is "still there", so the gap is 0.
    front_gap = ((c.now_sec - c.front_dep) / 60).clip(lower=0)
    out = pd.DataFrame({
        "service_date": c.service_date, "direction_id": c.direction_id, "trip_id": c.trip_id,
        "stop_id": c.stop_id, "stop": np.where(c.direction_id == 0, "W ", "E ") + c.stop_name,
        "arr_sec": c.now_sec.astype(int), "hour": (c.now_sec // 3600 % 24).astype(int),
        "weekend": (c.weekday >= 5).astype(int),
        "front_gap_min": front_gap,
        "behind_eta_min": c.behind_eta_min,
        "sched_dev_min": (c.now_sec - c.sched_arr_sec) / 60,
        "delay_alert": c.delay_alert,
        "dwell_sec": c.dwell,
        "behind_minutes_actual": (c.behind_arr - c.now_sec) / 60,
        "added_trip": added.astype(int),
        "goes_express": c.goes_express.astype(int),
        "split": c.split,
    })
    out.to_parquet(CLEANED / "model_table.parquet", index=False)

    print(f"model_table: {len(out):,} rows, {out.goes_express.sum():,} express ({out.goes_express.mean():.2%})")
    print(f"{n_alerts} Green-B DELAY alerts; {out.delay_alert.mean():.1%} of rows arrive during one")
    print(out.groupby("split").goes_express.agg(["size", "sum", "mean"]))
    print("missing share per column:")
    print(out.isna().mean().round(4)[lambda s: s > 0])


if __name__ == "__main__":
    main()
