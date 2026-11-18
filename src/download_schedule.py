"""Step 1b: download MBTA LAMP Subway Performance Data and keep the B line schedule columns.

The ArcGIS headway / travel-time CSVs have no schedule, so they cannot give the README's third
feature (scheduled vs actual time at the current stop). LAMP's daily Subway Performance files have
the same arrival / departure events (verified identical) plus the scheduled times for each trip-stop.
They do NOT say whether a train ran express (the MBTA confirmed this by email).

Outputs:
  datasets/performance/YYYY-MM-DD.parquet          raw daily files (not in git, ~360 MB)
  data_processed/data_cleaned/schedule_bline.parquet
      one row per B line trip x stop: scheduled departure / arrival and scheduled headway
"""
import datetime as dt
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import duckdb

from common import BLINE, CLEANED, RAW_PERFORMANCE, sql_list

URL = "https://performancedata.mbta.com/lamp/subway-on-time-performance-v1/{d}-subway-on-time-performance-v1.parquet"
FIRST_DAY, LAST_DAY = dt.date(2026, 1, 1), dt.date(2026, 9, 30)  # same months as the ArcGIS files


def fetch(day):
    out = RAW_PERFORMANCE / f"{day}.parquet"
    if out.exists():
        return "cached"
    req = urllib.request.Request(URL.format(d=day), headers={"User-Agent": "python-urllib"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}"
    tmp = out.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.rename(out)
    return "downloaded"


def main():
    RAW_PERFORMANCE.mkdir(parents=True, exist_ok=True)
    CLEANED.mkdir(parents=True, exist_ok=True)
    days = [FIRST_DAY + dt.timedelta(n) for n in range((LAST_DAY - FIRST_DAY).days + 1)]
    with ThreadPoolExecutor(4) as pool:
        results = dict(zip(days, pool.map(fetch, days)))
    missing = [str(d) for d, r in results.items() if r.startswith("HTTP")]
    print(f"{len(days) - len(missing)} daily files available, missing: {missing or 'none'}")

    con = duckdb.connect()
    con.sql(f"""
        COPY (
          SELECT strptime(service_date::VARCHAR, '%Y%m%d')::DATE AS service_date, trip_id,
                 direction_id::INT AS direction_id, stop_id,
                 scheduled_departure_time AS sched_dep_sec, scheduled_arrival_time AS sched_arr_sec,
                 scheduled_headway_branch AS sched_hw_branch
          FROM read_parquet('{RAW_PERFORMANCE}/*.parquet', union_by_name=true)
          WHERE route_id = 'Green-B' AND stop_id IN {sql_list(BLINE[0] + BLINE[1])}
        ) TO '{CLEANED}/schedule_bline.parquet'
    """)
    n, n_sched, n_days = con.sql(f"""SELECT count(*), count(sched_dep_sec), count(DISTINCT service_date)
                                     FROM '{CLEANED}/schedule_bline.parquet'""").fetchone()
    print(f"schedule_bline: {n:,} rows over {n_days} days, {n_sched / n:.1%} with a scheduled departure")


if __name__ == "__main__":
    main()
