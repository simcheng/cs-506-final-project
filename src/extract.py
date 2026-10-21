"""Step 1: shrink the raw MBTA CSVs (~15 GB) to small B line parquet files.

Outputs (data_processed/data_cleaned/):
  headways_bline.parquet       every B line departure event, all stops
  traveltimes_bline.parquet    B line travel times ending at any B line stop (gives every arrival
                               time, so we know where a train stopped and where it rolled through)
"""
import duckdb

from common import BLINE, CLEANED, RAW_HEADWAYS, RAW_TRAVEL


def sql_list(ids):
    return "(" + ",".join(f"'{s}'" for s in ids) + ")"


def main():
    CLEANED.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    bline_stops = sql_list(BLINE[0] + BLINE[1])

    # Read everything as text; monthly files mix "NA" and "" for missing headways.
    con.sql(f"""
        COPY (
          SELECT service_date::DATE AS service_date, trip_id, direction_id::INT AS direction_id,
                 stop_id, stop_name,
                 stop_departure_sec::INT AS dep_sec,
                 TRY_CAST(headway_trunk_seconds AS INT) AS hw_trunk,
                 TRY_CAST(headway_branch_seconds AS INT) AS hw_branch
          FROM read_csv('{RAW_HEADWAYS}/*_Headways.csv', all_varchar=true, union_by_name=true, header=true)
          WHERE route_id = 'Green-B' AND stop_id IN {bline_stops}
        ) TO '{CLEANED}/headways_bline.parquet'
    """)

    con.sql(f"""
        COPY (
          SELECT service_date::DATE AS service_date, trip_id, direction_id::INT AS direction_id,
                 from_stop_id, to_stop_id,
                 from_stop_departure_sec::INT AS from_dep_sec,
                 to_stop_arrival_sec::INT AS to_arr_sec,
                 travel_time_sec::INT AS travel_time_sec
          FROM read_csv('{RAW_TRAVEL}/*_LR*TravelTimes.csv', all_varchar=true, union_by_name=true, header=true)
          WHERE route_id = 'Green-B'
            AND to_stop_id IN {bline_stops}
        ) TO '{CLEANED}/traveltimes_bline.parquet'
    """)

    for name in ("headways_bline", "traveltimes_bline"):
        n, days = con.sql(
            f"SELECT count(*), count(DISTINCT service_date) FROM '{CLEANED}/{name}.parquet'"
        ).fetchone()
        print(f"{name}: {n:,} rows over {days} service days")


if __name__ == "__main__":
    main()
