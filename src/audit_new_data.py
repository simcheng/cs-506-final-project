"""Can the MBTA LAMP exports give us ground truth for express runs? (November check-in)

Checks the three LAMP sources against what we need, and writes the answers to
data_processed/new_data_audit.json:
  1. LAMP_RT_ALERTS          rider alerts: does any Green-B alert mention express / skipped stops?
  2. Subway Performance Data daily trip-stop events (datasets/performance/, Jan-Sep 2026): any skip
                             flag? Are its times the same as our ArcGIS-based stop_events? What's new?
  3. LAMP_ALL_RT_fields      the big all-years trip-stop export: any skip flag? How far back does it go?

Needs: download_schedule.py run first, and the two LAMP Tableau exports in datasets/lamp/.
"""
import json

import duckdb

from common import CLEANED, PROCESSED, RAW_ALERTS, RAW_ALL_RT, RAW_PERFORMANCE

KEYWORDS = ["express", "skip", "bypass", "not stop", "run through"]
SKIP_WORDS = ("skip", "express", "schedule_relationship", "bypass")


def one(con, sql):
    return con.sql(sql).fetchone()[0]


def alerts(con):
    con.sql(f"""CREATE VIEW a AS SELECT id, effect, effect_detail, created_datetime,
                "informed_entity.route_id" AS route_id,
                lower(coalesce("header_text.translation.text", '') || ' ' ||
                      coalesce("description_text.translation.text", '')) AS txt
                FROM '{RAW_ALERTS}'""")
    gb = "route_id = 'Green-B'"
    in_study = "created_datetime BETWEEN '2026-01-01' AND '2026-10-01'"
    out = {
        "rows": one(con, "SELECT count(*) FROM a"),
        "distinct_alerts": one(con, "SELECT count(DISTINCT id) FROM a"),
        "created_first": str(one(con, "SELECT min(created_datetime) FROM a")),
        "created_last": str(one(con, "SELECT max(created_datetime) FROM a")),
        "green_b_alerts_all_years": one(con, f"SELECT count(DISTINCT id) FROM a WHERE {gb}"),
        "green_b_alerts_jan_sep_2026": one(con, f"SELECT count(DISTINCT id) FROM a WHERE {gb} AND {in_study}"),
        "green_b_jan_sep_2026_by_effect_detail": dict(con.sql(f"""
            SELECT effect_detail, count(DISTINCT id) n FROM a WHERE {gb} AND {in_study}
            GROUP BY 1 ORDER BY n DESC""").fetchall()),
        "keyword_alerts": {},
    }
    for kw in KEYWORDS:
        out["keyword_alerts"][kw] = {
            "any_route": one(con, f"SELECT count(DISTINCT id) FROM a WHERE txt LIKE '%{kw}%'"),
            "green_b": one(con, f"SELECT count(DISTINCT id) FROM a WHERE {gb} AND txt LIKE '%{kw}%'"),
            "green_b_effect_detail": dict(con.sql(f"""SELECT effect_detail, count(DISTINCT id) FROM a
                WHERE {gb} AND txt LIKE '%{kw}%' GROUP BY 1""").fetchall()),
        }
    out["routes_with_express_alerts_top5"] = dict(con.sql("""SELECT route_id, count(DISTINCT id) n FROM a
        WHERE txt LIKE '%express%' GROUP BY 1 ORDER BY n DESC LIMIT 5""").fetchall())
    return out


def performance(con):
    files = f"'{RAW_PERFORMANCE}/*.parquet'"
    con.sql(f"""CREATE VIEW p AS SELECT * FROM read_parquet({files}, union_by_name=true)
                WHERE route_id = 'Green-B'""")
    cols = [r[0] for r in con.sql("DESCRIBE p").fetchall()]
    # Our events, matched on the same trip at the same stop on the same service day.
    con.sql(f"""CREATE VIEW m AS
        SELECT e.arr_sec, e.dwell, e.hw_branch,
               epoch(timezone('America/New_York', to_timestamp(p.stop_timestamp)))
                 - epoch(e.service_date::TIMESTAMP) AS lamp_arr_sec,
               p.dwell_time_seconds AS lamp_dwell, p.headway_branch_seconds AS lamp_hw
        FROM '{CLEANED}/stop_events.parquet' e
        JOIN p ON p.trip_id = e.trip_id AND p.stop_id = e.stop_id
              AND strptime(p.service_date::VARCHAR, '%Y%m%d')::DATE = e.service_date""")
    return {
        "daily_files_jan_sep_2026": len(list(RAW_PERFORMANCE.glob("*.parquet"))),
        "green_b_rows": one(con, "SELECT count(*) FROM p"),
        "green_b_days": one(con, "SELECT count(DISTINCT service_date) FROM p"),
        "columns": cols,
        "columns_about_skipping": [c for c in cols if any(w in c.lower() for w in SKIP_WORDS)],
        "green_b_share_with_scheduled_departure": one(con, "SELECT avg((scheduled_departure_time IS NOT NULL)::INT) FROM p"),
        "green_b_share_with_occupancy": one(con, "SELECT avg((occupancy_status IS NOT NULL)::INT) FROM p"),
        "our_stop_events": one(con, f"SELECT count(*) FROM '{CLEANED}/stop_events.parquet'"),
        "matched_to_our_stop_events": one(con, "SELECT count(*) FROM m"),
        "matched_arrival_differs": one(con, "SELECT count(*) FROM m WHERE arr_sec IS DISTINCT FROM lamp_arr_sec AND arr_sec IS NOT NULL"),
        "matched_dwell_differs": one(con, "SELECT count(*) FROM m WHERE dwell IS DISTINCT FROM lamp_dwell AND dwell IS NOT NULL"),
        "matched_headway_differs": one(con, "SELECT count(*) FROM m WHERE hw_branch IS DISTINCT FROM lamp_hw AND hw_branch IS NOT NULL"),
        # Among matched rows where both sources have all three values: share where all three agree.
        "matched_rows_with_all_values": one(con, """SELECT count(*) FROM m WHERE arr_sec IS NOT NULL
            AND lamp_arr_sec IS NOT NULL AND dwell IS NOT NULL AND lamp_dwell IS NOT NULL
            AND hw_branch IS NOT NULL AND lamp_hw IS NOT NULL"""),
        "share_identical_where_both_have_values": one(con, """SELECT avg((arr_sec = lamp_arr_sec AND dwell = lamp_dwell
            AND hw_branch = lamp_hw)::INT) FROM m WHERE arr_sec IS NOT NULL AND lamp_arr_sec IS NOT NULL
            AND dwell IS NOT NULL AND lamp_dwell IS NOT NULL AND hw_branch IS NOT NULL AND lamp_hw IS NOT NULL"""),
    }


def all_rt(con):
    f = f"'{RAW_ALL_RT}'"
    cols = [r[0] for r in con.sql(f"DESCRIBE SELECT * FROM {f}").fetchall()]
    return {
        "rows": one(con, f"SELECT count(*) FROM {f}"),
        "first_service_date": str(one(con, f"SELECT min(service_date) FROM {f}")),
        "last_service_date": str(one(con, f"SELECT max(service_date) FROM {f}")),
        "green_b_rows": one(con, f"SELECT count(*) FROM {f} WHERE route_id = 'Green-B'"),
        "columns": cols,
        "columns_about_skipping": [c for c in cols if any(w in c.lower() for w in SKIP_WORDS)],
    }


def main():
    con = duckdb.connect()
    audit = {"LAMP_RT_ALERTS": alerts(con), "subway_performance_data": performance(con),
             "LAMP_ALL_RT_fields": all_rt(con)}
    (PROCESSED / "new_data_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    for name, part in audit.items():
        print(f"== {name}")
        for k, v in part.items():
            if k != "columns":
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
