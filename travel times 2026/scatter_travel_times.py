import argparse
import csv
from datetime import date
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_trips(csv_path, service_date, from_stop, to_stop):
    required = {
        "service_date", "trip_id", "from_stop_name", "to_stop_name",
        "from_stop_departure_sec", "travel_time_sec",
    }
    trips = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
        for row in reader:
            if not (
                row["service_date"] == service_date
                and row["from_stop_name"] == from_stop
                and row["to_stop_name"] == to_stop
            ):
                continue
            try:
                trip_id = row["trip_id"].strip()
                if not trip_id:
                    raise ValueError("empty trip_id")
                departure = float(row["from_stop_departure_sec"])
                travel_time = float(row["travel_time_sec"])
                if not all(math.isfinite(v) and v >= 0 for v in (departure, travel_time)):
                    raise ValueError("departure and travel seconds must be finite and nonnegative")
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(
                    f"Invalid matching row near CSV line {reader.line_num}: {exc}"
                ) from exc
            point = (departure, travel_time)
            if trip_id in trips and trips[trip_id] != point:
                raise ValueError(f"Conflicting records for trip_id {trip_id!r}.")
            trips[trip_id] = point
    if not trips:
        raise ValueError("No trips match the service date and directed stop pair.")
    return trips


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("date", help="Service date in YYYY-MM-DD format")
    parser.add_argument("from_stop", help="Exact from_stop_name")
    parser.add_argument("to_stop", help="Exact to_stop_name")
    parser.add_argument("-o", "--output", type=Path, help="Output PNG path")
    args = parser.parse_args()
    try:
        if date.fromisoformat(args.date).isoformat() != args.date:
            raise ValueError("Use YYYY-MM-DD for the date.")
        trips = read_trips(args.csv_file, args.date, args.from_stop, args.to_stop)
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))

    departures, travel_times = zip(*trips.values())
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(departures, travel_times, s=30, alpha=0.65, color="#237baf", edgecolors="none")
    ax.set_xlabel("Departure time (seconds after midnight)")
    ax.set_ylabel("Travel time (seconds)")
    ax.set_title(f"{args.from_stop} → {args.to_stop}\nService date: {args.date}")
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.text(0.5, 0.02, f"{len(trips)} unique trips • One point per trip", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    output = args.output or Path(f"travel_time_scatter_{args.date}.png")
    try:
        fig.savefig(output, dpi=180)
    except OSError as exc:
        parser.error(str(exc))
    finally:
        plt.close(fig)
    print(f"Plotted {len(trips)} unique trips.")
    print(f"Saved chart: {output.resolve()}")


if __name__ == "__main__":
    main()
