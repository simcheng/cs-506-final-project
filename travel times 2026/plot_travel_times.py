import argparse
import csv
from datetime import date, datetime
import math
from pathlib import Path
import sys


def hourly_travel_times(csv_path, service_date, from_stop, to_stop):
    required = {
        "service_date", "trip_id", "from_stop_name", "to_stop_name",
        "from_stop_departure_datetime", "travel_time_sec",
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
                departure = datetime.fromisoformat(
                    row["from_stop_departure_datetime"].strip().replace("Z", "+00:00")
                )
                seconds = float(row["travel_time_sec"])
                if not math.isfinite(seconds) or seconds < 0:
                    raise ValueError("travel_time_sec must be finite and nonnegative")
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(
                    f"Invalid matching row near CSV line {reader.line_num}: {exc}"
                ) from exc

            record = (departure, seconds)
            if trip_id in trips and trips[trip_id] != record:
                raise ValueError(
                    f"Conflicting records for trip_id {trip_id!r}; "
                    "each trip must have a single departure and travel time."
                )
            trips[trip_id] = record

    if not trips:
        raise ValueError("No trips match the service date and directed stop pair.")

    buckets = [[] for _ in range(24)]
    for departure, seconds in trips.values():
        buckets[departure.hour].append(seconds)
    counts = [len(bucket) for bucket in buckets]
    means = [math.fsum(bucket) / len(bucket) if bucket else None for bucket in buckets]
    return means, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("date", help="Service date, YYYY-MM-DD")
    parser.add_argument("from_stop", help="Exact from_stop_name (quote names with spaces)")
    parser.add_argument("to_stop", help="Exact to_stop_name (quote names with spaces)")
    parser.add_argument("-o", "--output", type=Path, help="Output PNG path")
    args = parser.parse_args()

    try:
        if date.fromisoformat(args.date).isoformat() != args.date:
            raise ValueError("Use YYYY-MM-DD for the date.")
        means, counts = hourly_travel_times(
            args.csv_file, args.date, args.from_stop, args.to_stop
        )
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))

    try:
        import matplotlib
        matplotlib.use("Agg")  # Works in a terminal without a graphical desktop.
        import matplotlib.pyplot as plt
    except ImportError:
        parser.error("Install matplotlib first: python3 -m pip install matplotlib")

    hours = [hour for hour in range(24) if means[hour] is not None]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(hours, [means[hour] for hour in hours], width=0.9, color="#237baf")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{hour:02d}:00" for hour in range(24)], rotation=45, ha="right")
    ax.set_xlim(-0.6, 23.6)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Departure hour (as recorded in CSV)")
    ax.set_ylabel("Average travel time (seconds)")
    ax.set_title(f"{args.from_stop} → {args.to_stop}\nService date: {args.date}")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.text(
        0.5, 0.02,
        f"{sum(counts)} unique trips • Each bar covers one hour • Gaps indicate no trips",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    output = args.output or Path(f"travel_time_{args.date}.png")
    try:
        fig.savefig(output, dpi=180)
    except OSError as exc:
        parser.error(str(exc))
    finally:
        plt.close(fig)

    print("Departure hour | Unique trips | Average seconds")
    for hour in range(24):
        average = f"{means[hour]:.2f}" if means[hour] is not None else "N/A"
        print(f"{hour:02d}:00          | {counts[hour]:12d} | {average}")
    print(f"Saved chart: {output.resolve()}")


if __name__ == "__main__":
    main()
