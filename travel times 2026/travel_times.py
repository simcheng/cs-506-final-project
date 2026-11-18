import argparse
import csv
from datetime import date
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator


BIN_SECONDS = 20
REQUIRED_COLUMNS = {
    "service_date", "trip_id", "from_stop_name", "to_stop_name", "travel_time_sec", "from_stop_departure_sec"
}


def read_travel_times(csv_paths, service_date, from_stop, to_stop):
    trips = {}
    for csv_path in csv_paths:
        with csv_path.open(newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{csv_path.name}: missing columns {', '.join(sorted(missing))}")

            for row in reader:
                if (
                    row["service_date"] != service_date
                    or row["from_stop_name"] != from_stop
                    or row["to_stop_name"] != to_stop
                ):
                    continue

                trip_id = (row["trip_id"] or "").strip()
                departure = float(row["from_stop_departure_sec"])
                if departure < 28800 or departure > 79200:
                    continue
                try:
                    seconds = float(row["travel_time_sec"])
                    if not trip_id or not math.isfinite(seconds) or seconds < 0:
                        raise ValueError
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"{csv_path.name}, line {reader.line_num}: expected a trip ID "
                        "and a finite, nonnegative travel time."
                    ) from exc

                previous = trips.setdefault(trip_id, seconds)
                if previous != seconds:
                    raise ValueError(
                        f"{csv_path.name}, line {reader.line_num}: trip {trip_id!r} "
                        f"has conflicting travel times ({previous:g} and {seconds:g} seconds)."
                    )

    return list(trips.values())


def save_histogram(times, from_stop, to_stop, service_date, output):
    first_edge = int(min(times) // BIN_SECONDS) * BIN_SECONDS
    last_edge = (int(max(times) // BIN_SECONDS) + 1) * BIN_SECONDS
    edges = range(first_edge, last_edge + BIN_SECONDS, BIN_SECONDS)

    fig, ax = plt.subplots(figsize=(11, 5))
    try:
        ax.hist(times, bins=edges, color="#237baf", edgecolor="white")
        ax.set_xlim(first_edge, last_edge)
        tick_step = BIN_SECONDS * max(1, math.ceil((len(edges) - 1) / 10))
        ax.xaxis.set_major_locator(MultipleLocator(tick_step))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Travel time (seconds)")
        ax.set_ylabel("Unique trip count")
        ax.set_title(
            f"{from_stop} to {to_stop}\n{service_date} | {len(times):,} trips | 30-second bins"
        )
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output, dpi=180)
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot a histogram of station-to-station travel times.")
    parser.add_argument("from_stop", help="Exact start station name")
    parser.add_argument("to_stop", help="Exact end station name")
    parser.add_argument("service_date", help="Service date in YYYY-MM-DD format")
    parser.add_argument("-o", "--output", type=Path, help="Output PNG path")
    args = parser.parse_args()

    try:
        service_date = date.fromisoformat(args.service_date)
        if service_date.isoformat() != args.service_date:
            raise ValueError
    except ValueError:
        parser.error("Enter the service date in YYYY-MM-DD format.")

    csv_paths = sorted(Path(__file__).resolve().parent.glob("*.csv"))
    if not csv_paths:
        parser.error("No CSV files found next to travel_times.py.")

    output = args.output or Path(f"travel_time_histogram_{service_date}.png")
    try:
        times = read_travel_times(csv_paths, args.service_date, args.from_stop, args.to_stop)
        if not times:
            parser.error("No trips match the service date and directed station pair.")
        save_histogram(times, args.from_stop, args.to_stop, args.service_date, output)
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))

    print(f"Plotted {len(times):,} unique trips. Saved chart to {output.resolve()}")


if __name__ == "__main__":
    main()
