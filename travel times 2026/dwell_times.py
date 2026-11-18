import argparse
from collections import defaultdict
import csv
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


BIN_SECONDS = 5
EVENT_COLUMNS = (
    ("to_stop_name", "to_stop_id", "to_stop_arrival_sec", False),
    ("from_stop_name", "from_stop_id", "from_stop_departure_sec", True),
)
REQUIRED_COLUMNS = {"service_date", "trip_id"} | {
    column for name, stop, seconds, _ in EVENT_COLUMNS for column in (name, stop, seconds)
}


def read_dwell_times(csv_paths, station, service_date):
    events = defaultdict(set)
    for path in csv_paths:
        with path.open(newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path.name}: missing columns {', '.join(sorted(missing))}")

            for row in reader:
                if row["service_date"] != service_date or not row["trip_id"]:
                    continue
                departure = float(row["from_stop_departure_sec"])
                if departure < 28800 or departure > 79200:
                    continue
                for name, stop, seconds, is_departure in EVENT_COLUMNS:
                    if row[name] != station or not row[stop] or not row[seconds]:
                        continue
                    try:
                        timestamp = int(row[seconds])
                    except ValueError as exc:
                        raise ValueError(
                            f"{path.name}, line {reader.line_num}: invalid {seconds}."
                        ) from exc
                    events[row["trip_id"], row[stop]].add((timestamp, is_departure))

    durations = []
    for stop_events in events.values():
        arrival = None
        # Arrivals sort before departures at the same second, allowing zero dwell.
        for timestamp, is_departure in sorted(stop_events):
            if not is_departure:
                arrival = timestamp
            elif arrival is not None:
                durations.append(timestamp - arrival)
                arrival = None
    return durations


def save_histogram(durations, station, service_date, output):
    last_edge = (max(durations) // BIN_SECONDS + 1) * BIN_SECONDS
    edges = range(0, last_edge + BIN_SECONDS, BIN_SECONDS)
    fig, ax = plt.subplots(figsize=(11, 5))
    try:
        ax.hist(durations, bins=edges, color="#237baf", edgecolor="white")
        ax.set_xlim(0, last_edge)
        tick_stride = max(1, (len(edges) - 1 + 9) // 10)
        ax.set_xticks(edges[::tick_stride])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Dwell time (seconds)")
        ax.set_ylabel("Station visits")
        ax.set_title(
            f"{station}\n{service_date} | {len(durations):,} visits | {BIN_SECONDS}-second bins"
        )
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output, dpi=180)
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot a histogram of train dwell times at a station.")
    parser.add_argument("station", help="Exact station name as recorded in the CSVs")
    parser.add_argument("service_date", help="Service date in YYYY-MM-DD format")
    parser.add_argument("-o", "--output", type=Path, help="Output PNG path")
    args = parser.parse_args()

    try:
        if date.fromisoformat(args.service_date).isoformat() != args.service_date:
            raise ValueError
    except ValueError:
        parser.error("Enter the service date in YYYY-MM-DD format.")

    csv_paths = sorted(Path(__file__).resolve().parent.glob("*.csv"))
    if not csv_paths:
        parser.error("No CSV files found next to dwell_times.py.")

    output = args.output or Path(f"dwell_time_histogram_{args.service_date}.png")
    try:
        durations = read_dwell_times(csv_paths, args.station, args.service_date)
        if not durations:
            parser.error("No matched arrival and departure events for this station and service date.")
        save_histogram(durations, args.station, args.service_date, output)
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))

    print(f"Plotted {len(durations):,} station visits. Saved chart to {output.resolve()}")


if __name__ == "__main__":
    main()
