#!/usr/bin/env python3
import argparse
import csv
import sys

STOPS = {
    "Kenmore",
    "Blandford Street",
    "Boston University East",
    "Boston University Central",
    "Amory Street",
    "Babcock Street",
    "Packard's Corner",
    "Harvard Avenue",
}

def main():
    parser = argparse.ArgumentParser(description="Filter Green-B CSV rows by stops.")
    parser.add_argument("input_csv", help="Path to the input CSV file")
    args = parser.parse_args()

    with open(args.input_csv, newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"route_id", "from_stop_name", "to_stop_name"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            parser.error(f"Missing required columns: {', '.join(sorted(missing))}")

        writer = csv.DictWriter(sys.stdout, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            if (
                row["route_id"] == "Green-B"
                and row["from_stop_name"] in STOPS
                and row["to_stop_name"] in STOPS
            ):
                writer.writerow(row)

if __name__ == "__main__":
    main()