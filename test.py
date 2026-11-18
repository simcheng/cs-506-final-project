#!/usr/bin/env python3
import argparse
import csv
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("date")
    parser.add_argument("start")
    parser.add_argument("end")
    args = parser.parse_args()

    with open(args.file, newline="", encoding="utf-16") as source:
        reader = csv.DictReader(source)
        bueast = 0
        amory = 0
        for row in reader:
            if args.date == row["service_date"]:
                if row["stop_name"] == args.start:
                    bueast += 1
                elif row["stop_name"] == args.end:
                    amory += 1

        print(bueast, amory)

if __name__ == "__main__":
    main()