import argparse
import csv
import sys

def main():
    files = [
             "filtered02.csv", 
             "filtered03.csv", 
             "filtered04.csv", 
             "filtered05.csv", 
             "filtered06.csv", 
             "filtered07.csv", 
             "filtered08.csv", 
             "filtered09.csv", 
             "filtered10.csv", ]

    outputs = [
             "headways02.csv", 
             "headways03.csv", 
             "headways04.csv", 
             "headways05.csv", 
             "headways06.csv", 
             "headways07.csv", 
             "headways08.csv", 
             "headways09.csv", 
             "headways10.csv", ]


    for i in range(10):
        with open(files[i], newline="", encoding="utf-16") as source:
            with open(outputs[i], newline="", mode='w', encoding="utf-16") as output:
                reader = csv.DictReader(source)

                writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
                writer.writeheader()

                for row in reader:
                    departure = float(row["stop_departure_sec"])
                    if departure > 28800 and departure < 79200:
                        writer.writerow(row)


if __name__ == "__main__":
    main()