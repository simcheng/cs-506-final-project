# CS 506 Final Project: Predicting B Line Express Runs Near BU

## Problem

BU students living in West Campus and Allston sometimes have to get off a train at peak hours because it becomes an express train, running directly to Harvard Avenue after two trains bunch together.

On the B line around the BU stops (Kenmore to Harvard Ave), we want to predict **if, and when**, a B line train *t* arriving at stop *s1* will go express to a station *s2*. Or if this is impossible, we could do smth easier:

using certain feature parameters to predict the possibility of a train is going express. The potential features are:
- Period of the day, e.g., morning or afternoon is peak period for students go to or go from BU.
- How far away the train behind, in number of stops, or in time.
- Weather?

## Goals

1. On the B line around the BU stops, predict whether a B line train will run express, based on:
   - Time since the departure of the last train at the same station *s1* (seconds)
   - Historical travel time between stations *s1* and *s2* (seconds)
   - Absolute difference between the scheduled and actual time of train *t* at *s1* (seconds)
2. Identify when and where express runs happen most often (hour, day of week, stop).

### Fallback

Predict wait time at BU stops based on class schedules and weather. This is a regression on the minutes until the next B train at BU East, BU Central, or Amory St, using hour, day, and upstream headway.

## How to Measure the Goal

Successfully predict whether a train will become an express train based on the input parameters of our model:

- Time since the departure of the last train at the same station (seconds)
- Historical travel time between stations (seconds)
- Absolute difference between the scheduled and actual time of train *t* at *s1* (seconds)

## Timeline (~10 weeks, due 12/9)

| Weeks | Dates | Tasks |
|---|---|---|
| 1–2 | Oct 5 – Oct 18 | Download headways and travel-time data and the schedule. Confirm that B line surface stops are covered. Define and extract express labels (trips with no events at intermediate stops). |
| 3–4 | Oct 19 – Nov 1 | Clean the data to keep only B line trains at stops from Kenmore to Harvard Ave. Build features: previous-train headway, historical travel time *s1* → *s2*, and schedule deviation. *(October check-in)* |
| 5–6 | Nov 2 – Nov 15 | Visualize express trains against each parameter. |
| 7–8 | Nov 16 – Nov 29 | Build data models (pending). *(November check-in)* |
| 9–10 | Nov 30 – Dec 9 | Makefile, tests, GitHub workflow, final README, and recorded presentation. |

## Data Sources

| Data | Purpose | Link |
|---|---|---|
| MBTA Rapid Transit Headways | Time between the current and last train at a station | [ArcGIS](https://mbta-massdot.opendata.arcgis.com/datasets/fffd5e8ff7f042deb7834f3badf49e58/about) |
| MBTA Rapid Transit Travel Times | Travel time between stations, which captures pedestrian and road traffic effects | [ArcGIS](https://mbta-massdot.opendata.arcgis.com/datasets/5f71a5c035fc4a4dad1b7fa73ba27ef8/about) |
| MBTA B Line Schedule | The MBTA's proposed schedule | [PDF](https://cdn.mbta.com/sites/default/files/media/route_pdfs/SUB-S3-P4.pdf) |

## Setup

Requires Python 3 (tested on 3.14).

```bash
git clone <repo-url>
cd cs-506-final-project
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Download the monthly CSVs (Jan–Sep 2026) from the two ArcGIS links above and put them in:

- `datasets/headway/`: files named `YYYY-MM_Headways.csv`
- `datasets/travelTime/`: the light-rail files, named `YYYY-MM_LR*TravelTimes.csv` (the `HR` heavy-rail files are not used)

The raw data is about 15 GB and is gitignored.

## Running

Run the three scripts in order from `src/`:

```bash
cd src
../.venv/bin/python extract.py     # raw CSVs → B line parquet files in data_processed/data_cleaned/
../.venv/bin/python process.py     # express labels → express_trains.parquet, GeneralStats.json
../.venv/bin/python visualize.py   # figures/ + dwellTime.csv + figure_data/*.csv
```

Settings such as the dwell threshold (25 s) and the 6:00–22:00 window are in `src/common.py`.

## Directory Tree

```
cs-506-final-project/
├── datasets/                     # raw MBTA CSVs (not in git)
│   ├── headway/
│   └── travelTime/
├── src/
│   ├── common.py                 # stop lists, thresholds, paths
│   ├── extract.py                # step 1: filter raw CSVs to the B line
│   ├── process.py                # step 2: label express runs, find trains in front/behind
│   └── visualize.py              # step 3: figures and the data behind them
├── data_processed/
│   ├── express_trains.parquet    # one row per express departure
│   ├── dwellTime.csv             # data for fig1
│   ├── GeneralStats.json         # trip, departure and express departure counts
│   ├── data_quality.json         # cleaning counts
│   ├── figure_data/              # data for fig2–fig7
│   └── data_cleaned/             # intermediate parquet files (not in git)
├── figures/                      # fig1–fig7 PNGs
└── requirements.txt
```
