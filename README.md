# CS 506 Final Project: Predicting B Line Express Runs Near BU

## Problem

BU students living in West Campus and Allston sometimes have to get off a train at peak hours because it becomes an express train, running directly to Harvard Avenue after two trains bunch together.

On the B line around the BU stops (Kenmore to Harvard Ave), we want to predict **if, and when**, a B line train *t* arriving at stop *s1* will go express to a station *s2*.

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

## How to Collect Data

1. Download scripts in the repo use Python `requests` to pull the CSVs from the ArcGIS portal and the GTFS zip, so the Makefile can reproduce everything.
2. Filter to route Green-B and the stops from Kenmore to Harvard Ave.
3. Join headway, travel-time, and schedule records on trip ID and stop, then derive the express labels.
