"""Shared constants for the B line express-run analysis."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_HEADWAYS = ROOT / "datasets" / "headway"
RAW_TRAVEL = ROOT / "datasets" / "travelTime"
PROCESSED = ROOT / "data_processed"
# Intermediate tables the pipeline needs but nobody asked to read directly.
CLEANED = PROCESSED / "data_cleaned"
FIGURES = ROOT / "figures"

# Corridor stops in travel order. Stop ids differ by direction (one platform per side).
# direction_id 0 = westbound (outbound, toward Boston College)
# direction_id 1 = eastbound (inbound, toward Park Street)
CORRIDOR = {
    0: ["71151", "70149", "70147", "70145", "170141", "170137", "70135", "70131"],
    1: ["70130", "70134", "170136", "170140", "70144", "70146", "70148", "71150"],
}
CORRIDOR_NAMES = {
    0: ["Kenmore", "Blandford St", "BU East", "BU Central", "Amory St", "Babcock St", "Packard's Cnr", "Harvard Ave"],
    1: ["Harvard Ave", "Packard's Cnr", "Babcock St", "Amory St", "BU Central", "BU East", "Blandford St", "Kenmore"],
}
# The stops a train can skip: everything between the two ends of the corridor.
INNER = {d: stops[1:-1] for d, stops in CORRIDOR.items()}

# The whole B line in travel order, used to count how many stops back the next train is.
BLINE = {
    0: ["70196", "70159", "70157", "70155", "70153", "71151", "70149", "70147", "70145", "170141",
        "170137", "70135", "70131", "70129", "70127", "70125", "70121", "70117", "70115", "70113", "70111"],
    1: ["70110", "70112", "70114", "70116", "70120", "70124", "70126", "70128", "70130", "70134",
        "170136", "170140", "70144", "70146", "70148", "71150", "70152", "70154", "70156", "70158", "70200"],
}

# A dwell (departure minus arrival) under this many seconds means the train rolled through the
# stop without opening its doors. Dwell times split into two groups, 10-20 s and 30-60 s,
# with a dip around 20-30 s (see figures/fig1_dwell_and_travel.png).
PASS_THRESHOLD_SEC = 25
# Express run: the train stops at a corridor stop (the src stop), then rolls through the next
# stop. Any number of skipped stops counts. The run ends at the first stop where it stops again
# (dst stop), which can be beyond the corridor.
# Analysis window, local hours: keep 6:00 <= departure < 22:00. Outside it, runs of skipped stops
# are mostly empty late-night / early-morning trains, not dispatcher expresses (see fig5).
DAY_START_HOUR, DAY_END_HOUR = 6, 22
# Westbound express rates are highest in these hours (start inclusive, end exclusive).
PEAK_WINDOWS = [(11, 13), (16, 18)]

DIRECTION_LABEL = {0: "Westbound (Kenmore → Harvard Ave)", 1: "Eastbound (Harvard Ave → Kenmore)"}
