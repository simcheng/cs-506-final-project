"""Step 3: figures for the October check-in.

Writes PNGs to figures/ and the exact numbers behind each figure to data_processed/
(dwellTime.csv for fig1, figure_data/figN_*.csv for the rest).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from common import (CORRIDOR_NAMES, DAY_END_HOUR, DAY_START_HOUR, DIRECTION_LABEL, FIGURES, INNER,
                    PASS_THRESHOLD_SEC, PEAK_WINDOWS, CLEANED, PROCESSED)

# Palette: validated categorical slots 1-2 for the two directions, neutral inks, one-hue blue ramp.
DIR_COLOR = {0: "#2a78d6", 1: "#eb6834"}
DIR_NAME = {0: "West", 1: "East"}
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#a3a29c", "#e6e5e0", "#fcfcfb"
PEAK_FILL = "#cde2fb"
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
FIG_DATA = PROCESSED / "figure_data"
WINDOW = f"{DAY_START_HOUR}:00–{DAY_END_HOUR}:00"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelcolor": INK_2, "axes.edgecolor": MUTED, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def in_window(sec):
    hour = sec // 3600
    return (hour >= DAY_START_HOUR) & (hour < DAY_END_HOUR)


def save(fig, name, table, table_path=None):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name)
    plt.close(fig)
    path = table_path or FIG_DATA / name.replace(".png", ".csv")
    path.parent.mkdir(exist_ok=True)
    table.to_csv(path, index=False)
    print("wrote", f"figures/{name}", "+", path.relative_to(PROCESSED.parent))


def rate_table(df, d, bins_col, cats):
    """Express % per category = express departures / departures, as a tidy table."""
    agg = df.groupby(pd.Categorical(bins_col, cats), observed=False).goes_express.agg(["sum", "size"])
    return pd.DataFrame({"direction": DIR_NAME[d], "bin": cats, "departures": agg["size"].values,
                         "express_departures": agg["sum"].values,
                         "express_pct": agg["sum"].values / agg["size"].values * 100})


def bar_pair(ax, tables, cats):
    """Side-by-side West/East bars."""
    width = 0.38
    for i, (d, t) in enumerate(tables.items()):
        x = np.arange(len(cats)) + (i - 0.5) * width
        ax.bar(x, t.express_pct, width * 0.92, color=DIR_COLOR[d], label=DIRECTION_LABEL[d].split(" (")[0])
    ax.set_xticks(range(len(cats)), cats, fontsize=9.5)


def minute_bins(values, edges, labels):
    return pd.cut(values, edges, labels=labels, right=False).astype(object)


def stop_bins(stops, top):
    """Stop counts as text categories, with everything >= top in one 'top+' bin."""
    return stops.map(lambda v: np.nan if pd.isna(v) else (f"{top}+" if v >= top else str(int(v))))


def fig1_dwell(ev):
    """Dwell times form two groups: pass-throughs (10-20 s) and real stops (30-60 s)."""
    ev = ev[in_window(ev.dep_sec)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    bins = np.arange(0, 151, 2.5)
    rows, n = [], 0
    for ax, d in zip(axes, (0, 1)):
        x = ev[(ev.direction_id == d) & ev.stop_id.isin(INNER[d])].dwell.dropna()
        n += len(x)
        w = np.full(len(x), 100 / len(x))
        h, _, _ = ax.hist(x.clip(upper=150), bins=bins, weights=w, color=DIR_COLOR[d], rwidth=0.85)
        rows.append(pd.DataFrame({"DwellTime": bins[:-1], "Share of stop visits": h.round(3),
                                  "Direction": DIR_NAME[d]}))
        ax.axvline(PASS_THRESHOLD_SEC, color=INK, lw=1.2, ls="--")
        share = (x < PASS_THRESHOLD_SEC).mean() * 100
        top = ax.get_ylim()[1]
        ax.text(1, top * 0.5, f"rolled\nthrough\n< {PASS_THRESHOLD_SEC} s\n({share:.0f}%)", ha="left", color=INK, fontsize=10)
        ax.text(75, top * 0.62, f"stopped, doors opened\n({100 - share:.0f}%)", ha="left", color=INK, fontsize=10)
        ax.set_title(DIRECTION_LABEL[d])
        ax.set_xlabel("Dwell at stop = departure − arrival (seconds)")
        ax.set_xlim(0, 150)
    axes[0].set_ylabel("Share of stop visits (%)")
    fig.suptitle(f"Dwell times at the 6 intermediate stops split into two groups  ({WINDOW}, n = {n:,} visits)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.03)
    save(fig, "fig1_dwell_histogram.png", pd.concat(rows), PROCESSED / "dwellTime.csv")


def neighbor_figure(cand, side, name):
    """Express rate by minutes and by stops to the train in front / behind, at the current stop."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
    if side == "front":
        edges = [0, 2, 4, 6, 8, 10, 12, 15, 20, 30, np.inf]
        mlabels = ["0–2", "2–4", "4–6", "6–8", "8–10", "10–12", "12–15", "15–20", "20–30", "30+"]
        top = 11
        scats = [str(i) for i in range(top)] + [f"{top}+"]
        xlab_m = "Minutes since the train in front departed the current stop"
        xlab_s = "Stops the train in front is ahead of the current stop"
        title = "Train in front"
    else:
        edges = [-np.inf, 0, 1, 2, 3, 4, 5, 7, 10, 15, np.inf]
        mlabels = ["<0", "0–1", "1–2", "2–3", "3–4", "4–5", "5–7", "7–10", "10–15", "15+"]
        top = 7
        scats = [str(i) for i in range(1, top)] + [f"{top}+"]
        xlab_m = "Minutes until the train behind arrives at the current stop\n(<0 = it was already there)"
        xlab_s = "Stops the train behind is behind the current stop"
        title = "Train behind"
    tables = []
    for ax, kind in zip(axes, ("minutes", "stops")):
        per_dir = {}
        for d in (0, 1):
            s = cand[cand.direction_id == d]
            if kind == "minutes":
                b, cats = minute_bins(s[f"{side}_minutes"], edges, mlabels), mlabels
            else:
                b, cats = stop_bins(s[f"{side}_stops"], top), scats
            per_dir[d] = rate_table(s, d, b, cats)
            tables.append(per_dir[d].assign(measure=f"{side}_{kind}"))
        bar_pair(ax, per_dir, cats)
        ax.set_xlabel(xlab_m if kind == "minutes" else xlab_s)
        ax.set_title(f"{title}: minutes" if kind == "minutes" else f"{title}: stops")
    axes[0].set_ylabel("Trains that go express\nfrom the current stop (%)")
    axes[0].legend(loc="upper right" if side == "behind" else "upper left")
    fig.suptitle(f"% of trains that go express from the current stop, by the {title.lower()}  "
                 f"({WINDOW})",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.03)
    table = pd.concat(tables)[["measure", "direction", "bin", "departures", "express_departures",
                               "express_pct"]].round(3)
    save(fig, name, table)


def fig4_gap_grid(cand):
    """Both neighbors together: express happens when the front train is far AND the one behind is close."""
    fe = [0, 4, 6, 8, 10, 12, 15, 20, np.inf]
    fl = ["0–4", "4–6", "6–8", "8–10", "10–12", "12–15", "15–20", "20+"]
    be = [-np.inf, 0, 2, 4, 6, 8, 10, 12, 15, np.inf]
    bl = ["<0", "0–2", "2–4", "4–6", "6–8", "8–10", "10–12", "12–15", "15+"]
    f = pd.Categorical(minute_bins(cand.front_minutes, fe, fl), fl)
    b = pd.Categorical(minute_bins(cand.behind_minutes, be, bl), bl)
    agg = cand.groupby([b, f], observed=False).goes_express.agg(["mean", "size", "sum"])
    rate = agg["mean"].unstack().where(agg["size"].unstack() >= 30) * 100
    fig, ax = plt.subplots(figsize=(10, 7))
    vmax = np.nanmax(rate.values)
    im = ax.imshow(rate.values, origin="lower", cmap=BLUE_RAMP, vmin=0, vmax=vmax, aspect="auto")
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            v = rate.values[i, j]
            txt = "·" if np.isnan(v) else (f"{v:.0f}" if v >= 1 else f"{v:.1f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color=MUTED if np.isnan(v) else ("white" if v > vmax * 0.55 else INK))
    ax.set_xticks(range(len(fl)), fl)
    ax.set_yticks(range(len(bl)), bl)
    ax.set_xlabel("Train in front: minutes since it departed the current stop")
    ax.set_ylabel("Train behind: minutes until it arrives at the current stop")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Trains that go express from the current stop (%)")
    cb.outline.set_visible(False)
    ax.set_title(f"Express rate (%) by both neighbors, both directions, {WINDOW}\n"
                 "(cells with < 30 departures shown as ·)", fontsize=12)
    table = agg.reset_index()
    table.columns = ["behind_minutes_bin", "front_minutes_bin", "express_rate", "departures", "express_departures"]
    table["express_pct"] = (table.pop("express_rate") * 100).round(3)
    save(fig, "fig4_front_vs_behind_heatmap.png", table)


def fig5_when(trips):
    """Express rate by hour of day, all hours (justifies the 6-22 window), and weekday x hour."""
    fig = plt.figure(figsize=(14, 9.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.55, wspace=0.12)
    ax = fig.add_subplot(grid[0, :])
    hours = list(range(5, 24)) + [0, 1]
    x = np.arange(len(hours))
    tables = []
    for d in (0, 1):
        s = trips[trips.direction_id == d]
        agg = s.groupby("hour").express.agg(["sum", "size"]).reindex(hours)
        p = agg["sum"] / agg["size"]
        ax.plot(x, p * 100, color=DIR_COLOR[d], lw=2, label=DIRECTION_LABEL[d].split(" (")[0])
        tables.append(pd.DataFrame({"panel": "by_hour", "direction": DIR_NAME[d], "hour": hours,
                                    "weekday": "all", "trips": agg["size"].values,
                                    "express_trips": agg["sum"].values, "express_pct": p.values * 100}))
    ax.axvspan(-0.5, hours.index(DAY_START_HOUR) - 0.5, color=GRID, lw=0, zorder=0)
    ax.axvspan(hours.index(DAY_END_HOUR) - 0.5, len(hours) - 0.5, color=GRID, lw=0, zorder=0)
    for a, b in PEAK_WINDOWS:
        ax.axvspan(hours.index(a) - 0.5, hours.index(b) - 0.5, color=PEAK_FILL, lw=0, zorder=0)
    ax.set_xticks(range(0, len(hours), 2), [f"{h}" for h in hours[::2]])
    ax.set_xlabel("Hour the train enters the corridor (local time)")
    ax.set_ylabel("Trips that go express (%)")
    ax.set_title("By hour, all hours")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [Patch(color=PEAK_FILL, label="11–13 and 16–18"),
                                 Patch(facecolor=GRID, edgecolor=MUTED, label="excluded (before 6, after 22)")],
              loc="upper right", ncol=2, fontsize=9.5)

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    t = trips[trips.in_window]
    aggs = {d: t[t.direction_id == d].groupby(["weekday", "hour"]).express.agg(["mean", "size", "sum"])
            for d in (0, 1)}
    vmax = max(a["mean"].max() for a in aggs.values()) * 100  # one shared scale for both directions
    heat_axes = [fig.add_subplot(grid[1, i]) for i in (0, 1)]
    for ax, d in zip(heat_axes, (0, 1)):
        piv = aggs[d]["mean"].unstack() * 100
        im = ax.imshow(piv.values, cmap=BLUE_RAMP, aspect="auto", vmin=0, vmax=vmax)
        ax.set_yticks(range(7), days if d == 0 else [])
        ax.set_xticks(range(piv.shape[1]), piv.columns, fontsize=9)
        ax.set_xlabel("Hour (local time)")
        ax.grid(False)
        ax.set_title(f"{DIRECTION_LABEL[d].split(' (')[0]}, by weekday × hour ({WINDOW})")
        wk = aggs[d].reset_index()
        tables.append(pd.DataFrame({"panel": "weekday_hour", "direction": DIR_NAME[d], "hour": wk.hour,
                                    "weekday": wk.weekday.map(dict(enumerate(days))), "trips": wk["size"],
                                    "express_trips": wk["sum"], "express_pct": wk["mean"] * 100}))
    cb = fig.colorbar(im, ax=heat_axes, pad=0.02, shrink=0.9)
    cb.set_label("Trips that go express (%)")
    cb.outline.set_visible(False)
    fig.suptitle("When express runs happen: highest early morning and late night, lowest in the afternoon",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", y=0.98)
    save(fig, "fig5_express_by_time.png", pd.concat(tables).round(3))


def fig6_where(trips):
    """Which intermediate stops get skipped on express trips."""
    t = trips[trips.in_window & trips.express]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), sharex=True, gridspec_kw={"wspace": 0.38})
    rows = []
    for ax, d in zip(axes, (0, 1)):
        s = t[t.direction_id == d]
        names = CORRIDOR_NAMES[d][1:]
        pct = s[[f"passed_{i + 1}" for i in range(len(names))]].mean().values * 100
        y = np.arange(len(names))[::-1]
        ax.barh(y, pct, 0.6, color=DIR_COLOR[d])
        for yy, v in zip(y, pct):
            ax.text(v + 1, yy, f"{v:.0f}%", va="center", fontsize=9, color=INK)
        ax.set_yticks(y, names)
        ax.set_xlim(0, 110)
        ax.set_xlabel("Share of express trips that skipped the stop (%)")
        ax.set_title(f"{DIRECTION_LABEL[d]}, n = {len(s):,}")
        ax.grid(axis="y", visible=False)
        rows.append(pd.DataFrame({"direction": DIR_NAME[d], "stop": names, "express_trips": len(s),
                                  "skipped_pct": pct.round(2)}))
    fig.suptitle(f"Where express trains skip, stops in travel order top to bottom ({WINDOW})", x=0.01,
                 ha="left", fontsize=13, fontweight="bold", y=1.03)
    save(fig, "fig6_skipped_stops.png", pd.concat(rows))


def fig7_travel(trips):
    """Sanity check on the label: express trips should be faster through the corridor."""
    t = trips[trips.in_window].dropna(subset=["corridor_travel_sec"])
    fig, ax = plt.subplots(figsize=(9, 4.4))
    data, pos, colors, rows = [], [], [], []
    for i, d in enumerate((0, 1)):
        for j, ex in enumerate((False, True)):
            x = t[(t.direction_id == d) & (t.express == ex)].corridor_travel_sec / 60
            data.append(x)
            pos.append(i * 2.6 + j)
            colors.append(DIR_COLOR[d] if ex else MUTED)
            q = x.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).values
            rows.append({"direction": DIR_NAME[d], "train_type": "express" if ex else "normal", "trips": len(x),
                         **{f"p{k}_minutes": round(v, 2) for k, v in zip((5, 25, 50, 75, 95), q)}})
    bp = ax.boxplot(data, positions=pos, widths=0.7, patch_artist=True, showfliers=False,
                    medianprops={"color": INK, "lw": 1.8}, whiskerprops={"color": INK_2},
                    capprops={"color": INK_2}, boxprops={"lw": 0})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    for p, x in zip(pos, data):
        ax.text(p + 0.37, x.median(), f" {x.median():.1f}", va="center", ha="left", fontsize=9, color=INK)
    ax.set_xticks(pos, ["normal", "express"] * 2)
    for i, d in enumerate((0, 1)):
        ax.text(i * 2.6 + 0.5, -0.17, DIRECTION_LABEL[d], transform=ax.get_xaxis_transform(), ha="center",
                fontsize=10, color=INK_2)
    ax.set_ylabel("Kenmore ↔ Harvard Ave travel time (min)")
    ax.set_title(f"Label check: express trips are faster through the corridor ({WINDOW})\n"
                 "(box = middle 50%, whiskers = 1.5 IQR, outliers hidden)", fontsize=12)
    ax.grid(axis="x", visible=False)
    save(fig, "fig7_travel_time_label_check.png", pd.DataFrame(rows))


def main():
    for old in ("fig2_marey_bunching_example.png", "fig3_express_vs_follower_gap.png",
                "fig4_gap_ahead_vs_behind_heatmap.png"):
        (FIGURES / old).unlink(missing_ok=True)
    ev = pd.read_parquet(CLEANED / "stop_events.parquet")
    trips = pd.read_parquet(CLEANED / "trips.parquet")
    cand = pd.read_parquet(CLEANED / "current_stops.parquet")
    fig1_dwell(ev)
    neighbor_figure(cand, "front", "fig2_train_in_front.png")
    neighbor_figure(cand, "behind", "fig3_train_behind.png")
    fig4_gap_grid(cand)
    fig5_when(trips)
    fig6_where(trips)
    fig7_travel(trips)


if __name__ == "__main__":
    main()
