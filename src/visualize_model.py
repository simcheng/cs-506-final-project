"""Step 6: figures for the November check-in (model performance and interpretation).

Reads model.py's outputs and writes figures/fig8-fig12 + the exact plotted numbers in
data_processed/figure_data/. All performance numbers are on the test months (Aug-Sep 2026),
which no model saw while being fit or tuned.
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import precision_recall_curve

from common import CLEANED, PROCESSED
from model import ALL_FEATURES, design
from visualize import BLUE_RAMP, GRID, INK, INK_2, MUTED, SURFACE, minute_bins, save

MODEL_DIR = CLEANED / "model"
# Categorical slots 1-4 of the validated palette, fixed per model.
MODEL_COLOR = {"boosting": "#2a78d6", "mlp": "#eb6834", "tree": "#1baf7a", "logistic": "#eda100"}
MODEL_NAME = {"boosting": "Gradient boosting", "mlp": "Neural network (MLP)", "tree": "Decision tree",
              "logistic": "Logistic regression"}
ML_MODELS = list(MODEL_COLOR)
POS, NEG = "#e34948", "#2a78d6"  # diverging pair: raises / lowers the odds of express
FEATURE_NAME = {
    "front_gap_min": "Front gap: min since previous train left",
    "behind_eta_min": "Train behind: ETA to this stop (min)",
    "sched_dev_min": "Minutes behind schedule",
    "delay_alert": "Green-B delay alert active",
    "weekend": "Weekend",
    "stop": "Current stop",
    "hour": "Hour of day",
}
TEST = "test months Aug–Sep 2026"
TREE_NAME = {"front_gap_min": "Front gap (min)", "behind_eta_min": "Behind ETA (min)", "sched_dev_min": "Min late",
             "delay_alert": "Delay alert",
             "weekend": "Weekend", "hour": "Hour"}


def feature_label(term):
    if term.startswith("missingindicator_"):
        return FEATURE_NAME[term.removeprefix("missingindicator_")] + " (no schedule match)"
    if term.startswith("stop_"):
        return "Stop: " + term.removeprefix("stop_")
    return FEATURE_NAME.get(term, term)


def fig8_pr(pred, results):
    """Precision-recall curves (interpolated at 101 recall levels) + average precision per model."""
    test = pred[pred.split == "test"]
    y = test.goes_express.to_numpy()
    grid = np.linspace(0, 1, 101)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.55})
    rows = []
    for name in ML_MODELS:
        prec, rec, _ = precision_recall_curve(y, test[f"p_{name}"])
        # Interpolated precision: best precision reachable at recall >= r (the standard PR curve shape).
        interp = np.array([prec[rec >= r].max() for r in grid])
        ax.plot(grid * 100, interp * 100, color=MODEL_COLOR[name], lw=2, label=MODEL_NAME[name], zorder=3)
        rows.append(pd.DataFrame({"panel": "pr_curve", "model": name, "recall_pct": grid * 100,
                                  "precision_pct": interp * 100}))
    base = y.mean() * 100
    ax.axhline(base, color=INK_2, lw=0.9, label=f"Random guess ({base:.1f}%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Recall: % of express departures caught")
    ax.set_ylabel("Precision: % of flagged trains that went express")
    ax.set_title("Precision–recall curves")
    ax.legend(loc="upper right", fontsize=9.5)

    names = sorted(results["models"], key=lambda n: results["models"][n]["test"]["average_precision"])
    ap = [results["models"][n]["test"]["average_precision"] * 100 for n in names]
    yy = np.arange(len(names))
    bx.barh(yy, ap, 0.62, color=[MODEL_COLOR[n] for n in names])
    for v, k in zip(ap, yy):
        bx.text(v + 0.4, k, f"{v:.1f}", va="center", fontsize=9.5, color=INK)
    bx.axvline(base, color=INK_2, lw=0.9)
    bx.text(base + 0.3, len(names) - 0.55, f"random guess = {base:.1f}", fontsize=9, color=INK_2, va="bottom")
    bx.set_ylim(-0.6, len(names) - 0.1)
    bx.set_yticks(yy, [MODEL_NAME[n] for n in names])
    bx.set_xlabel("Average precision (%) = area under the PR curve")
    bx.set_title("Average precision by model")
    bx.grid(axis="y", visible=False)
    rows.append(pd.DataFrame({"panel": "average_precision", "model": names, "average_precision_pct": ap}))
    fig.suptitle(f"How well each model finds express departures ({TEST}; {int(y.sum())} express in "
                 f"{len(y):,} departures)", x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.02)
    save(fig, "fig8_model_pr_curves.png", pd.concat(rows).round(3))


def fig9_confusion(results):
    """Test-month confusion matrices at the threshold each model chose on July."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), gridspec_kw={"wspace": 0.35})
    rows = []
    for ax, name in zip(axes, ML_MODELS):
        s = results["models"][name]["test"]
        counts = np.array([[s["tn"], s["fp"]], [s["fn"], s["tp"]]])
        pct = counts / counts.sum(axis=1, keepdims=True) * 100
        ax.imshow(pct, cmap=BLUE_RAMP, vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{counts[i, j]:,}\n{pct[i, j]:.1f}%", ha="center", va="center", fontsize=10,
                        color="white" if pct[i, j] > 55 else INK)
                rows.append({"model": name, "true": ["normal", "express"][i], "predicted": ["normal", "express"][j],
                             "departures": int(counts[i, j]), "pct_of_true_class": round(pct[i, j], 2)})
        ax.set_xticks([0, 1], ["normal", "express"])
        ax.set_yticks([0, 1], ["normal", "express"] if name == ML_MODELS[0] else [])
        ax.set_xlabel("Predicted")
        if name == ML_MODELS[0]:
            ax.set_ylabel("Actual (dwell-time label)")
        ax.grid(False)
        ax.set_title(f"{MODEL_NAME[name]}\nprecision {s['precision'] * 100:.0f}%, recall {s['recall'] * 100:.0f}%",
                     fontsize=11)
    rec = [results["models"][n]["test"]["recall"] * 100 for n in ML_MODELS]
    fig.suptitle(f"At the threshold picked on July, every model catches {min(rec):.0f}–{max(rec):.0f}% of express "
                 f"departures ({TEST}; cell % = share of the actual class)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.08)
    save(fig, "fig9_confusion_matrices.png", pd.DataFrame(rows))


def fig10_importance(results):
    imp = pd.read_csv(MODEL_DIR / "permutation_importance.csv")
    coef = pd.read_csv(MODEL_DIR / "logistic_coefficients.csv")
    # Hour terms are left out (16 of them); of the was-missing flags, only the schedule one matters
    # (11% of rows: ADDED trips and trips without a schedule match); the rest are < 0.2% of rows.
    keep_missing = ["missingindicator_sched_dev_min"]
    coef = coef[~coef.term.str.startswith("hour_")
                & (~coef.term.str.startswith("missingindicator_") | coef.term.isin(keep_missing))]
    coef = coef.sort_values("coefficient")
    best = results["best_model_by_val"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 6.4), gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.9})
    imp = imp.sort_values("ap_drop_mean")
    y = np.arange(len(imp))
    ax.barh(y, imp.ap_drop_mean * 100, 0.62, color=MODEL_COLOR[best])
    for v, k in zip(imp.ap_drop_mean * 100, y):
        ax.text(max(v, 0) + 0.2, k, f"{v:.1f}", va="center", fontsize=9, color=INK)
    ax.axvline(0, color=MUTED, lw=0.9)
    ax.set_yticks(y, [FEATURE_NAME[f] for f in imp.feature])
    ax.set_xlabel("Drop in average precision (points) when the feature is shuffled")
    ax.set_title(f"{MODEL_NAME[best]}, the best model on July:\nwhat it relies on", fontsize=12)
    ax.grid(axis="y", visible=False)

    y = np.arange(len(coef))
    bx.barh(y, coef.coefficient, 0.62, color=[POS if v > 0 else NEG for v in coef.coefficient])
    bx.axvline(0, color=MUTED, lw=0.9)
    bx.set_yticks(y, [feature_label(t) for t in coef.term], fontsize=9.5)
    bx.set_xlabel("Coefficient (change in log-odds per 1 SD; stops vs. average)\n"
                  "red = more likely express, blue = less likely")
    bx.set_title("Logistic regression: direction of each effect\n(hour of day terms not shown)", fontsize=12)
    bx.grid(axis="y", visible=False)
    fig.suptitle(f"Which inputs predict an express run ({TEST})", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", y=1.02)
    table = pd.concat([imp.assign(panel="permutation_importance", model=best)[
                           ["panel", "model", "feature", "ap_drop_mean"]].rename(columns={"ap_drop_mean": "value"}),
                       coef.assign(panel="logistic_coefficient", model="logistic")[
                           ["panel", "model", "term", "coefficient"]].rename(
                           columns={"term": "feature", "coefficient": "value"})])
    save(fig, "fig10_feature_importance.png", table.round(4))


def fig11_tree():
    """The top 3 levels of the tuned decision tree (it is 8 deep; deeper nodes are summarised)."""
    # Our own file, written by model.py on this machine (not downloaded), so unpickling it is safe.
    tree = joblib.load(MODEL_DIR / "decision_tree.joblib")
    t, names = tree.tree_, list(tree.feature_names_in_)
    # Raw training departures and express departures through each node (tree_.value is weighted
    # when class_weight="balanced", so count them directly).
    train = pd.read_parquet(CLEANED / "model_table.parquet").query("split == 'train'")
    path = tree.decision_path(design(train, ALL_FEATURES, hour_onehot=False))
    n_node = np.asarray(path.sum(axis=0)).ravel()
    ex_node = np.asarray(path[train.goes_express.to_numpy() == 1].sum(axis=0)).ravel()
    depth_shown = 3
    nodes, pos = [], {}

    def walk(node, depth, x0, x1, parent, edge):
        x = (x0 + x1) / 2
        n = int(n_node[node])
        share = ex_node[node] / n
        leaf = t.children_left[node] == -1 or depth == depth_shown
        f = None if t.children_left[node] == -1 else names[t.feature[node]]
        nodes.append({"node": node, "depth": depth, "parent": parent, "edge_from_parent": edge, "x": x,
                      "feature": None if leaf else f, "threshold": None if leaf else float(t.threshold[node]),
                      "departures": n, "express_pct": share * 100, "summarised": depth == depth_shown
                      and t.children_left[node] != -1})
        pos[node] = (x, -depth)
        if leaf:
            return
        dummy = f.startswith(("stop_", "hour_"))
        # sklearn: left child = feature <= threshold. For a 0/1 stop dummy that means "not this stop".
        walk(t.children_left[node], depth + 1, x0, x, node, "no" if dummy else "yes")
        walk(t.children_right[node], depth + 1, x, x1, node, "yes" if dummy else "no")

    walk(0, 0, 0, 1, None, None)
    fig, ax = plt.subplots(figsize=(17, 7.2))
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-depth_shown - 0.55, 0.6)
    ax.axis("off")
    vmax = max(n["express_pct"] for n in nodes)
    for n in nodes:
        if n["parent"] is not None:
            (x0, y0), (x1, y1) = pos[n["parent"]], pos[n["node"]]
            ax.plot([x0, x1], [y0 - 0.22, y1 + 0.22], color=MUTED, lw=1, zorder=1)
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, n["edge_from_parent"], fontsize=9, color=INK_2, ha="center",
                    va="center", bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none"})
    for n in nodes:
        x, y = pos[n["node"]]
        if n["feature"] is None:
            rule = "… deeper splits" if n["summarised"] else "leaf"
        elif n["feature"].startswith("stop_"):
            rule = f"At {n['feature'].removeprefix('stop_')}?"
        else:
            rule = f"{TREE_NAME[n['feature']]} ≤ {n['threshold']:.1f}?"
        text = f"{rule}\n{n['departures']:,} departures\nexpress {n['express_pct']:.1f}%"
        color = BLUE_RAMP(n["express_pct"] / vmax)
        w = 0.115 if n["depth"] < 3 else 0.105
        ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.22), w, 0.44, boxstyle="round,pad=0.005,rounding_size=0.01",
                                    fc=color, ec="none", zorder=2))
        ax.text(x, y, text, ha="center", va="center", fontsize=8.6, zorder=3,
                color="white" if n["express_pct"] / vmax > 0.55 else INK)
    ax.set_title(f"Decision tree, top {depth_shown} of its 8 levels: training departures (Jan–Jun) through each "
                 f"split; box shade = express % (overall {nodes[0]['express_pct']:.1f}%)", fontsize=13)
    save(fig, "fig11_decision_tree.png", pd.DataFrame(nodes).drop(columns=["x"]).round(3))


def fig12_pred_vs_obs(pred, results):
    """Observed vs predicted express rate on the test months, by the two gap features."""
    best = results["best_model_by_val"]
    test = pred[pred.split == "test"]
    fe = [0, 4, 6, 8, 10, 12, 15, 20, np.inf]
    fl = ["0–4", "4–6", "6–8", "8–10", "10–12", "12–15", "15–20", "20+"]
    be = [-np.inf, 2, 4, 6, 8, 10, 12, 15, np.inf]
    bl = ["<2", "2–4", "4–6", "6–8", "8–10", "10–12", "12–15", "15+"]
    f = pd.Categorical(minute_bins(test.front_gap_min, fe, fl), fl)
    b = pd.Categorical(minute_bins(test.behind_eta_min, be, bl), bl)
    agg = test.groupby([b, f], observed=False).agg(departures=("goes_express", "size"),
                                                   express=("goes_express", "sum"),
                                                   observed=("goes_express", "mean"),
                                                   predicted=(f"p_{best}", "mean"))
    enough = agg.departures.unstack() >= 30
    grids = {"Observed (dwell-time label)": agg.observed.unstack().where(enough) * 100,
             f"Predicted ({MODEL_NAME[best].lower()})": agg.predicted.unstack().where(enough) * 100}
    vmax = max(np.nanmax(g.values) for g in grids.values())
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, (title, g) in zip(axes, grids.items()):
        im = ax.imshow(g.values, origin="lower", cmap=BLUE_RAMP, vmin=0, vmax=vmax, aspect="auto")
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                v = g.values[i, j]
                txt = "·" if np.isnan(v) else (f"{v:.0f}" if v >= 1 else f"{v:.1f}")
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        color=MUTED if np.isnan(v) else ("white" if v > vmax * 0.55 else INK))
        ax.set_xticks(range(len(fl)), fl)
        ax.set_xlabel("Front gap: minutes since the previous train left the stop")
        ax.set_title(title)
        ax.grid(False)
    axes[0].set_yticks(range(len(bl)), bl)
    axes[0].set_ylabel("Train behind: estimated minutes until it reaches the stop")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.015)
    cb.set_label("Express departures (%)")
    cb.outline.set_visible(False)
    fig.suptitle(f"The model reproduces the bunching pattern: express when the gap ahead is long and the train "
                 f"behind is close\n({TEST}; cells with < 30 departures shown as ·)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.03)
    table = agg.reset_index()
    table.columns = ["behind_eta_bin", "front_gap_bin", "departures", "express_departures", "observed", "predicted"]
    table["observed_pct"] = (table.pop("observed") * 100).round(3)
    table["predicted_pct"] = (table.pop("predicted") * 100).round(3)
    table["shown"] = table.departures >= 30
    save(fig, "fig12_predicted_vs_observed.png", table.assign(model=best))


def main():
    results = json.loads((PROCESSED / "model_results.json").read_text())
    pred = pd.read_parquet(MODEL_DIR / "predictions.parquet")
    fig8_pr(pred, results)
    fig9_confusion(results)
    fig10_importance(results)
    fig11_tree()
    fig12_pred_vs_obs(pred, results)


if __name__ == "__main__":
    main()
