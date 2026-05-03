"""Aggregate CBSLAO results and produce publication-ready plots + summary table."""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    plt = None
    HAS_MATPLOTLIB = False


POLICY_ORDER = [
    "react_uncapped",
    "react_capped",
    "nonadaptive_greedy",
    "bwk_vanilla",
    "lagrangian",
    "cvar_bwk",
    "precheck_ucb",
    "cc_knapsack",
    "cbuc_aggr",
    "cbuc_mod",
    "cbuc",
]
POLICY_LABEL = {
    "react_uncapped": "ReAct (uncapped)",
    "react_capped": "ReAct (capped)",
    "nonadaptive_greedy": "Non-adaptive greedy (informed oracle)",
    "bwk_vanilla": "BwK (vanilla)",
    "lagrangian": "Lagrangian primal-dual",
    "cvar_bwk": "CVaR-BwK",
    "precheck_ucb": "Pre-check UCB",
    "cc_knapsack": "CC-knapsack oracle",
    "cbuc_aggr": "CBUC-aggr (z=0)",
    "cbuc_mod": "CBUC-mod (z=0.5)",
    "cbuc": "CBUC (ours, z=1.28)",
}
COLORS = {
    "react_uncapped": "#a0522d",
    "react_capped": "#d2691e",
    "nonadaptive_greedy": "#708090",
    "bwk_vanilla": "#4682b4",
    "lagrangian": "#6a5acd",
    "cvar_bwk": "#8a2be2",
    "precheck_ucb": "#20b2aa",
    "cc_knapsack": "#2f4f4f",
    "cbuc_aggr": "#90ee90",
    "cbuc_mod": "#3cb371",
    "cbuc": "#2e8b57",
}


def load(csv_path):
    rows = []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            for k in ("K",):
                row[k] = int(row[k])
            for k in ("rho", "cost_tail", "utility_sparsity",
                      "utility", "cost", "latency", "budget", "deadline",
                      "oracle_utility", "regret", "n_calls"):
                row[k] = float(row[k])
            for k in ("budget_violated", "deadline_violated"):
                row[k] = int(row[k])
            rows.append(row)
    return rows


def aggregate(rows):
    """Group by (policy, K, rho, cost_tail) and compute means / CIs."""
    groups = defaultdict(list)
    for r in rows:
        key = (r["policy"], r["K"], r["rho"], r["cost_tail"])
        groups[key].append(r)
    summary = {}
    for key, grp in groups.items():
        u = np.array([r["utility"] for r in grp])
        b = np.array([r["budget_violated"] for r in grp])
        d = np.array([r["deadline_violated"] for r in grp])
        reg = np.array([r["regret"] for r in grp])
        cost = np.array([r["cost"] for r in grp])
        summary[key] = dict(
            n=len(grp),
            u_mean=u.mean(), u_se=u.std(ddof=1) / max(1, np.sqrt(len(grp))),
            b_rate=b.mean(), b_se=np.sqrt(b.mean() * (1 - b.mean()) / max(1, len(grp))),
            d_rate=d.mean(), d_se=np.sqrt(d.mean() * (1 - d.mean()) / max(1, len(grp))),
            regret=reg.mean(), regret_se=reg.std(ddof=1) / max(1, np.sqrt(len(grp))),
            cost_mean=cost.mean(),
        )
    return summary


def plot_violation_vs_rho(summary, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    Ks = sorted({k[1] for k in summary})
    rhos = sorted({k[2] for k in summary})
    tails = sorted({k[3] for k in summary})
    for ax, tail in zip(axes, tails):
        for pol in POLICY_ORDER:
            ys = []
            for rho in rhos:
                vals = [summary[(pol, K, rho, tail)]["b_rate"]
                        for K in Ks if (pol, K, rho, tail) in summary]
                ys.append(np.mean(vals) if vals else np.nan)
            if all(np.isnan(y) for y in ys):
                continue
            ax.plot(rhos, ys, marker="o", label=POLICY_LABEL[pol], color=COLORS[pol])
        ax.set_title(f"Pareto tail α = {tail}")
        ax.set_xlabel(r"Budget tightness $\rho = B / \Sigma \mathbb{E}[c]$")
        ax.set_ylabel("Budget-overrun rate")
        ax.grid(True, alpha=0.3)
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Budget-overrun rate vs. budget tightness (avg over K)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_utility_vs_rho(summary, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    Ks = sorted({k[1] for k in summary})
    rhos = sorted({k[2] for k in summary})
    tails = sorted({k[3] for k in summary})
    for ax, tail in zip(axes, tails):
        for pol in POLICY_ORDER:
            ys = []
            for rho in rhos:
                vals = [summary[(pol, K, rho, tail)]["u_mean"]
                        for K in Ks if (pol, K, rho, tail) in summary]
                ys.append(np.mean(vals) if vals else np.nan)
            if all(np.isnan(y) for y in ys):
                continue
            ax.plot(rhos, ys, marker="s", label=POLICY_LABEL[pol], color=COLORS[pol])
        ax.set_title(f"Pareto tail α = {tail}")
        ax.set_xlabel(r"Budget tightness $\rho$")
        ax.set_ylabel("Utility achieved")
        ax.grid(True, alpha=0.3)
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Utility vs. budget tightness (avg over K)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_scaling_K(summary, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    Ks = sorted({k[1] for k in summary})
    rho = 0.5
    tail = 2.5
    for pol in POLICY_ORDER:
        ys = []
        es = []
        for K in Ks:
            key = (pol, K, rho, tail)
            if key in summary:
                ys.append(summary[key]["b_rate"])
                es.append(summary[key]["b_se"])
            else:
                ys.append(np.nan); es.append(0.0)
        if all(np.isnan(y) for y in ys):
            continue
        ax.errorbar(Ks, ys, yerr=es, marker="o", label=POLICY_LABEL[pol], color=COLORS[pol])
    ax.set_xscale("log")
    ax.set_xlabel("Registry size K (log scale)")
    ax.set_ylabel("Budget-overrun rate")
    ax.set_title(f"Scaling with K (ρ={rho}, tail α={tail})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def ci95(values, binomial=False):
    arr = np.array(values, dtype=float)
    mean = arr.mean()
    if len(arr) <= 1:
        return mean, 0.0
    if binomial:
        se = np.sqrt(mean * (1 - mean) / len(arr))
    else:
        se = arr.std(ddof=1) / np.sqrt(len(arr))
    return mean, 1.96 * se


def make_table(rows, out_path):
    """One row per policy, aggregated across raw replicate rows."""
    by_pol = defaultdict(list)
    for row in rows:
        by_pol[row["policy"]].append(row)
    lines = ["| Policy | Budget-overrun | Deadline-miss | Utility | Regret vs. oracle |",
             "|---|---:|---:|---:|---:|"]
    for pol in POLICY_ORDER:
        grp = by_pol[pol]
        if not grp:
            continue
        b, b_ci = ci95([r["budget_violated"] for r in grp], binomial=True)
        d, d_ci = ci95([r["deadline_violated"] for r in grp], binomial=True)
        u, u_ci = ci95([r["utility"] for r in grp])
        reg, reg_ci = ci95([r["regret"] for r in grp])
        lines.append(
            f"| {POLICY_LABEL[pol]} | {b:.3f} (+/- {b_ci:.3f}) | "
            f"{d:.3f} (+/- {d_ci:.3f}) | {u:.3f} (+/- {u_ci:.3f}) | "
            f"{reg:+.3f} (+/- {reg_ci:.3f}) |"
        )
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "out/results_stronger.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "plots"
    os.makedirs(out_dir, exist_ok=True)
    rows = load(csv_path)
    summary = aggregate(rows)
    if HAS_MATPLOTLIB:
        plot_violation_vs_rho(summary, os.path.join(out_dir, "budget_overrun_vs_rho.png"))
        plot_utility_vs_rho(summary, os.path.join(out_dir, "utility_vs_rho.png"))
        plot_scaling_K(summary, os.path.join(out_dir, "scaling_K.png"))
    else:
        print("matplotlib is not installed; skipping PNG plot generation")
    make_table(rows, os.path.join(out_dir, "summary_table.md"))
    print(f"\noutputs written to {out_dir}")


if __name__ == "__main__":
    main()
