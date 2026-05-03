"""
CBSLAO ablation studies.

A1. Sensitivity to CBUC confidence-width parameters α, β, γ.
A2. Behaviour under utility-dense distractors.
A3. Calibrated benchmark-trace replay (NOT a true replay — network
    restricted; workloads parameterized to published summary statistics
    of ToolBench, τ-bench, BFCL).

Each study writes a CSV to out/ and a PNG to plots/.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reuse simulator
sys.path.insert(0, os.path.dirname(__file__))
from cbslao_sim import (  # noqa: E402
    Registry, Task, Tool, make_registry, make_task,
    run_policy_react_capped, run_policy_bwk_vanilla,
    run_policy_cbuc, oracle_nonadaptive, saturate_utility,
)


# ============================================================
# A1. α/β/γ sensitivity
# ============================================================

def ablation_abg(out_csv: str, n_seeds: int = 40):
    """One-at-a-time sweep with defaults (α=1.0, β=1.5, γ=1.5)."""
    default = dict(alpha=1.0, beta=1.5, gamma=1.5)
    grid = {
        "alpha": [0.25, 0.5, 1.0, 1.5, 2.0, 3.0],
        "beta":  [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "gamma": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    }
    # fixed environment
    K = 50; rho = 0.5; tail = 2.0; sparsity = 0.2
    rows = []
    for param, values in grid.items():
        for v in values:
            kwargs = dict(default); kwargs[param] = v
            for s in range(n_seeds):
                seed = hash((param, round(v*100), s)) & 0xffffffff
                master = np.random.default_rng(seed)
                registry = make_registry(K, tail, sparsity, master)
                task = make_task(registry, rho, master)
                sub = np.random.default_rng(seed * 7 + 1)
                r = run_policy_cbuc(registry, task, sub,
                                     alpha=kwargs["alpha"],
                                     beta=kwargs["beta"],
                                     gamma=kwargs["gamma"],
                                     z_level=1.2816,
                                     variant_name="cbuc")
                rows.append(dict(
                    param=param, value=v, seed=s,
                    utility=r.utility, cost=r.cost,
                    budget=r.budget, deadline=r.deadline,
                    n_calls=r.n_calls,
                    budget_violated=int(r.budget_violated),
                    deadline_violated=int(r.deadline_violated),
                ))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"A1 wrote {out_csv}  ({len(rows)} rows)")


def plot_abg(csv_path: str, out_png: str):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            r["value"] = float(r["value"])
            r["utility"] = float(r["utility"])
            r["budget_violated"] = int(r["budget_violated"])
            rows.append(r)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["param"], r["value"])].append(r)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), sharey=False)
    titles = {"alpha": r"α (utility bonus)",
              "beta":  r"β (cost/deadline margin)",
              "gamma": r"γ (cost lower bound)"}
    for ax, param in zip(axes, ["alpha", "beta", "gamma"]):
        vals = sorted({k[1] for k in groups if k[0] == param})
        o_mean = [np.mean([r["budget_violated"] for r in groups[(param, v)]]) for v in vals]
        u_mean = [np.mean([r["utility"] for r in groups[(param, v)]]) for v in vals]
        ax2 = ax.twinx()
        ax.plot(vals, o_mean, marker="o", color="#c0392b",
                label="Budget-overrun")
        ax2.plot(vals, u_mean, marker="s", color="#27ae60",
                 label="Utility")
        ax.set_title(titles[param])
        ax.set_xlabel("parameter value")
        ax.set_ylabel("overrun rate", color="#c0392b")
        ax2.set_ylabel("utility", color="#27ae60")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("CBUC sensitivity to α, β, γ (K=50, ρ=0.5, tail=2.0, n=40)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"A1 plot  -> {out_png}")


# ============================================================
# A2. Utility-dense distractors
# ============================================================

def ablation_distractors(out_csv: str, n_seeds: int = 30):
    sparsities = [0.1, 0.25, 0.5, 0.75, 1.0]
    K = 50; rho = 0.5; tail = 2.0
    rows = []
    for sp in sparsities:
        for s in range(n_seeds):
            seed = hash(("sparsity", round(sp*100), s)) & 0xffffffff
            master = np.random.default_rng(seed)
            registry = make_registry(K, tail, sp, master)
            task = make_task(registry, rho, master)
            oracle_u = oracle_nonadaptive(registry.tools, task)
            def sub(x): return np.random.default_rng(seed * 13 + x)
            results = [
                ("react_capped",
                    run_policy_react_capped(registry.tools, task, sub(1))),
                ("bwk_vanilla",
                    run_policy_bwk_vanilla(registry.tools, task, sub(2))),
                ("cbuc",
                    run_policy_cbuc(registry, task, sub(3),
                                    z_level=1.2816, variant_name="cbuc")),
                ("cbuc_aggr",
                    run_policy_cbuc(registry, task, sub(4),
                                    z_level=0.0, variant_name="cbuc_aggr")),
            ]
            for name, r in results:
                rows.append(dict(
                    sparsity=sp, seed=s, policy=name,
                    utility=r.utility, cost=r.cost,
                    budget=r.budget,
                    budget_violated=int(r.budget_violated),
                    oracle_utility=oracle_u,
                    regret=oracle_u - r.utility,
                ))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"A2 wrote {out_csv}  ({len(rows)} rows)")


def plot_distractors(csv_path: str, out_png: str):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            r["sparsity"] = float(r["sparsity"])
            for k in ("utility", "oracle_utility", "regret"):
                r[k] = float(r[k])
            r["budget_violated"] = int(r["budget_violated"])
            rows.append(r)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    colors = {"react_capped": "#d2691e", "bwk_vanilla": "#4682b4",
              "cbuc": "#2e8b57", "cbuc_aggr": "#90ee90"}
    labels = {"react_capped": "ReAct (capped)", "bwk_vanilla": "BwK (vanilla)",
              "cbuc": "CBUC (z=1.28)", "cbuc_aggr": "CBUC-aggr (z=0)"}
    sparsities = sorted({r["sparsity"] for r in rows})
    # Utility vs sparsity
    for pol in colors:
        ys = []; es = []
        for sp in sparsities:
            vals = [r["utility"] for r in rows
                    if r["policy"] == pol and abs(r["sparsity"] - sp) < 1e-6]
            ys.append(np.mean(vals))
            es.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        axes[0].errorbar(sparsities, ys, yerr=es, marker="o",
                         color=colors[pol], label=labels[pol])
    axes[0].set_xlabel("Utility sparsity (fraction of tools with non-zero utility)")
    axes[0].set_ylabel("Utility achieved")
    axes[0].set_title("Utility vs. distractor density")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    # Budget-overrun vs sparsity
    for pol in colors:
        ys = []
        for sp in sparsities:
            vals = [r["budget_violated"] for r in rows
                    if r["policy"] == pol and abs(r["sparsity"] - sp) < 1e-6]
            ys.append(np.mean(vals))
        axes[1].plot(sparsities, ys, marker="o", color=colors[pol], label=labels[pol])
    axes[1].set_xlabel("Utility sparsity")
    axes[1].set_ylabel("Budget-overrun rate")
    axes[1].set_title("Safety vs. distractor density")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Effect of distractor density (K=50, ρ=0.5, tail=2.0, n=30)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"A2 plot  -> {out_png}")


# ============================================================
# A3. Calibrated "replay" — parameterized to public benchmark stats
# ============================================================
# Published summary statistics used (verify against current benchmark
# documentation before camera-ready):
#   - ToolBench: ~16k APIs total; tasks typically touch 1-8 tools;
#     per-call tokens have heavy-tailed distribution.
#   - τ-bench: 2 domains, ~10-15 tools each; tasks involve 3-7 calls.
#   - BFCL: function-calling with ~40 tools per eval subset.
# We map these to three calibration profiles below.
CALIBRATION = [
    dict(name="toolbench_subset", K=100, tools_per_task_expected=6,
         cost_tail=1.8, utility_sparsity=0.08,
         budget_mult=0.4,
         comment="ToolBench-like: large pool, heavy tail, sparse relevance"),
    dict(name="tau_bench",        K=15,  tools_per_task_expected=5,
         cost_tail=2.5, utility_sparsity=0.4,
         budget_mult=0.6,
         comment="τ-bench-like: small pool, moderate tail, denser relevance"),
    dict(name="bfcl_subset",      K=40,  tools_per_task_expected=2,
         cost_tail=2.2, utility_sparsity=0.15,
         budget_mult=0.5,
         comment="BFCL-like: mid-size pool, single-function-call tasks"),
]


def ablation_replay(out_csv: str, n_seeds: int = 50):
    rows = []
    for prof in CALIBRATION:
        for s in range(n_seeds):
            seed = hash(("replay", prof["name"], s)) & 0xffffffff
            master = np.random.default_rng(seed)
            registry = make_registry(prof["K"], prof["cost_tail"],
                                     prof["utility_sparsity"], master)
            # budget calibrated to expected tools-per-task × mean cost
            mean_cost = float(np.mean([t.mean_cost for t in registry.tools]))
            budget = (prof["tools_per_task_expected"] * mean_cost
                      / max(prof["budget_mult"], 0.01))
            # Override task
            deadline = 10.0 * max(t.mean_latency for t in registry.tools)
            task = Task(frozenset([0, 1]), budget, deadline, 0.1)
            oracle_u = oracle_nonadaptive(registry.tools, task)
            def sub(x): return np.random.default_rng(seed * 11 + x)
            results = [
                ("react_capped",
                    run_policy_react_capped(registry.tools, task, sub(1))),
                ("bwk_vanilla",
                    run_policy_bwk_vanilla(registry.tools, task, sub(2))),
                ("cbuc",
                    run_policy_cbuc(registry, task, sub(3),
                                    z_level=1.2816, variant_name="cbuc")),
                ("cbuc_aggr",
                    run_policy_cbuc(registry, task, sub(4),
                                    z_level=0.0, variant_name="cbuc_aggr")),
            ]
            for name, r in results:
                rows.append(dict(
                    profile=prof["name"], K=prof["K"],
                    seed=s, policy=name,
                    utility=r.utility, cost=r.cost,
                    budget=r.budget, n_calls=r.n_calls,
                    budget_violated=int(r.budget_violated),
                    deadline_violated=int(r.deadline_violated),
                    oracle_utility=oracle_u,
                    regret=oracle_u - r.utility,
                ))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"A3 wrote {out_csv}  ({len(rows)} rows)")


def plot_replay(csv_path: str, out_png: str):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            for k in ("utility", "regret", "cost", "oracle_utility"):
                r[k] = float(r[k])
            r["budget_violated"] = int(r["budget_violated"])
            rows.append(r)
    profiles = ["toolbench_subset", "tau_bench", "bfcl_subset"]
    policies = ["react_capped", "bwk_vanilla", "cbuc_aggr", "cbuc"]
    labels = {"react_capped": "ReAct", "bwk_vanilla": "BwK",
              "cbuc_aggr": "CBUC-aggr", "cbuc": "CBUC"}
    colors = {"react_capped": "#d2691e", "bwk_vanilla": "#4682b4",
              "cbuc_aggr": "#90ee90", "cbuc": "#2e8b57"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(profiles))
    w = 0.2
    # Left: budget-overrun
    for i, pol in enumerate(policies):
        ys = []
        for prof in profiles:
            vals = [r["budget_violated"] for r in rows
                    if r["policy"] == pol and r["profile"] == prof]
            ys.append(np.mean(vals))
        axes[0].bar(x + (i - 1.5) * w, ys, w, color=colors[pol],
                    label=labels[pol])
    axes[0].set_xticks(x); axes[0].set_xticklabels(profiles, rotation=15)
    axes[0].set_ylabel("Budget-overrun rate")
    axes[0].set_title("Overrun rate by profile")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")
    # Right: utility
    for i, pol in enumerate(policies):
        ys = []
        for prof in profiles:
            vals = [r["utility"] for r in rows
                    if r["policy"] == pol and r["profile"] == prof]
            ys.append(np.mean(vals))
        axes[1].bar(x + (i - 1.5) * w, ys, w, color=colors[pol],
                    label=labels[pol])
    axes[1].set_xticks(x); axes[1].set_xticklabels(profiles, rotation=15)
    axes[1].set_ylabel("Utility achieved")
    axes[1].set_title("Utility by profile")
    axes[1].grid(True, alpha=0.3, axis="y")
    fig.suptitle("Calibrated replay (NOT a live-trace replay; see §caveat)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"A3 plot  -> {out_png}")


def summary_tables(out_dir: str):
    """Write markdown summary tables for each ablation."""
    # A1
    path = os.path.join(out_dir, "ablation_abg.csv")
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["value"] = float(r["value"]); r["utility"] = float(r["utility"])
        r["budget_violated"] = int(r["budget_violated"])
    groups = defaultdict(list)
    for r in rows: groups[(r["param"], r["value"])].append(r)
    lines = ["| Param | Value | Overrun | Utility |", "|---|---|---|---|"]
    for (p, v), g in sorted(groups.items()):
        o = np.mean([r["budget_violated"] for r in g])
        u = np.mean([r["utility"] for r in g])
        lines.append(f"| {p} | {v} | {o:.3f} | {u:.3f} |")
    with open(os.path.join(out_dir, "ablation_abg_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # A2
    path = os.path.join(out_dir, "ablation_distractors.csv")
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["sparsity"] = float(r["sparsity"]); r["utility"] = float(r["utility"])
        r["budget_violated"] = int(r["budget_violated"])
    groups = defaultdict(list)
    for r in rows: groups[(r["policy"], r["sparsity"])].append(r)
    lines = ["| Policy | Sparsity | Overrun | Utility |", "|---|---|---|---|"]
    for (p, s), g in sorted(groups.items()):
        o = np.mean([r["budget_violated"] for r in g])
        u = np.mean([r["utility"] for r in g])
        lines.append(f"| {p} | {s} | {o:.3f} | {u:.3f} |")
    with open(os.path.join(out_dir, "ablation_distractors_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # A3
    path = os.path.join(out_dir, "ablation_replay.csv")
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["utility"] = float(r["utility"]); r["regret"] = float(r["regret"])
        r["budget_violated"] = int(r["budget_violated"])
    groups = defaultdict(list)
    for r in rows: groups[(r["profile"], r["policy"])].append(r)
    lines = ["| Profile | Policy | Overrun | Utility | Regret |",
             "|---|---|---|---|---|"]
    for (prof, pol), g in sorted(groups.items()):
        o = np.mean([r["budget_violated"] for r in g])
        u = np.mean([r["utility"] for r in g])
        rg = np.mean([r["regret"] for r in g])
        lines.append(f"| {prof} | {pol} | {o:.3f} | {u:.3f} | {rg:+.3f} |")
    with open(os.path.join(out_dir, "ablation_replay_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("summary tables written")


def main(out_dir: str = "out", plot_dir: str = "plots"):
    os.makedirs(out_dir, exist_ok=True); os.makedirs(plot_dir, exist_ok=True)
    print("== A1: α/β/γ sensitivity ==")
    ablation_abg(os.path.join(out_dir, "ablation_abg.csv"), n_seeds=40)
    plot_abg(os.path.join(out_dir, "ablation_abg.csv"),
             os.path.join(plot_dir, "ablation_abg.png"))
    print("== A2: utility-dense distractors ==")
    ablation_distractors(os.path.join(out_dir, "ablation_distractors.csv"),
                         n_seeds=30)
    plot_distractors(os.path.join(out_dir, "ablation_distractors.csv"),
                     os.path.join(plot_dir, "ablation_distractors.png"))
    print("== A3: calibrated benchmark-trace replay ==")
    ablation_replay(os.path.join(out_dir, "ablation_replay.csv"), n_seeds=50)
    plot_replay(os.path.join(out_dir, "ablation_replay.csv"),
                os.path.join(plot_dir, "ablation_replay.png"))
    summary_tables(out_dir)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "out"
    plt_out = sys.argv[2] if len(sys.argv) > 2 else "plots"
    main(out, plt_out)
