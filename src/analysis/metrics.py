"""
Analysis tools for scheming evaluation experiments.

Computes key metrics from the paper:
- Covert action rates (per model, per environment, aggregate)
- Bayesian credible intervals (Beta-Binomial, following the paper)
- CoT analysis statistics (eval awareness rate, spec citation rate)
- Cross-experiment comparisons

Generates visualizations matching the paper's style:
- Bar charts with credible intervals (like Figure 9)
- Density plots for posterior distributions
- Heatmaps for model × environment comparisons
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Optional imports with fallbacks
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams.update({'font.size': 11, 'figure.figsize': (12, 6)})
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_results(results_path: str) -> list[dict]:
    """Load results from JSONL file."""
    results = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def results_to_dataframe(results: list[dict]) -> "pd.DataFrame":
    """Convert results to a pandas DataFrame for analysis."""
    if not HAS_PANDAS:
        raise ImportError("pandas required for DataFrame analysis. Install with: pip install pandas")
    
    rows = []
    for r in results:
        if "error" in r:
            continue  # Skip error records
        
        response = r["response"]
        for monitor_id, score in r["scores"].items():
            rows.append({
                "experiment_id": r["experiment_id"],
                "run_index": r["run_index"],
                "response_id": response["response_id"],
                "task_id": response["task_id"],
                "task_family": response["task_family"],
                "environment_id": response["environment_id"],
                "model_id": response["model_id"],
                "monitor_id": monitor_id,
                "label": score["label"],
                "confidence": score["confidence"],
                "has_cot": response.get("chain_of_thought") is not None,
                "cot_length": len(response.get("chain_of_thought", "") or ""),
                "output_length": len(response.get("final_output", "")),
                "timestamp": response.get("timestamp", ""),
                **{f"metric_{k}": v for k, v in score.get("metrics", {}).items()},
            })
    
    return pd.DataFrame(rows)


# ──────────────────── Metrics ────────────────────

def covert_action_rate(labels: list[str]) -> dict:
    """
    Compute covert action rate with Bayesian credible interval.
    
    Uses a Beta(1,1) (uniform) prior, following the paper's methodology.
    Returns posterior mean and 95% credible interval.
    """
    n_total = len(labels)
    n_covert = sum(1 for l in labels if l == "covert")
    n_aligned = sum(1 for l in labels if l == "aligned")
    n_ambiguous = sum(1 for l in labels if l == "ambiguous")
    
    if n_total == 0:
        return {"rate": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0, "n_covert": 0}
    
    # Beta-Binomial posterior with uniform prior
    alpha = n_covert + 1  # Beta prior alpha = 1
    beta = (n_total - n_covert) + 1  # Beta prior beta = 1
    
    rate = n_covert / n_total
    
    # 95% credible interval
    if HAS_SCIPY:
        ci_low = scipy_stats.beta.ppf(0.025, alpha, beta)
        ci_high = scipy_stats.beta.ppf(0.975, alpha, beta)
    else:
        # Approximate with normal (for when scipy not available)
        se = np.sqrt(rate * (1 - rate) / max(n_total, 1))
        ci_low = max(0, rate - 1.96 * se)
        ci_high = min(1, rate + 1.96 * se)
    
    return {
        "rate": rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": n_total,
        "n_covert": n_covert,
        "n_aligned": n_aligned,
        "n_ambiguous": n_ambiguous,
    }


def compute_all_rates(df: "pd.DataFrame", monitor_id: str = "rule_based") -> "pd.DataFrame":
    """
    Compute covert action rates for each (model, task, environment) combination.
    
    Returns a DataFrame with columns:
    model_id, task_id, environment_id, rate, ci_low, ci_high, n, n_covert
    """
    if not HAS_PANDAS:
        raise ImportError("pandas required")
    
    # Filter to specified monitor
    monitor_df = df[df["monitor_id"] == monitor_id]
    
    rows = []
    for (model_id, task_id, env_id), group in monitor_df.groupby(["model_id", "task_id", "environment_id"]):
        labels = group["label"].tolist()
        metrics = covert_action_rate(labels)
        rows.append({
            "model_id": model_id,
            "task_id": task_id,
            "environment_id": env_id,
            **metrics,
        })
    
    return pd.DataFrame(rows)


def aggregate_rates_by_model(rates_df: "pd.DataFrame") -> "pd.DataFrame":
    """Compute average covert action rate per model across all environments."""
    if not HAS_PANDAS:
        raise ImportError("pandas required")
    
    rows = []
    for model_id, group in rates_df.groupby("model_id"):
        # Aggregate: sum of covert / sum of total across all environments
        total_n = group["n"].sum()
        total_covert = group["n_covert"].sum()
        
        rate = total_covert / total_n if total_n > 0 else 0
        
        # Monte Carlo for aggregate CI (following paper's methodology)
        if HAS_SCIPY and total_n > 0:
            samples = []
            for _ in range(10000):
                env_rates = []
                for _, row in group.iterrows():
                    alpha = row["n_covert"] + 1
                    beta = (row["n"] - row["n_covert"]) + 1
                    env_rates.append(np.random.beta(alpha, beta))
                samples.append(np.mean(env_rates))
            ci_low = np.percentile(samples, 2.5)
            ci_high = np.percentile(samples, 97.5)
        else:
            se = np.sqrt(rate * (1 - rate) / max(total_n, 1))
            ci_low = max(0, rate - 1.96 * se)
            ci_high = min(1, rate + 1.96 * se)
        
        rows.append({
            "model_id": model_id,
            "rate": rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n": total_n,
            "n_covert": total_covert,
            "n_environments": len(group),
        })
    
    return pd.DataFrame(rows)


def cot_analysis_summary(df: "pd.DataFrame") -> "pd.DataFrame":
    """Summarize CoT analysis metrics across models and environments."""
    if not HAS_PANDAS:
        raise ImportError("pandas required")
    
    cot_df = df[df["monitor_id"] == "cot_analyzer"]
    if cot_df.empty:
        return pd.DataFrame()
    
    rows = []
    for (model_id, task_id), group in cot_df.groupby(["model_id", "task_id"]):
        n_with_cot = group["has_cot"].sum()
        rows.append({
            "model_id": model_id,
            "task_id": task_id,
            "n_total": len(group),
            "n_with_cot": int(n_with_cot),
            "pct_with_cot": n_with_cot / len(group) if len(group) > 0 else 0,
            "avg_cot_length": group.loc[group["has_cot"], "cot_length"].mean() if n_with_cot > 0 else 0,
            "eval_awareness_rate": group.get("metric_has_eval_awareness", pd.Series([False])).mean(),
            "deceptive_reasoning_rate": group.get("metric_has_deceptive_reasoning", pd.Series([False])).mean(),
            "spec_citation_rate": group.get("metric_has_spec_citation", pd.Series([False])).mean(),
        })
    
    return pd.DataFrame(rows)


# ──────────────────── Visualization ────────────────────

def plot_covert_rates_comparison(
    rates_df: "pd.DataFrame",
    title: str = "Covert Action Rates by Model and Environment",
    save_path: Optional[str] = None,
):
    """
    Bar chart comparing covert action rates across models and environments.
    Style modeled after Figure 9 in the paper.
    """
    if not HAS_MATPLOTLIB or not HAS_PANDAS:
        raise ImportError("matplotlib and pandas required for plotting")
    
    environments = rates_df["environment_id"].unique()
    models = rates_df["model_id"].unique()
    
    fig, ax = plt.subplots(figsize=(14, max(6, len(environments) * 0.8)))
    
    y_positions = np.arange(len(environments))
    bar_height = 0.8 / len(models)
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    for i, model_id in enumerate(models):
        model_data = rates_df[rates_df["model_id"] == model_id]
        
        rates = []
        ci_lows = []
        ci_highs = []
        
        for env in environments:
            env_data = model_data[model_data["environment_id"] == env]
            if len(env_data) > 0:
                row = env_data.iloc[0]
                rates.append(row["rate"])
                ci_lows.append(row["rate"] - row["ci_low"])
                ci_highs.append(row["ci_high"] - row["rate"])
            else:
                rates.append(0)
                ci_lows.append(0)
                ci_highs.append(0)
        
        positions = y_positions + i * bar_height
        ax.barh(
            positions, rates, bar_height,
            label=model_id, color=colors[i], alpha=0.85,
            xerr=[ci_lows, ci_highs], 
            error_kw={"capsize": 3, "capthick": 1, "elinewidth": 1},
        )
        
        # Add rate labels
        for j, (rate, pos) in enumerate(zip(rates, positions)):
            if rate > 0:
                ax.text(rate + 0.01, pos, f"{rate:.1%}", va="center", fontsize=9)
    
    ax.set_yticks(y_positions + bar_height * (len(models) - 1) / 2)
    ax.set_yticklabels(environments)
    ax.set_xlabel("Covert Action Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(0, min(1.0, rates_df["ci_high"].max() * 1.3 + 0.05))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    
    return fig


def plot_aggregate_comparison(
    agg_df: "pd.DataFrame",
    title: str = "Average Covert Action Rate by Model",
    save_path: Optional[str] = None,
):
    """Bar chart of aggregate covert action rates per model."""
    if not HAS_MATPLOTLIB or not HAS_PANDAS:
        raise ImportError("matplotlib and pandas required")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    models = agg_df["model_id"].tolist()
    rates = agg_df["rate"].tolist()
    ci_lows = [r - ci for r, ci in zip(rates, agg_df["ci_low"])]
    ci_highs = [ci - r for r, ci in zip(rates, agg_df["ci_high"])]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    bars = ax.bar(
        models, rates, color=colors, alpha=0.85,
        yerr=[ci_lows, ci_highs],
        error_kw={"capsize": 5, "capthick": 1.5, "elinewidth": 1.5},
    )
    
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{rate:.1%}", ha="center", va="bottom", fontweight="bold",
        )
    
    ax.set_ylabel("Average Covert Action Rate")
    ax.set_title(title)
    ax.set_ylim(0, max(rates) * 1.3 + 0.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_model_environment_heatmap(
    rates_df: "pd.DataFrame",
    title: str = "Covert Action Rates: Model × Environment",
    save_path: Optional[str] = None,
):
    """Heatmap showing covert action rates for each model-environment pair."""
    if not HAS_MATPLOTLIB or not HAS_PANDAS:
        raise ImportError("matplotlib and pandas required")
    
    pivot = rates_df.pivot_table(
        values="rate", index="environment_id", columns="model_id", aggfunc="mean"
    )
    
    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 2), max(6, len(pivot) * 0.6)))
    
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                       color="white" if val > 0.5 else "black", fontsize=10)
    
    plt.colorbar(im, ax=ax, label="Covert Action Rate")
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def generate_report(results_path: str, output_dir: str = None) -> str:
    """Generate a complete analysis report from experiment results."""
    results = load_results(results_path)
    
    if not results:
        return "No results to analyze."
    
    if output_dir is None:
        output_dir = str(Path(results_path).parent / "analysis")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    df = results_to_dataframe(results)
    rates = compute_all_rates(df)
    agg = aggregate_rates_by_model(rates)
    
    # Generate report text
    report = []
    report.append("=" * 60)
    report.append("SCHEMING EVALUATION ANALYSIS REPORT")
    report.append("=" * 60)
    report.append(f"\nTotal responses analyzed: {len(df[df['monitor_id'] == 'rule_based'])}")
    report.append(f"Models: {', '.join(df['model_id'].unique())}")
    report.append(f"Tasks: {', '.join(df['task_id'].unique())}")
    report.append(f"Environments: {', '.join(df['environment_id'].unique())}")
    
    report.append("\n" + "-" * 40)
    report.append("AGGREGATE COVERT ACTION RATES")
    report.append("-" * 40)
    for _, row in agg.iterrows():
        report.append(
            f"  {row['model_id']}: {row['rate']:.1%} "
            f"(95% CI: [{row['ci_low']:.1%}, {row['ci_high']:.1%}], "
            f"n={row['n']}, covert={row['n_covert']})"
        )
    
    report.append("\n" + "-" * 40)
    report.append("PER-ENVIRONMENT BREAKDOWN")
    report.append("-" * 40)
    for _, row in rates.iterrows():
        report.append(
            f"  {row['model_id']} | {row['task_id']}/{row['environment_id']}: "
            f"{row['rate']:.1%} ({row['n_covert']}/{row['n']})"
        )
    
    # Generate plots
    if HAS_MATPLOTLIB:
        try:
            plot_covert_rates_comparison(rates, save_path=f"{output_dir}/rates_by_environment.png")
            plot_aggregate_comparison(agg, save_path=f"{output_dir}/aggregate_rates.png")
            if len(df["model_id"].unique()) > 1:
                plot_model_environment_heatmap(rates, save_path=f"{output_dir}/heatmap.png")
            report.append(f"\nPlots saved to: {output_dir}/")
        except Exception as e:
            report.append(f"\nWarning: Could not generate plots: {e}")
    
    report_text = "\n".join(report)
    
    # Save report
    report_path = f"{output_dir}/report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    
    print(report_text)
    return report_text
