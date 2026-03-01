#!/usr/bin/env python3
"""
Analyze results from a scheming evaluation experiment.

Usage:
    python scripts/analyze_results.py experiments/moltbook_eval_v1/results.jsonl
    python scripts/analyze_results.py experiments/moltbook_eval_v1/results.jsonl --output analysis_output/
    python scripts/analyze_results.py experiments/exp1/results.jsonl experiments/exp2/results.jsonl --compare
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.metrics import (
    load_results, results_to_dataframe, compute_all_rates,
    aggregate_rates_by_model, cot_analysis_summary, generate_report,
    plot_covert_rates_comparison, plot_aggregate_comparison,
    plot_model_environment_heatmap,
)


def main():
    parser = argparse.ArgumentParser(description="Analyze scheming evaluation results")
    parser.add_argument("results", nargs="+", help="Path(s) to results.jsonl files")
    parser.add_argument("--output", "-o", help="Output directory for analysis")
    parser.add_argument("--compare", action="store_true", help="Compare multiple experiments")
    parser.add_argument("--monitor", default="rule_based", help="Monitor to use for scoring")
    args = parser.parse_args()
    
    if args.compare and len(args.results) > 1:
        compare_experiments(args.results, args.output, args.monitor)
    else:
        for results_path in args.results:
            print(f"\nAnalyzing: {results_path}")
            generate_report(results_path, args.output)


def compare_experiments(result_paths: list, output_dir: str, monitor_id: str):
    """Compare results across multiple experiments."""
    import pandas as pd
    
    all_rates = []
    for path in result_paths:
        results = load_results(path)
        df = results_to_dataframe(results)
        rates = compute_all_rates(df, monitor_id=monitor_id)
        rates["experiment"] = os.path.basename(os.path.dirname(path))
        all_rates.append(rates)
    
    combined = pd.concat(all_rates, ignore_index=True)
    
    print("\n" + "=" * 60)
    print("CROSS-EXPERIMENT COMPARISON")
    print("=" * 60)
    
    for exp, group in combined.groupby("experiment"):
        agg = aggregate_rates_by_model(group)
        print(f"\n{exp}:")
        for _, row in agg.iterrows():
            print(f"  {row['model_id']}: {row['rate']:.1%} (n={row['n']})")


if __name__ == "__main__":
    main()
