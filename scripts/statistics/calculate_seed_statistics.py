#!/usr/bin/env python3
"""
Calculate benchmark-wise statistics (mean, std) across multiple seed runs.

This script scans a directory for model results, reads the seed and configuration
from dpo_training.yaml in each folder, groups by configuration (base_run_name),
and calculates mean and standard deviation for each benchmark across the different seeds.

Supports two modes:
1. --unique_seeds: Use only one run per unique seed value (for cross-seed variance)
2. --same_seed N: Use N runs with the same seed (for reproducibility/training noise analysis)

Usage:
    # Cross-seed variance (one run per unique seed, using first occurrence)
    python calculate_seed_statistics.py --results_dir /path/to/models --unique_seeds --display
    
    # Same-seed variance (5 runs with seed=42 to measure training noise)
    python calculate_seed_statistics.py --results_dir /path/to/models --same_seed 42 --num_runs 5 --display
    
Example:
    python calculate_seed_statistics.py \
        --results_dir /iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/models/dpo_seeds2 \
        --unique_seeds \
        --output seed_stats.csv \
        --display
"""

import pandas as pd
import argparse
import numpy as np
import json
import os
import csv
import yaml
from pathlib import Path
from collections import defaultdict


# ==============================================================================
#                           CONFIG LOADING
# ==============================================================================

def load_dpo_config(dir_path):
    """
    Load configuration from dpo_training.yaml in a model directory.
    
    Args:
        dir_path: Path to the model directory
    
    Returns:
        Dictionary with config data or None if not found
    """
    config_path = os.path.join(dir_path, "dpo_training.yaml")
    
    if not os.path.exists(config_path):
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"  ⚠️  Error reading {config_path}: {e}")
        return None


# ==============================================================================
#                           BENCHMARK LOADING
# ==============================================================================

def _read_score_from_file(filepath):
    """Read a score from a JSON or CSV file."""
    try:
        if not os.path.exists(filepath):
            return None
            
        if filepath.endswith(".json"):
            with open(filepath, 'r') as f:
                data = json.load(f)
                if "metrics" in data and isinstance(data["metrics"], list):
                    return data["metrics"][0].get("primary_score")
                return data.get("score") or data.get("primary_score")
                
        elif filepath.endswith(".csv"):
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                first_row = next(reader)
                return float(first_row.get("length_controlled_winrate", 0)) / 100.0
    except Exception as e:
        print(f"  ⚠️  Error reading {filepath}: {e}")
    return None


def load_dpo_results(dir_path):
    """
    Load DPO benchmark results from a model directory.
    
    Args:
        dir_path: Path to the model directory
    
    Returns:
        Dictionary with benchmark scores or empty dict if not found
    """
    data = {}
    results_path = os.path.join(dir_path, "results")
    
    if not os.path.exists(results_path):
        return data
    
    BENCHMARK_PATHS = {
        "GSM8K": "gsm8k_tulu/metrics.json",
        "IF_Eval": "ifeval_tulu/metrics.json",
        "TruthfulQA": "truthfulqa_tulu/metrics.json",
        "AlpacaEval": "alpaca_eval/activeuf/leaderboard.csv",
    }
    
    for benchmark, rel_path in BENCHMARK_PATHS.items():
        score_file = os.path.join(results_path, rel_path)
        score = _read_score_from_file(score_file)
        if score is not None:
            # Normalize to 0-1 scale if needed (scores > 1 are assumed to be 0-100)
            if score > 1.0:
                score = score / 100.0
            data[benchmark] = score
            data[benchmark] = score
    
    return data


def load_rm_results(dir_path):
    """
    Load Reward Model benchmark results from a model directory.
    
    Args:
        dir_path: Path to the model directory
    
    Returns:
        Dictionary with benchmark scores or empty dict if not found
    """
    data = {}
    metrics_file = os.path.join(dir_path, "metrics.json")
    
    if not os.path.exists(metrics_file):
        return data
    
    try:
        with open(metrics_file, 'r') as f:
            rm_data = json.load(f)
        
        # Map of benchmark names to possible keys in the JSON
        RM_BENCHMARKS = {
            "Factuality": ["Rewardbench/Factuality", "factuality", "accuracy_factuality"],
            "Focus": ["Rewardbench/Focus", "focus", "accuracy_focus"],
            "Math": ["Rewardbench/Math", "math", "accuracy_math"],
            "Precise_IF": ["Rewardbench/Precise IF", "precise_if", "accuracy_precise_if"],
            "Safety": ["Rewardbench/Safety", "safety", "accuracy_safety"],
            "Ties": ["Rewardbench/Ties", "ties", "accuracy_ties"],
        }
        
        for benchmark, search_keys in RM_BENCHMARKS.items():
            score = None
            for key in search_keys:
                if key in rm_data:
                    score = rm_data[key]
                    break
            
            if score is not None:
                data[benchmark] = score
                
    except Exception as e:
        print(f"  ⚠️  Error reading RM metrics: {e}")
    
    return data


# ==============================================================================
#                           GROUPING LOGIC
# ==============================================================================

def group_results_by_config(results_dir, result_type="dpo"):
    """
    Group model results by configuration (base_run_name from dpo_training.yaml).
    
    Args:
        results_dir: Directory containing model subdirectories
        result_type: "dpo" or "rm"
    
    Returns:
        Dictionary mapping config_name -> list of (seed, scores_dict, dir_name)
    """
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return {}
    
    load_fn = load_dpo_results if result_type == "dpo" else load_rm_results
    
    subdirs = sorted([
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ])
    
    print(f"\n📁 Scanning {results_dir}")
    print(f"Found {len(subdirs)} directories\n")
    
    grouped = defaultdict(list)
    
    for subdir in subdirs:
        subdir_path = os.path.join(results_dir, subdir)
        
        # Load config from dpo_training.yaml
        config_data = load_dpo_config(subdir_path)
        
        if config_data is None:
            print(f"Skipping {subdir}: no dpo_training.yaml found")
            continue
        
        # Extract config name and seed from the yaml file
        config_name = config_data.get("base_run_name", subdir)
        seed = config_data.get("seed", None)
        
        print(f"Loading {subdir}")
        print(f"  → Config: {config_name}")
        print(f"  → Seed: {seed}")
        
        scores = load_fn(subdir_path)
        
        if scores:
            print(f"  → Found {len(scores)} benchmarks: {list(scores.keys())}")
            grouped[config_name].append((seed, scores, subdir))
        else:
            print(f"  → No results found")
        print()
    
    return grouped


def filter_unique_seeds(grouped_results):
    """
    Filter results to keep only one run per unique seed value.
    Takes the first occurrence of each seed.
    
    Args:
        grouped_results: Dict mapping config -> list of (seed, scores, dir_name)
    
    Returns:
        Filtered dictionary with unique seeds only
    """
    filtered = {}
    
    for config, seed_results in grouped_results.items():
        seen_seeds = set()
        unique_results = []
        
        for seed, scores, dir_name in seed_results:
            if seed not in seen_seeds:
                seen_seeds.add(seed)
                unique_results.append((seed, scores, dir_name))
        
        if unique_results:
            filtered[config] = unique_results
            
    return filtered


def filter_same_seed(grouped_results, target_seed, num_runs=5):
    """
    Filter results to keep only runs with a specific seed value.
    Used for measuring reproducibility/training noise.
    
    Args:
        grouped_results: Dict mapping config -> list of (seed, scores, dir_name)
        target_seed: The seed value to filter for (e.g., 42)
        num_runs: Maximum number of runs to include
    
    Returns:
        Filtered dictionary with only runs matching target_seed
    """
    filtered = {}
    
    for config, seed_results in grouped_results.items():
        matching_results = [
            (seed, scores, dir_name)
            for seed, scores, dir_name in seed_results
            if seed == target_seed
        ]
        
        # Take up to num_runs
        if matching_results:
            filtered[config] = matching_results[:num_runs]
            
    return filtered


# ==============================================================================
#                           STATISTICS CALCULATION
# ==============================================================================

def calculate_statistics(grouped_results, min_seeds=2):
    """
    Calculate mean and standard deviation for each configuration.
    
    Args:
        grouped_results: Dict mapping config -> list of (seed, scores, dir_name)
        min_seeds: Minimum number of seeds required to include a config
    
    Returns:
        DataFrame with statistics
    """
    stats_rows = []
    
    for config, seed_results in grouped_results.items():
        if len(seed_results) < min_seeds:
            print(f"⚠️  Skipping {config}: only {len(seed_results)} seed(s), need at least {min_seeds}")
            continue
        
        # Collect all benchmarks seen across all seeds
        all_benchmarks = set()
        for _, scores, _ in seed_results:
            all_benchmarks.update(scores.keys())
        
        all_benchmarks = sorted(all_benchmarks)
        
        # Build a matrix: rows=seeds, cols=benchmarks
        seed_numbers = [seed if seed is not None else i for i, (seed, _, _) in enumerate(seed_results)]
        dir_names = [dir_name for _, _, dir_name in seed_results]
        benchmark_matrix = defaultdict(list)
        
        for seed, scores, _ in seed_results:
            for benchmark in all_benchmarks:
                benchmark_matrix[benchmark].append(scores.get(benchmark, np.nan))
        
        # Calculate statistics
        row = {
            "Config": config,
            "Num_Seeds": len(seed_results),
            "Seeds": ",".join(map(str, sorted(seed_numbers))),
            "Directories": ",".join(dir_names),
        }
        
        for benchmark in all_benchmarks:
            values = [v for v in benchmark_matrix[benchmark] if not np.isnan(v)]
            
            if len(values) > 0:
                mean = np.mean(values)
                std = np.std(values, ddof=1) if len(values) > 1 else 0.0
                row[f"{benchmark}_mean"] = mean
                row[f"{benchmark}_std"] = std
                row[f"{benchmark}_n"] = len(values)
            else:
                row[f"{benchmark}_mean"] = np.nan
                row[f"{benchmark}_std"] = np.nan
                row[f"{benchmark}_n"] = 0
        
        # Calculate per-run overall means, then compute mean and std across runs
        per_run_overall_means = []
        for seed, scores, _ in seed_results:
            run_values = [scores.get(b, np.nan) for b in all_benchmarks]
            run_values = [v for v in run_values if not np.isnan(v)]
            if run_values:
                per_run_overall_means.append(np.mean(run_values))
        
        if per_run_overall_means:
            row["Overall_mean"] = np.mean(per_run_overall_means)
            row["Overall_std"] = np.std(per_run_overall_means, ddof=1) if len(per_run_overall_means) > 1 else 0.0
        
        stats_rows.append(row)
    
    return pd.DataFrame(stats_rows)


# ==============================================================================
#                           OUTPUT
# ==============================================================================

def format_results_table(df, result_type="dpo"):
    """
    Format results as a clean table for display.
    
    Args:
        df: DataFrame with statistics
        result_type: "dpo" or "rm"
    
    Returns:
        Formatted DataFrame
    """
    if df.empty:
        return df
    
    # Extract benchmark names (columns ending in _mean)
    benchmark_cols = [c.replace("_mean", "") for c in df.columns if c.endswith("_mean") and c != "Overall_mean"]
    
    formatted_rows = []
    for _, row in df.iterrows():
        formatted = {
            "Configuration": row["Config"],
            "Seeds": f"{row['Num_Seeds']} ({row['Seeds']})",
        }
        
        for benchmark in benchmark_cols:
            mean = row.get(f"{benchmark}_mean", np.nan)
            std = row.get(f"{benchmark}_std", np.nan)
            n = row.get(f"{benchmark}_n", 0)
            
            if not np.isnan(mean) and n > 0:
                if n > 1:
                    formatted[benchmark] = f"{mean:.3f} ± {std:.3f}"
                else:
                    formatted[benchmark] = f"{mean:.3f}"
            else:
                formatted[benchmark] = "-"
        
        # Overall
        if "Overall_mean" in row:
            overall_mean = row["Overall_mean"]
            overall_std = row.get("Overall_std", np.nan)
            if not np.isnan(overall_mean):
                formatted["Overall"] = f"{overall_mean:.3f} ± {overall_std:.3f}"
        
        formatted_rows.append(formatted)
    
    return pd.DataFrame(formatted_rows)


def save_results(df, output_path, format_type="both"):
    """
    Save results to file(s).
    
    Args:
        df: DataFrame with statistics
        output_path: Output file path
        format_type: "csv", "tex", or "both"
    """
    if df.empty:
        print("⚠️  No results to save")
        return
    
    base_path = output_path.rsplit('.', 1)[0]
    
    if format_type in ["csv", "both"]:
        csv_path = f"{base_path}.csv"
        df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"✓ Saved CSV: {csv_path}")
    
    if format_type in ["tex", "both"]:
        tex_path = f"{base_path}.tex"
        with open(tex_path, 'w') as f:
            f.write("% Auto-generated statistics table\n\n")
            f.write(df.to_latex(index=False, escape=False, float_format='%.3f'))
        print(f"✓ Saved LaTeX: {tex_path}")


# ==============================================================================
#                           MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Calculate benchmark statistics across multiple seed runs"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing model result subdirectories"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="seed_statistics.csv",
        help="Output file path (default: seed_statistics.csv)"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["dpo", "rm"],
        default="dpo",
        help="Result type: dpo or rm (default: dpo)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "tex", "both"],
        default="both",
        help="Output format (default: both)"
    )
    parser.add_argument(
        "--min_seeds",
        type=int,
        default=2,
        help="Minimum number of seeds/runs required (default: 2)"
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display formatted table in console"
    )
    
    # Filtering options
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--unique_seeds",
        action="store_true",
        help="Use only one run per unique seed value (for cross-seed variance)"
    )
    filter_group.add_argument(
        "--same_seed",
        type=int,
        metavar="SEED",
        help="Use only runs with this specific seed value (for reproducibility analysis)"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Maximum number of runs to use with --same_seed (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Determine mode description
    if args.unique_seeds:
        mode_desc = "Unique Seeds (one run per seed value)"
    elif args.same_seed is not None:
        mode_desc = f"Same Seed ({args.same_seed}) - up to {args.num_runs} runs"
    else:
        mode_desc = "All runs (no filtering)"
    
    print(f"\n{'='*70}")
    print(f"Calculating Statistics for {args.type.upper()} Results")
    print(f"{'='*70}\n")
    print(f"Results directory: {args.results_dir}")
    print(f"Mode: {mode_desc}")
    print(f"Minimum runs required: {args.min_seeds}")
    print(f"Output: {args.output}")
    print()
    
    # Load and group results
    grouped = group_results_by_config(args.results_dir, args.type)
    
    if not grouped:
        print("❌ No results found")
        return
    
    # Apply filtering based on mode
    if args.unique_seeds:
        print(f"\n{'='*70}")
        print("Filtering to unique seeds only...")
        print(f"{'='*70}\n")
        grouped = filter_unique_seeds(grouped)
        
        for config, results in grouped.items():
            seeds = [seed for seed, _, _ in results]
            print(f"  {config}: {len(results)} unique seeds: {seeds}")
        print()
        
    elif args.same_seed is not None:
        print(f"\n{'='*70}")
        print(f"Filtering to runs with seed={args.same_seed}...")
        print(f"{'='*70}\n")
        grouped = filter_same_seed(grouped, args.same_seed, args.num_runs)
        
        for config, results in grouped.items():
            dirs = [dir_name for _, _, dir_name in results]
            print(f"  {config}: {len(results)} runs with seed={args.same_seed}")
            for d in dirs:
                print(f"    - {d}")
        print()
    
    if not grouped:
        print("❌ No results found after filtering")
        return
    
    print(f"\n{'='*70}")
    print(f"Found {len(grouped)} unique configurations")
    print(f"{'='*70}\n")
    
    # Calculate statistics
    stats_df = calculate_statistics(grouped, args.min_seeds)
    
    if stats_df.empty:
        print("❌ No configurations have enough runs")
        return
    
    # Save raw statistics
    save_results(stats_df, args.output, args.format)
    
    # Display formatted table
    if args.display:
        formatted_df = format_results_table(stats_df, args.type)
        print(f"\n{'='*70}")
        print("Results Summary")
        print(f"{'='*70}\n")
        print(formatted_df.to_string(index=False))
        print()
    
    print(f"\n{'='*70}")
    print(f"✓ Statistics calculated for {len(stats_df)} configurations")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
