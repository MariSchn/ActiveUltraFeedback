#!/usr/bin/env python3
"""
Generate tables from directory-based results.
Instead of fetching from wandb, this scans a directory for subdirectories,
uses them as rows, and loads results from each.
"""

import pandas as pd
import argparse
import numpy as np
import json
import os
import csv
from collections import defaultdict
from pathlib import Path

# ==============================================================================
#                               RAW BASELINE DATA (same as generate_tables.py)
# ==============================================================================

DPO_TASK_BASELINES = {
    "GSM8K": { "SFT": 0.758, "DeltaQwen": 0.813, "MaxMin": 0.780, "Random": 0.782, "UltraFeedback": 0.795 },
    "IF Eval": { "SFT": 0.713, "DeltaQwen": 0.760, "MaxMin": 0.697, "Random": 0.741, "UltraFeedback": 0.712 },
    "Truthful QA": { "SFT": 0.468, "DeltaQwen": 0.598, "MaxMin": 0.618, "Random": 0.524, "UltraFeedback": 0.507 },
    "Alpaca Eval": { "SFT": 0.083, "DeltaQwen": 0.399, "MaxMin": 0.372, "Random": 0.160, "UltraFeedback": 0.155 },
}

IPO_TASK_BASELINES = {
    "GSM8K": { "Random": 0.824, "MaxMin": 0.827, "DeltaQwen": 0.815, "UltraFeedback": 0.832 },
    "IF Eval": { "Random": 0.613, "MaxMin": 0.706, "DeltaQwen": 0.752, "UltraFeedback": 0.713 },
    "Truthful QA": { "Random": 0.580, "MaxMin": 0.595, "DeltaQwen": 0.492, "UltraFeedback": 0.517 },
    "Alpaca Eval": { "Random": 0.498, "MaxMin": 0.498, "DeltaQwen": 0.357, "UltraFeedback": 0.498 }
}

SIMPO_TASK_BASELINES = {
    "GSM8K": { "UltraFeedback": 0.796, "DeltaQwen": 0.821, "MaxMin": 0.764, "Random": 0.803 },
    "IF Eval": { "UltraFeedback": 0.670, "DeltaQwen": 0.731, "MaxMin": 0.654, "Random": 0.706 },
    "Truthful QA": { "UltraFeedback": 0.630, "DeltaQwen": 0.532, "MaxMin": 0.653, "Random": 0.600 },
    "Alpaca Eval": { "UltraFeedback": 0.650, "DeltaQwen": 0.517, "MaxMin": 0.543, "Random": 0.579 }
}

DPO_MEAN_BASELINES = { "SFT": 0.506, "DeltaQwen": 0.643, "MaxMin": 0.617, "Random": 0.552, "UltraFeedback": 0.542 }

RM_MEAN_BASELINES = { "SFT": 0.290, "DeltaQwen": 0.390, "MaxMin": 0.608, "Random": 0.568, "UltraFeedback": 0.577 }

RM_TASK_BASELINES = {
    "Factuality": { "SFT": 0.316, "DeltaQwen": 0.511, "MaxMin": 0.693, "Random": 0.759, "UltraFeedback": 0.759 },
    "Focus": { "SFT": 0.277, "DeltaQwen": 0.243, "MaxMin": 0.760, "Random": 0.486, "UltraFeedback": 0.465 },
    "Math": { "SFT": 0.445, "DeltaQwen": 0.473, "MaxMin": 0.601, "Random": 0.601, "UltraFeedback": 0.658 },
    "Precise IF": { "SFT": 0.261, "DeltaQwen": 0.328, "MaxMin": 0.384, "Random": 0.394, "UltraFeedback": 0.375 },
    "Safety": { "SFT": 0.347, "DeltaQwen": 0.563, "MaxMin": 0.717, "Random": 0.764, "UltraFeedback": 0.828 },
    "Ties": { "SFT": 0.095, "DeltaQwen": 0.221, "MaxMin": 0.495, "Random": 0.405, "UltraFeedback": 0.379 },
}

# ==============================================================================
#                               CONFIGURATION
# ==============================================================================

POLICY_TASK_COLS = ["GSM8K", "IF Eval", "Truthful QA", "Alpaca Eval", "Mean"]
RM_COLS = ["Factuality", "Focus", "Math", "Precise IF", "Safety", "Ties", "Mean"]
BASELINE_METHODS_ORDER = ["Random", "UltraFeedback", "MaxMin", "DeltaQwen"]

KEY_MAPPING = {
    "DPO": {
        "GSM8K": "DPO/GSM8K", "IF Eval": "DPO/IF Eval", "Truthful QA": "DPO/Truthful QA",
        "Mean": "DPO/Mean", "Alpaca Eval": "DPO/Alpaca Eval",
    },
    "RM": {
        "Factuality": "Rewardbench/Factuality",
        "Focus": "Rewardbench/Focus", 
        "Math": "Rewardbench/Math",
        "Precise IF": "Rewardbench/Precise IF",
        "Safety": "Rewardbench/Safety", 
        "Ties": "Rewardbench/Ties",
        "Mean": "Rewardbench/Mean",
    },
}

# Known acquisition function names to extract
ACQUISITION_FUNCTIONS = ["DRTS", "DTS", "DeltaUCB", "InfoMax", "MaxMinLCB", "Random", "MaxMin", "UltraFeedback", "DeltaQwen"]

def clean_row_name(name):
    """Clean row name by removing job ID prefixes (e.g., '1234567-name' -> 'name')."""
    import re
    # Remove leading job ID pattern (digits followed by dash or underscore)
    cleaned = re.sub(r'^\d+[-_]', '', name)
    return cleaned if cleaned else name

# Globals
SFT_BASE = None
OTHER_BASELINES_DELTAS = None

# ==============================================================================
#                           DATA PROCESSING LOGIC
# ==============================================================================

def _read_score_from_file(filepath):
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
        pass
    return None

def overwrite_baseline_scores(parent_dir, task_baselines, mean_baselines, type_key, cols):
    if not parent_dir or not os.path.exists(parent_dir):
        return

    print(f"[{type_key}] Scanning for baselines in: {parent_dir}")
    available_folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]

    BENCHMARK_PATHS = {
        "GSM8K": "results/gsm8k_tulu/metrics.json",
        "IF Eval": "results/ifeval_tulu/metrics.json",
        "Truthful QA": "results/truthfulqa_tulu/metrics.json",
        "Alpaca Eval": "results/alpaca_eval/activeuf/leaderboard.csv",
    }

    for method in BASELINE_METHODS_ORDER:
        matched_folder = None
        for folder in available_folders:
            # Fuzzy match (e.g. "delta_qwen" matches "DeltaQwen")
            clean_method = method.lower().replace("-", "").replace("_", "")
            clean_folder = folder.lower().replace("-", "").replace("_", "")
            if clean_method in clean_folder:
                matched_folder = folder
                break
        
        if not matched_folder:
            print(f"  [Skip] No folder found for baseline '{method}'")
            continue

        full_model_path = os.path.join(parent_dir, matched_folder)
        print(f"  [Found] '{method}' -> {matched_folder}")

        if type_key == "RM":
            json_file = os.path.join(full_model_path, "metrics.json")
            if os.path.exists(json_file):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    print(f"    -> JSON keys found: {list(data.keys())[:10]}...")
                    
                    for col in cols:
                        # Build comprehensive search keys
                        target_key = KEY_MAPPING["RM"].get(col)
                        col_lower = col.lower()
                        col_underscore = col.lower().replace(" ", "_")
                        
                        search_keys = []
                        if target_key:
                            search_keys.append(target_key)
                            search_keys.append(target_key.lower())
                        search_keys += [
                            col, col_lower, col_underscore,
                            f"Rewardbench/{col}", f"rewardbench/{col_lower}", f"rewardbench/{col_underscore}",
                            f"Rewardbench/{col_underscore}", f"accuracy_{col_underscore}",
                            f"accuracy_{col_lower.replace(' ', '_')}",
                        ]
                        
                        score = next((data[k] for k in search_keys if k in data), None)
                        
                        if score is not None:
                            print(f"      [RM] {col} = {score}")
                            if col == "Mean":
                                if mean_baselines is not None: mean_baselines[method] = score
                            else:
                                if col not in task_baselines: task_baselines[col] = {}
                                task_baselines[col][method] = score
                        else:
                            print(f"      [RM] {col} NOT FOUND (tried: {search_keys[:4]}...)")
                                
                    print(f"    -> Updated RM scores for {method}")
                except Exception as e:
                    print(f"    -> Error reading RM json: {e}")
            else:
                print(f"    -> Missing metrics.json for RM {method}")

        else:
            found_any = False
            for col in cols:
                if col == "Mean": continue
                rel_path = BENCHMARK_PATHS.get(col)
                if not rel_path: continue
                score_file = os.path.join(full_model_path, rel_path)
                score = _read_score_from_file(score_file)
                if score is not None:
                    if col not in task_baselines: task_baselines[col] = {}
                    task_baselines[col][method] = score
                    found_any = True
            
            if found_any:
                print(f"    -> Updated Task scores for {method}")
                if mean_baselines is not None:
                    new_mean = calculate_mean_for_dict(task_baselines, method)
                    if not np.isnan(new_mean):
                        mean_baselines[method] = new_mean

def calculate_mean_for_dict(task_baselines, method):
    scores = []
    tasks = ["GSM8K", "IF Eval", "Truthful QA", "Alpaca Eval"]
    for t in tasks:
        val = task_baselines.get(t, {}).get(method, np.nan)
        if not np.isnan(val):
            scores.append(val)
    return np.mean(scores) if scores else np.nan

def process_baselines():
    sft_base = {"DPO": {}, "IPO": {}, "SIMPO": {}, "RM": {}}
    other_deltas = {"DPO": {}, "IPO": {}, "SIMPO": {}, "RM": {}}

    for type_key in ["DPO", "IPO", "SIMPO", "RM"]:
        for method in BASELINE_METHODS_ORDER:
            if method not in other_deltas[type_key]:
                other_deltas[type_key][method] = []

    def process_type(type_name, cols, task_baselines, mean_baselines=None):
        for col in cols:
            if type_name in ["IPO", "SIMPO"]:
                if col == "Mean":
                     sft_val = DPO_MEAN_BASELINES.get("SFT", np.nan)
                else:
                    sft_val = DPO_TASK_BASELINES.get(col, {}).get("SFT", np.nan)
            else:
                if col == "Mean" and mean_baselines:
                    sft_val = mean_baselines.get("SFT", np.nan)
                else:
                    sft_val = task_baselines.get(col, {}).get("SFT", np.nan)
            
            sft_base[type_name][col] = sft_val

            for method in BASELINE_METHODS_ORDER:
                if col == "Mean":
                    if mean_baselines:
                        val = mean_baselines.get(method, np.nan)
                    else:
                        val = calculate_mean_for_dict(task_baselines, method)
                else:
                    val = task_baselines.get(col, {}).get(method, np.nan)
                
                delta = (
                    val - sft_val if not np.isnan(val) and not np.isnan(sft_val) else np.nan
                )
                other_deltas[type_name][method].append(delta)

    process_type("DPO", POLICY_TASK_COLS, DPO_TASK_BASELINES, DPO_MEAN_BASELINES)
    process_type("IPO", POLICY_TASK_COLS, IPO_TASK_BASELINES, None)
    process_type("SIMPO", POLICY_TASK_COLS, SIMPO_TASK_BASELINES, None)
    process_type("RM", RM_COLS, RM_TASK_BASELINES, RM_MEAN_BASELINES)

    return sft_base, other_deltas


def format_delta(val, is_best=False):
    if val is None or np.isnan(val): return "-"
    sign = "+" if val >= 0 else ""
    text = f"{sign}{val:.3f}"
    return f"\\textbf{{{text}}}" if is_best else text

def format_sft(val):
    return "-" if val is None or np.isnan(val) else f"{val:.3f}"

def process_section_dataframe(df_numeric, cols):
    if df_numeric.empty: return pd.DataFrame()
    df_formatted = df_numeric.copy()
    max_series = df_numeric.max(numeric_only=True)
    for col in cols:
        if col not in df_numeric.columns: continue
        max_val = max_series[col]
        for idx in df_numeric.index:
            val = df_numeric.at[idx, col]
            is_best = (val == max_val) and (not np.isnan(val))
            df_formatted.at[idx, col] = format_delta(val, is_best)
    return df_formatted

def get_static_data():
    static_packs = {}
    specs = [("DPO", POLICY_TASK_COLS), ("IPO", POLICY_TASK_COLS), ("SIMPO", POLICY_TASK_COLS), ("RM", RM_COLS)]

    for type_name, cols in specs:
        sft_df = pd.DataFrame([SFT_BASE[type_name]], index=["SFT Base Model"])
        sft_df = sft_df[cols]
        for c in sft_df.columns: sft_df[c] = sft_df[c].apply(format_sft)
        
        base_dict = {}
        for name, deltas in OTHER_BASELINES_DELTAS[type_name].items():
            base_dict[name] = dict(zip(cols, deltas))
        df_base_num = pd.DataFrame.from_dict(base_dict, orient="index")[cols]
        static_packs[type_name] = (sft_df, df_base_num)

    return static_packs

# ==============================================================================
#                           DIRECTORY-BASED LOGIC
# ==============================================================================

def load_results_from_dir(dir_path):
    """
    Load results from a directory.
    - DPO scores: looks in results/ subdirectory for benchmark files
    - RM scores: looks for metrics.json directly in the directory
    
    Args:
        dir_path: Path to the experiment directory
    
    Returns:
        Dictionary with scores or None if not found
    """
    data = {}
    
    # Read DPO scores from results/ subdirectory
    results_path = os.path.join(dir_path, "results")
    
    if os.path.exists(results_path):
        BENCHMARK_PATHS = {
            "GSM8K": "gsm8k_tulu/metrics.json",
            "IF Eval": "ifeval_tulu/metrics.json",
            "Truthful QA": "truthfulqa_tulu/metrics.json",
            "Alpaca Eval": "alpaca_eval/activeuf/leaderboard.csv",
        }
        
        for col, rel_path in BENCHMARK_PATHS.items():
            score_file = os.path.join(results_path, rel_path)
            score = _read_score_from_file(score_file)
            if score is not None:
                data[f"DPO/{col}"] = score
    
    # Read RM scores from metrics.json in the directory itself
    metrics_file = os.path.join(dir_path, "metrics.json")
    
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                rm_data = json.load(f)
            
            # Search for RM scores using comprehensive key patterns (same as baseline loading)
            RM_COLS_LIST = ["Factuality", "Focus", "Math", "Precise IF", "Safety", "Ties", "Mean"]
            
            for col in RM_COLS_LIST:
                # Build comprehensive search keys
                target_key = KEY_MAPPING["RM"].get(col)
                col_lower = col.lower()
                col_underscore = col.lower().replace(" ", "_")
                
                search_keys = []
                if target_key:
                    search_keys.append(target_key)
                    search_keys.append(target_key.lower())
                search_keys += [
                    col, col_lower, col_underscore,
                    f"Rewardbench/{col}", f"rewardbench/{col_lower}", f"rewardbench/{col_underscore}",
                    f"Rewardbench/{col_underscore}", f"accuracy_{col_underscore}",
                    f"accuracy_{col_lower.replace(' ', '_')}",
                ]
                
                score = None
                for key in search_keys:
                    if key in rm_data:
                        score = rm_data[key]
                        break
                
                if score is not None:
                    data[f"Rewardbench/{col}"] = score
                    
        except Exception as e:
            print(f"  ⚠️  Error reading metrics.json: {e}")
    
    return data if data else None


def extract_scores_from_data(data, type_key, cols):
    """
    Extract scores from data dictionary.
    
    Args:
        data: Dictionary with metrics
        type_key: "DPO" or "RM"
        cols: List of column names
    
    Returns:
        Dictionary with scores or None if insufficient data
    """
    row_data = {}
    has_data = False
    
    for col in cols:
        if col == "Mean":
            continue  # Skip Mean for now, will compute it
        
        # Try to find the score using various key patterns
        target_key = KEY_MAPPING.get(type_key, {}).get(col)
        val = None
        
        # Try primary mapping key
        if target_key:
            val = data.get(target_key)
        
        # Fallback: try other patterns
        if val is None:
            candidates = [f"{type_key}/{col}", col, col.lower()]
            for cand in candidates:
                if cand in data:
                    val = data[cand]
                    break
        
        if val is not None:
            has_data = True
            try:
                score = float(val)
                base_val = SFT_BASE[type_key].get(col, np.nan)
                row_data[col] = score - base_val if not np.isnan(base_val) else score
            except:
                row_data[col] = np.nan
        else:
            row_data[col] = np.nan
    
    # Compute mean
    if has_data:
        metric_vals = [row_data[c] for c in cols[:-1] if c != "Mean"]
        metric_vals = [v for v in metric_vals if not np.isnan(v)]
        row_data["Mean"] = np.mean(metric_vals) if metric_vals else np.nan
    
    return row_data if has_data else None


def fetch_results_from_directory(results_dir):
    """
    Scan results_dir for subdirectories and load results from each.
    Uses directory names as row labels.
    
    Returns:
        Tuple of (df_dpo, df_ipo, df_simpo, df_rm)
    """
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    subdirs = sorted([
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ])
    
    print(f"\n📁 Scanning {results_dir}")
    print(f"Found {len(subdirs)} directories:\n")
    for d in subdirs:
        print(f"  - {d}")
    print()
    
    parsed_results = {}
    
    for subdir in subdirs:
        subdir_path = os.path.join(results_dir, subdir)
        clean_name = clean_row_name(subdir)
        print(f"Loading {subdir} -> {clean_name}...")
        
        data = load_results_from_dir(subdir_path)
        if data is None:
            continue
        
        parsed_results[clean_name] = {
            "dpo_row": extract_scores_from_data(data, "DPO", POLICY_TASK_COLS),
            "ipo_row": extract_scores_from_data(data, "IPO", POLICY_TASK_COLS),
            "simpo_row": extract_scores_from_data(data, "SIMPO", POLICY_TASK_COLS),
            "rm_row": extract_scores_from_data(data, "RM", RM_COLS),
        }
    
    # Build dataframes
    def build_df(row_key, cols):
        data = {}
        for name, rows in parsed_results.items():
            if rows[row_key] is not None:
                data[name] = rows[row_key]
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame.from_dict(data, orient="index")
        return df[cols] if all(c in df.columns for c in cols) else df
    
    df_dpo = build_df("dpo_row", POLICY_TASK_COLS)
    df_ipo = build_df("ipo_row", POLICY_TASK_COLS)
    df_simpo = build_df("simpo_row", POLICY_TASK_COLS)
    df_rm = build_df("rm_row", RM_COLS)
    
    print(f"\n✓ DPO rows: {len(df_dpo)}, IPO rows: {len(df_ipo)}, SimPO rows: {len(df_simpo)}, RM rows: {len(df_rm)}\n")
    
    return df_dpo, df_ipo, df_simpo, df_rm


def write_latex_table(f, title, df_sft, df_base, df_results, cols):
    col_fmt = "l" + "c" * len(cols)
    f.write(f"\\section*{{{title}}}\n\\begin{{table}}[h]\n\\centering\n\\resizebox{{\\textwidth}}{{!}}{{\n\\begin{{tabular}}{{{col_fmt}}}\n\\toprule\n")
    
    # SFT
    lines = df_sft.to_latex(header=True, index=True).splitlines()
    f.write(lines[2] + "\n\\midrule\n" + lines[4] + "\n\\midrule\n\\midrule\n")
    
    # Baselines
    if not df_base.empty:
        body = df_base.to_latex(header=False, index=True, escape=False)
        for line in body.splitlines()[2:-2]:
            f.write(line + "\n")
            if line.strip().startswith("MaxMin"): f.write("\\midrule\n")
    
    # Results
    f.write("\\midrule\n")
    if not df_results.empty:
        body = df_results.to_latex(header=False, index=True, escape=False)
        for line in body.splitlines()[2:-2]: f.write(line + "\n")
    else: f.write("% No results\n")
    
    f.write(f"\\bottomrule\n\\end{{tabular}}\n}}\n\\caption{{{title}}}\n\\end{{table}}\n\n")

def save_latex(filename, packs):
    with open(filename, "w") as f:
        f.write("% Auto-generated LaTeX table from directory results\n\n")
        for t, s, b, w, c in packs:
            write_latex_table(f, t, s, b, w, c)
    print(f"✓ Generated {filename}")


def save_csv(filename, packs):
    """Save results as CSV files (one per section)."""
    base_name = filename.rsplit('.', 1)[0]
    
    def round_value(val):
        """Round numeric values to 3 decimal places."""
        if isinstance(val, (int, float)) and not np.isnan(val):
            return round(val, 3)
        return val
    
    for title, df_sft, df_base, df_results, cols in packs:
        section_name = title.lower().replace(" ", "_")
        csv_file = f"{base_name}_{section_name}.csv"
        
        # Combine all dataframes
        combined_rows = []
        
        # Add SFT row
        for idx in df_sft.index:
            row = {"Method": idx, "Type": "SFT"}
            for col in cols:
                if col in df_sft.columns:
                    row[col] = round_value(df_sft.at[idx, col])
            combined_rows.append(row)
        
        # Add baseline rows
        if not df_base.empty:
            for idx in df_base.index:
                row = {"Method": idx, "Type": "Baseline"}
                for col in cols:
                    if col in df_base.columns:
                        row[col] = round_value(df_base.at[idx, col])
                combined_rows.append(row)
        
        # Add result rows
        if not df_results.empty:
            for idx in df_results.index:
                row = {"Method": idx, "Type": "Active"}
                for col in cols:
                    if col in df_results.columns:
                        row[col] = round_value(df_results.at[idx, col])
                combined_rows.append(row)
        
        if combined_rows:
            df_combined = pd.DataFrame(combined_rows)
            df_combined.to_csv(csv_file, index=False)
            print(f"✓ Generated {csv_file}")

# ==============================================================================
#                               MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tables from directory-based results")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing result subdirectories")
    parser.add_argument("--output", type=str, default="tables_dir.tex", help="Output file (LaTeX or CSV)")
    parser.add_argument("--format", type=str, choices=["tex", "csv", "both"], default="tex", help="Output format: tex, csv, or both")
    parser.add_argument("--dpo_baseline_path", type=str, help="Path to DPO baseline models directory")
    parser.add_argument("--rm_baseline_path", type=str, help="Path to RM baseline models directory")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("Generating tables from directory results")
    print(f"{'='*70}\n")

    # Overwrite baselines if paths provided
    if args.dpo_baseline_path:
        overwrite_baseline_scores(args.dpo_baseline_path, DPO_TASK_BASELINES, DPO_MEAN_BASELINES, "DPO", POLICY_TASK_COLS)
    if args.rm_baseline_path:
        overwrite_baseline_scores(args.rm_baseline_path, RM_TASK_BASELINES, RM_MEAN_BASELINES, "RM", RM_COLS)

    SFT_BASE, OTHER_BASELINES_DELTAS = process_baselines()
    static_packs = get_static_data()
    
    dpo, ipo, simpo, rm = fetch_results_from_directory(args.results_dir)

    packs = [
        ("DPO Results", static_packs["DPO"][0], static_packs["DPO"][1], dpo, POLICY_TASK_COLS),
        ("IPO Results", static_packs["IPO"][0], static_packs["IPO"][1], ipo, POLICY_TASK_COLS),
        ("SimPO Results", static_packs["SIMPO"][0], static_packs["SIMPO"][1], simpo, POLICY_TASK_COLS),
        ("RM Results", static_packs["RM"][0], static_packs["RM"][1], rm, RM_COLS),
    ]
    
    # For CSV: use raw numeric values (before formatting)
    if args.format in ["csv", "both"]:
        save_csv(args.output, packs)
    
    # For LaTeX: format the dataframes
    if args.format in ["tex", "both"]:
        final_packs = [(t, s, process_section_dataframe(b, c), process_section_dataframe(w, c), c) for t, s, b, w, c in packs]
        save_latex(args.output, final_packs)
    
    print(f"\n{'='*70}")
    print(f"Done! Output: {args.output}")
    print(f"{'='*70}\n")
