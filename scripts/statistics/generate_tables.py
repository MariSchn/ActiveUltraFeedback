import wandb
import pandas as pd
import argparse
import sys
import numpy as np
import json
import os
import csv
from collections import defaultdict

# ==============================================================================
#                               RAW BASELINE DATA
# ==============================================================================

# DPO RAW SCORES
DPO_TASK_BASELINES = {
    "GSM8K": { "SFT": 0.758, "DeltaQwen": 0.813, "MaxMin": 0.780, "Random": 0.782, "UltraFeedback": 0.795 },
    "IF Eval": { "SFT": 0.713, "DeltaQwen": 0.760, "MaxMin": 0.697, "Random": 0.741, "UltraFeedback": 0.712 },
    "Truthful QA": { "SFT": 0.468, "DeltaQwen": 0.598, "MaxMin": 0.618, "Random": 0.524, "UltraFeedback": 0.507 },
    "Alpaca Eval": { "SFT": 0.083, "DeltaQwen": 0.399, "MaxMin": 0.372, "Random": 0.160, "UltraFeedback": 0.155 },
}

# IPO RAW SCORES 
IPO_TASK_BASELINES = {
    "GSM8K": { "Random": 0.824, "MaxMin": 0.827, "DeltaQwen": 0.815, "UltraFeedback": 0.832 },
    "IF Eval": { "Random": 0.613, "MaxMin": 0.706, "DeltaQwen": 0.752, "UltraFeedback": 0.713 },
    "Truthful QA": { "Random": 0.580, "MaxMin": 0.595, "DeltaQwen": 0.492, "UltraFeedback": 0.517 },
    "Alpaca Eval": { "Random": 0.498, "MaxMin": 0.498, "DeltaQwen": 0.357, "UltraFeedback": 0.498 }
}

# SIMPO RAW SCORES
SIMPO_TASK_BASELINES = {
    "GSM8K": { "UltraFeedback": 0.796, "DeltaQwen": 0.821, "MaxMin": 0.764, "Random": 0.803 },
    "IF Eval": { "UltraFeedback": 0.670, "DeltaQwen": 0.731, "MaxMin": 0.654, "Random": 0.706 },
    "Truthful QA": { "UltraFeedback": 0.630, "DeltaQwen": 0.532, "MaxMin": 0.653, "Random": 0.600 },
    "Alpaca Eval": { "UltraFeedback": 0.650, "DeltaQwen": 0.517, "MaxMin": 0.543, "Random": 0.579 }
}

# Recalculated Means for DPO
DPO_MEAN_BASELINES = { "SFT": 0.506, "DeltaQwen": 0.643, "MaxMin": 0.617, "Random": 0.552, "UltraFeedback": 0.542 }

# RM RAW SCORES
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

# *** UPDATED MAPPING TO MATCH YOUR WANDB KEYS ***
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

ACQ_MAP = { "dts": "DTS", "infomax": "InfoMax", "maxminlcb": "MaxMinLCB", "drts": "DRTS", "deltaucb": "DeltaUCB" }

BENCHMARK_PATHS = {
    "GSM8K": "results/gsm8k_tulu/metrics.json",
    "IF Eval": "results/ifeval_tulu/metrics.json",
    "Truthful QA": "results/truthfulqa_tulu/metrics.json",
    "Alpaca Eval": "results/alpaca_eval/activeuf/leaderboard.csv",
}

# Globals
SFT_BASE = None
OTHER_BASELINES_DELTAS = None

# ==============================================================================
#                           DATA PROCESSING LOGIC
# ==============================================================================

def calculate_mean_for_dict(task_baselines, method):
    scores = []
    tasks = ["GSM8K", "IF Eval", "Truthful QA", "Alpaca Eval"]
    for t in tasks:
        val = task_baselines.get(t, {}).get(method, np.nan)
        if not np.isnan(val):
            scores.append(val)
    return np.mean(scores) if scores else np.nan

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
                    
                    print(f"    -> JSON keys found: {list(data.keys())[:10]}...")  # Debug
                    
                    for col in cols:
                        # Build comprehensive search keys
                        target_key = KEY_MAPPING["RM"].get(col)
                        col_lower = col.lower()
                        col_underscore = col.lower().replace(" ", "_")
                        
                        search_keys = []
                        if target_key:
                            search_keys.append(target_key)
                            search_keys.append(target_key.lower())  # lowercase version
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

def get_static_data(combined_mode=None, raw=False):
    static_packs = {}
    specs = [("DPO", POLICY_TASK_COLS), ("IPO", POLICY_TASK_COLS), ("SIMPO", POLICY_TASK_COLS), ("RM", RM_COLS)]

    for type_name, cols in specs:
        sft_df = pd.DataFrame([SFT_BASE[type_name]], index=["SFT Base Model"])
        sft_df = sft_df[cols]
        if not raw:
            for c in sft_df.columns: sft_df[c] = sft_df[c].apply(format_sft)
        
        base_dict = {}
        for name, deltas in OTHER_BASELINES_DELTAS[type_name].items():
            base_dict[name] = dict(zip(cols, deltas))
        df_base_num = pd.DataFrame.from_dict(base_dict, orient="index")[cols]
        static_packs[type_name] = (sft_df, df_base_num)

    combined_pack = None
    if combined_mode == 'all':
        comb_cols = ["DPO Mean", "RM Mean", "IPO Mean", "SimPO Mean", "Average"]
        keys_to_avg = ["DPO", "RM", "IPO", "SIMPO"]
    elif combined_mode == 'basic':
        comb_cols = ["DPO Mean", "RM Mean", "Average"]
        keys_to_avg = ["DPO", "RM"]
    else:
        return static_packs, None

    # SFT Combined
    sft_means = {}
    total_score = 0
    valid_count = 0
    for k in keys_to_avg:
        m = SFT_BASE[k]["Mean"]
        sft_means[f"{k if k != 'SIMPO' else 'SimPO'} Mean"] = m
        if not np.isnan(m):
            total_score += m
            valid_count += 1
    sft_means["Average"] = total_score / valid_count if valid_count > 0 else np.nan
    combined_sft = pd.DataFrame([sft_means], index=["SFT Base Model"])[comb_cols]
    if not raw:
        for c in combined_sft.columns: combined_sft[c] = combined_sft[c].apply(format_sft)

    # Baselines Combined
    combined_base_dict = {}
    for method in BASELINE_METHODS_ORDER:
        row_data = {}
        delta_sum = 0
        delta_count = 0
        for k in keys_to_avg:
            col_list = RM_COLS if k == "RM" else POLICY_TASK_COLS
            mean_idx = col_list.index("Mean")
            val = OTHER_BASELINES_DELTAS[k][method][mean_idx]
            row_data[f"{k if k != 'SIMPO' else 'SimPO'} Mean"] = val
            if not np.isnan(val):
                delta_sum += val
                delta_count += 1
        row_data["Average"] = delta_sum / delta_count if delta_count > 0 else np.nan
        combined_base_dict[method] = row_data

    df_combined_base_num = pd.DataFrame.from_dict(combined_base_dict, orient="index")[comb_cols]
    combined_pack = (combined_sft, df_combined_base_num, comb_cols)
    return static_packs, combined_pack

# --- WANDB HELPERS ---

def get_nested_config(config, key):
    if key in config: return config[key]
    parts = key.split('.')
    if len(parts) > 1:
        val = config
        for p in parts:
            if isinstance(val, dict) and p in val: val = val[p]
            else:
                val = None
                break
        if val is not None: return val
    return None

def generate_run_name(config):
    acq = ACQ_MAP.get(config.get("acquisition_function_type"), "Unknown")
    beta = get_nested_config(config, "acquisition_function.beta") or get_nested_config(config, "beta")
    decay = get_nested_config(config, "enn.regularization.exponential_decay_base") or get_nested_config(config, "exponential_decay_base")
    rb = get_nested_config(config, "replay_buffer_factor")
    
    beta_str = f"{beta}" if beta is not None else "None"
    decay_str = f"{decay}" if decay is not None else "None"
    rb_str = f"{rb}" if rb is not None else "None"
    return f"{acq} ($\\beta={beta_str}$, $d={decay_str}$, $rb={rb_str}$)"

def standardize_scores_per_group(parsed_runs, task_cols, type_key, row_key):
    grouped = defaultdict(list)
    for i, r in enumerate(parsed_runs):
        if r[row_key] is not None: grouped[r["acq_group"]].append(i)

    metrics = [c for c in task_cols if c != "Mean"]
    for acq, indices in grouped.items():
        for m in metrics:
            vals = [parsed_runs[idx][row_key].get(m, np.nan) for idx in indices]
            vals = [v for v in vals if not np.isnan(v)]
            if not vals: continue
            mu, sigma = np.mean(vals), np.std(vals)
            
            for idx in indices:
                orig = parsed_runs[idx][row_key].get(m, np.nan)
                if not np.isnan(orig):
                    parsed_runs[idx][row_key][m] = 0.0 if sigma == 0 else (orig - mu) / sigma
        
        for idx in indices:
            row_vals = [parsed_runs[idx][row_key].get(m, np.nan) for m in metrics]
            row_vals = [v for v in row_vals if not np.isnan(v)]
            parsed_runs[idx][row_key]["Mean"] = np.mean(row_vals) if row_vals else np.nan

def filter_top_n_runs(runs_data, sort_key_fn, top_n):
    if top_n is None: return runs_data
    grouped = defaultdict(list)
    for r in runs_data: grouped[r["acq_group"]].append(r)
    filtered = []
    for group in grouped.values():
        group.sort(key=lambda x: (sort_key_fn(x) if sort_key_fn(x) is not None and not np.isnan(sort_key_fn(x)) else -float("inf")), reverse=True)
        filtered.extend(group[:top_n])
    return filtered

def fetch_wandb_runs(entity, project, sweep_id, acq_filter_list=None, top_n=None, combined_mode=None, standardize=False):
    api = wandb.Api()
    print(f"Fetching runs for Sweep: {sweep_id}...")
    runs = api.runs(f"{entity}/{project}", filters={"sweep": sweep_id})
    parsed_runs = []

    for run in runs:
        acq_type = run.config.get("acquisition_function_type")
        if acq_filter_list and acq_type not in acq_filter_list: continue
        
        name = generate_run_name(run.config)
        acq_group = ACQ_MAP.get(acq_type, "Unknown")

        def extract_row(type_key, cols):
            row_data = {}
            has_data = False
            for col in cols:
                # Primary mapping check
                target_key = KEY_MAPPING.get(type_key, {}).get(col)
                val = run.summary.get(target_key)
                
                # Fallback: Check for other common patterns if primary fails
                if val is None:
                    candidates = [f"{type_key}/{col}", col]
                    for cand in candidates:
                        if cand in run.summary:
                            val = run.summary[cand]
                            break
                
                if val is not None:
                    has_data = True
                    try:
                        score = float(val)
                        base_val = SFT_BASE[type_key].get(col, np.nan)
                        row_data[col] = score - base_val if not np.isnan(base_val) else score
                    except: row_data[col] = np.nan
                else:
                    row_data[col] = np.nan
            return row_data if has_data else None

        parsed_runs.append({
            "name": name, "acq_group": acq_group,
            "dpo_row": extract_row("DPO", POLICY_TASK_COLS),
            "ipo_row": extract_row("IPO", POLICY_TASK_COLS),
            "simpo_row": extract_row("SIMPO", POLICY_TASK_COLS),
            "rm_row": extract_row("RM", RM_COLS),
        })

    if standardize:
        print("Standardizing scores...")
        for k, rk in [("DPO","dpo_row"), ("IPO","ipo_row"), ("SIMPO","simpo_row")]:
            standardize_scores_per_group(parsed_runs, POLICY_TASK_COLS, k, rk)
        standardize_scores_per_group(parsed_runs, RM_COLS, "RM", "rm_row")

    def get_df(type_key, row_key, cols):
        cands = [r for r in parsed_runs if r[row_key] is not None]
        final = filter_top_n_runs(cands, lambda x: x[row_key].get("Mean"), top_n)
        if not final: return pd.DataFrame()
        df = pd.DataFrame.from_dict({r["name"]: r[row_key] for r in final}, orient="index")
        return df[cols].sort_index()

    df_dpo = get_df("DPO", "dpo_row", POLICY_TASK_COLS)
    df_ipo = get_df("IPO", "ipo_row", POLICY_TASK_COLS)
    df_simpo = get_df("SIMPO", "simpo_row", POLICY_TASK_COLS)
    df_rm = get_df("RM", "rm_row", RM_COLS)

    df_comb = pd.DataFrame()
    comb_out = []
    if combined_mode:
        if combined_mode == 'all':
            keys = ["dpo_row", "rm_row", "ipo_row", "simpo_row"]
            comb_out = ["DPO Mean", "RM Mean", "IPO Mean", "SimPO Mean", "Average"]
        else:
            keys = ["dpo_row", "rm_row"]
            comb_out = ["DPO Mean", "RM Mean", "Average"]
            
        cands = [r for r in parsed_runs if all(r[k] is not None for k in keys)]
        def get_comb_mean(r):
            vals = [r[k].get("Mean") for k in keys if r[k].get("Mean") is not None]
            return np.mean(vals) if vals else np.nan
            
        final = filter_top_n_runs(cands, get_comb_mean, top_n)
        data = {}
        for r in final:
            row = {}
            vals = []
            if "dpo_row" in keys: 
                m = r["dpo_row"]["Mean"]; row["DPO Mean"] = m; vals.append(m)
            if "rm_row" in keys:
                m = r["rm_row"]["Mean"]; row["RM Mean"] = m; vals.append(m)
            if "ipo_row" in keys:
                m = r["ipo_row"]["Mean"]; row["IPO Mean"] = m; vals.append(m)
            if "simpo_row" in keys:
                m = r["simpo_row"]["Mean"]; row["SimPO Mean"] = m; vals.append(m)
            row["Average"] = np.mean(vals) if vals else np.nan
            data[r["name"]] = row
        
        if data: df_comb = pd.DataFrame.from_dict(data, orient="index")[comb_out].sort_index()

    return df_dpo, df_ipo, df_simpo, df_rm, df_comb, comb_out

def write_latex_table(f, title, df_sft, df_base, df_wandb, cols):
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
    
    # WandB
    f.write("\\midrule\n")
    if not df_wandb.empty:
        body = df_wandb.to_latex(header=False, index=True, escape=False)
        for line in body.splitlines()[2:-2]: f.write(line + "\n")
    else: f.write("% No WandB runs\n")
    
    f.write(f"\\bottomrule\n\\end{{tabular}}\n}}\n\\caption{{{title}}}\n\\end{{table}}\n\n")

def save_latex(filename, packs, combined_pack=None):
    with open(filename, "w") as f:
        f.write("% Preamble...\n\n")
        for t, s, b, w, c in packs: write_latex_table(f, t, s, b, w, c)
        if combined_pack:
            s, b, w, c = combined_pack
            suffix = "(DPO+RM+IPO+SimPO)" if "IPO" in c else "(DPO+RM)"
            write_latex_table(f, f"Combined {suffix}", s, b, w, c)
    print(f"Generated {filename}")

def save_csv(filename_base, packs, combined_pack=None):
    """Save results as CSV files. Creates one CSV per result type."""
    
    def round_numeric_columns(df):
        """Round all numeric columns to 3 decimal places."""
        df_rounded = df.copy()
        for col in df_rounded.columns:
            if col != 'Type' and df_rounded[col].dtype in ['float64', 'float32']:
                df_rounded[col] = df_rounded[col].round(3)
        return df_rounded
    
    for title, df_sft, df_base, df_wandb, cols in packs:
        # Create a sanitized filename from the title
        safe_title = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
        csv_filename = f"{filename_base}_{safe_title}.csv"
        
        # Combine all dataframes with a Type column
        combined_df = pd.DataFrame()
        
        # Add SFT row
        if not df_sft.empty:
            sft_copy = df_sft.copy()
            sft_copy.insert(0, 'Type', 'SFT')
            sft_copy.index.name = 'Method'
            combined_df = pd.concat([combined_df, sft_copy])
        
        # Add baseline rows
        if not df_base.empty:
            base_copy = df_base.copy()
            base_copy.insert(0, 'Type', 'Baseline')
            base_copy.index.name = 'Method'
            combined_df = pd.concat([combined_df, base_copy])
        
        # Add WandB rows
        if not df_wandb.empty:
            wandb_copy = df_wandb.copy()
            wandb_copy.insert(0, 'Type', 'Active')
            wandb_copy.index.name = 'Method'
            combined_df = pd.concat([combined_df, wandb_copy])
        
        if not combined_df.empty:
            combined_df = round_numeric_columns(combined_df)
            combined_df.to_csv(csv_filename)
            print(f"Generated {csv_filename}")
    
    # Save combined results if available
    if combined_pack:
        df_sft, df_base, df_wandb, cols = combined_pack
        suffix = "dpo_rm_ipo_simpo" if "IPO Mean" in cols else "dpo_rm"
        csv_filename = f"{filename_base}_combined_{suffix}.csv"
        
        combined_df = pd.DataFrame()
        
        if not df_sft.empty:
            sft_copy = df_sft.copy()
            sft_copy.insert(0, 'Type', 'SFT')
            sft_copy.index.name = 'Method'
            combined_df = pd.concat([combined_df, sft_copy])
        
        if not df_base.empty:
            base_copy = df_base.copy()
            base_copy.insert(0, 'Type', 'Baseline')
            base_copy.index.name = 'Method'
            combined_df = pd.concat([combined_df, base_copy])
        
        if not df_wandb.empty:
            wandb_copy = df_wandb.copy()
            wandb_copy.insert(0, 'Type', 'Active')
            wandb_copy.index.name = 'Method'
            combined_df = pd.concat([combined_df, wandb_copy])
        
        if not combined_df.empty:
            combined_df = round_numeric_columns(combined_df)
            combined_df.to_csv(csv_filename)
            print(f"Generated {csv_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=str, required=True)
    parser.add_argument("--entity", type=str, default="ActiveUF")
    parser.add_argument("--project", type=str, default="loop")
    parser.add_argument("--output", type=str, default="tables.tex")
    parser.add_argument("--acq_type", nargs="+")
    parser.add_argument("--top_n", type=int)
    parser.add_argument("--combined", action="store_true")
    parser.add_argument("--combined_all", action="store_true")
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--dpo_baseline_path", type=str)
    parser.add_argument("--rm_baseline_path", type=str)
    parser.add_argument("--csv", action="store_true", help="Save results as CSV instead of LaTeX")
    args = parser.parse_args()

    if args.dpo_baseline_path: overwrite_baseline_scores(args.dpo_baseline_path, DPO_TASK_BASELINES, DPO_MEAN_BASELINES, "DPO", POLICY_TASK_COLS)
    if args.rm_baseline_path: overwrite_baseline_scores(args.rm_baseline_path, RM_TASK_BASELINES, RM_MEAN_BASELINES, "RM", RM_COLS)

    SFT_BASE, OTHER_BASELINES_DELTAS = process_baselines()
    mode = 'all' if args.combined_all else ('basic' if args.combined else None)
    
    # Get static data - raw for CSV, formatted for LaTeX
    static_packs, comb_static = get_static_data(mode, raw=args.csv)
    
    dpo, ipo, simpo, rm, comb_w, c_cols = fetch_wandb_runs(
        args.entity, args.project, args.sweep_id, args.acq_type, args.top_n, mode, args.standardize
    )

    packs = [
        ("DPO Results", static_packs["DPO"][0], static_packs["DPO"][1], dpo, POLICY_TASK_COLS),
        ("IPO Results", static_packs["IPO"][0], static_packs["IPO"][1], ipo, POLICY_TASK_COLS),
        ("SimPO Results", static_packs["SIMPO"][0], static_packs["SIMPO"][1], simpo, POLICY_TASK_COLS),
        ("RM Results", static_packs["RM"][0], static_packs["RM"][1], rm, RM_COLS),
    ]
    
    if args.csv:
        # For CSV, use raw numeric data (unformatted)
        output_base = args.output.rsplit('.', 1)[0] if '.' in args.output else args.output
        
        # Build raw packs without LaTeX formatting
        raw_comb = None
        if mode and comb_static:
            raw_comb = (comb_static[0], comb_static[1], comb_w, c_cols)
        
        save_csv(output_base, packs, raw_comb)
    else:
        # For LaTeX, apply formatting
        final_comb = None
        if mode and comb_static:
            final_comb = (comb_static[0], process_section_dataframe(comb_static[1], c_cols), process_section_dataframe(comb_w, c_cols), c_cols)
            
        final_packs = [(t, s, process_section_dataframe(b, c), process_section_dataframe(w, c), c) for t, s, b, w, c in packs]
        save_latex(args.output, final_packs, final_comb)