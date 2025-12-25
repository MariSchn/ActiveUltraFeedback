import wandb
import pandas as pd
import argparse
import sys
import numpy as np
from collections import defaultdict

# ==============================================================================
#                               RAW BASELINE DATA
# ==============================================================================

# DPO RAW SCORES
DPO_TASK_BASELINES = {
    "GSM8K": {
        "SFT": 0.758,
        "DeltaQwen": 0.813,
        "MaxMin": 0.780,
        "Random": 0.782,
        "UltraFeedback": 0.795,
    },
    "IF Eval": {
        "SFT": 0.713,
        "DeltaQwen": 0.760,
        "MaxMin": 0.697,
        "Random": 0.741,
        "UltraFeedback": 0.712,
    },
    "Truthful QA": {
        "SFT": 0.468,
        "DeltaQwen": 0.598,
        "MaxMin": 0.618,
        "Random": 0.524,
        "UltraFeedback": 0.507,
    },
    "Alpaca Eval": {
        "SFT": 0.083,
        "DeltaQwen": 0.399,
        "MaxMin": 0.372,
        "Random": 0.160,
        "UltraFeedback": 0.155,
    },
}

# Recalculated Means
DPO_MEAN_BASELINES = {
    "SFT": 0.506,
    "DeltaQwen": 0.643,
    "MaxMin": 0.617,
    "Random": 0.552,
    "UltraFeedback": 0.542,
}

# RM RAW SCORES
RM_MEAN_BASELINES = {
    "SFT": 0.290,
    "DeltaQwen": 0.390,
    "MaxMin": 0.608,
    "Random": 0.568,
    "UltraFeedback": 0.577,
}

RM_TASK_BASELINES = {
    "Factuality": {
        "SFT": 0.316,
        "DeltaQwen": 0.511,
        "MaxMin": 0.693,
        "Random": 0.759,
        "UltraFeedback": 0.759,
    },
    "Focus": {
        "SFT": 0.277,
        "DeltaQwen": 0.243,
        "MaxMin": 0.760,
        "Random": 0.486,
        "UltraFeedback": 0.465,
    },
    "Math": {
        "SFT": 0.445,
        "DeltaQwen": 0.473,
        "MaxMin": 0.601,
        "Random": 0.601,
        "UltraFeedback": 0.658,
    },
    "Precise IF": {
        "SFT": 0.261,
        "DeltaQwen": 0.328,
        "MaxMin": 0.384,
        "Random": 0.394,
        "UltraFeedback": 0.375,
    },
    "Safety": {
        "SFT": 0.347,
        "DeltaQwen": 0.563,
        "MaxMin": 0.717,
        "Random": 0.764,
        "UltraFeedback": 0.828,
    },
    "Ties": {
        "SFT": 0.095,
        "DeltaQwen": 0.221,
        "MaxMin": 0.495,
        "Random": 0.405,
        "UltraFeedback": 0.379,
    },
}

# ==============================================================================
#                               CONFIGURATION
# ==============================================================================

DPO_COLS = ["GSM8K", "IF Eval", "Truthful QA", "Alpaca Eval", "Mean"]
RM_COLS = ["Factuality", "Focus", "Math", "Precise IF", "Safety", "Ties", "Mean"]
COMBINED_COLS = ["DPO Mean", "RM Mean", "Average"]

BASELINE_METHODS_ORDER = ["Random", "UltraFeedback", "MaxMin", "DeltaQwen"]

KEY_MAPPING = {
    "DPO": {
        "GSM8K": "DPO/GSM8K",
        "IF Eval": "DPO/IF Eval",
        "Truthful QA": "DPO/Truthful QA",
        "Mean": "DPO/Mean",
        "Alpaca Eval": "DPO/Alpaca Eval",
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

ACQ_MAP = {
    "dts": "DTS",
    "infomax": "InfoMax",
    "maxminlcb": "MaxMinLCB",
    "drts": "DRTS",
    "deltaucb": "DeltaUCB",
}

# ==============================================================================
#                           DATA PROCESSING LOGIC
# ==============================================================================


def process_baselines():
    sft_base = {"DPO": {}, "RM": {}}
    other_deltas = {"DPO": {}, "RM": {}}

    for method in BASELINE_METHODS_ORDER:
        other_deltas["DPO"][method] = []
        other_deltas["RM"][method] = []

    # --- PROCESS DPO ---
    for col in DPO_COLS:
        if col == "Mean":
            source = DPO_MEAN_BASELINES
        else:
            source = DPO_TASK_BASELINES.get(col, {})

        sft_val = source.get("SFT", np.nan)
        sft_base["DPO"][col] = sft_val

        for method in BASELINE_METHODS_ORDER:
            val = source.get(method, np.nan)
            delta = (
                val - sft_val if not np.isnan(val) and not np.isnan(sft_val) else np.nan
            )
            other_deltas["DPO"][method].append(delta)

    # --- PROCESS RM ---
    for col in RM_COLS:
        if col == "Mean":
            source = RM_MEAN_BASELINES
        else:
            source = RM_TASK_BASELINES.get(col, {})

        sft_val = source.get("SFT", np.nan)
        sft_base["RM"][col] = sft_val

        for method in BASELINE_METHODS_ORDER:
            val = source.get(method, np.nan)
            delta = (
                val - sft_val if not np.isnan(val) and not np.isnan(sft_val) else np.nan
            )
            other_deltas["RM"][method].append(delta)

    return sft_base, other_deltas


SFT_BASE, OTHER_BASELINES_DELTAS = process_baselines()


def format_delta(val, is_best=False):
    if val is None or np.isnan(val):
        return "-"
    sign = "+" if val >= 0 else ""
    text = f"{sign}{val:.3f}"
    if is_best:
        return f"\\textbf{{{text}}}"
    return text


def format_sft(val):
    if val is None or np.isnan(val):
        return "-"
    return f"{val:.3f}"


def process_section_dataframe(df_numeric, cols):
    if df_numeric.empty:
        return pd.DataFrame()

    df_formatted = df_numeric.copy()
    max_series = df_numeric.max(numeric_only=True)

    for col in cols:
        if col not in df_numeric.columns:
            continue
        max_val = max_series[col]

        for idx in df_numeric.index:
            val = df_numeric.at[idx, col]
            # Highlight best if it's the max and not NaN
            is_best = (val == max_val) and (not np.isnan(val))
            df_formatted.at[idx, col] = format_delta(val, is_best)

    return df_formatted


def get_static_data(include_combined=False):
    # 1. SFT
    dpo_sft = pd.DataFrame([SFT_BASE["DPO"]], index=["SFT Base Model"])
    dpo_sft = dpo_sft[DPO_COLS]
    for c in dpo_sft.columns:
        dpo_sft[c] = dpo_sft[c].apply(format_sft)

    rm_sft = pd.DataFrame([SFT_BASE["RM"]], index=["SFT Base Model"])
    rm_sft = rm_sft[RM_COLS]
    for c in rm_sft.columns:
        rm_sft[c] = rm_sft[c].apply(format_sft)

    # 2. Baselines
    dpo_base_dict = {}
    for name, deltas in OTHER_BASELINES_DELTAS["DPO"].items():
        dpo_base_dict[name] = dict(zip(DPO_COLS, deltas))
    df_dpo_base_num = pd.DataFrame.from_dict(dpo_base_dict, orient="index")[DPO_COLS]

    rm_base_dict = {}
    for name, deltas in OTHER_BASELINES_DELTAS["RM"].items():
        rm_base_dict[name] = dict(zip(RM_COLS, deltas))
    df_rm_base_num = pd.DataFrame.from_dict(rm_base_dict, orient="index")[RM_COLS]

    # 3. Combined (Optional)
    combined_pack = None
    if include_combined:
        # SFT Combined Absolute
        sft_dpo_mean = SFT_BASE["DPO"]["Mean"]
        sft_rm_mean = SFT_BASE["RM"]["Mean"]
        sft_combined_mean = (sft_dpo_mean + sft_rm_mean) / 2

        combined_sft = pd.DataFrame(
            [
                {
                    "DPO Mean": sft_dpo_mean,
                    "RM Mean": sft_rm_mean,
                    "Average": sft_combined_mean,
                }
            ],
            index=["SFT Base Model"],
        )
        for c in combined_sft.columns:
            combined_sft[c] = combined_sft[c].apply(format_sft)

        # Baselines Combined Deltas
        combined_base_dict = {}
        for method in BASELINE_METHODS_ORDER:
            # We need the "Mean" column index from the list
            dpo_mean_idx = DPO_COLS.index("Mean")
            rm_mean_idx = RM_COLS.index("Mean")

            dpo_delta = OTHER_BASELINES_DELTAS["DPO"][method][dpo_mean_idx]
            rm_delta = OTHER_BASELINES_DELTAS["RM"][method][rm_mean_idx]

            avg_delta = np.nan
            if not np.isnan(dpo_delta) and not np.isnan(rm_delta):
                avg_delta = (dpo_delta + rm_delta) / 2

            combined_base_dict[method] = {
                "DPO Mean": dpo_delta,
                "RM Mean": rm_delta,
                "Average": avg_delta,
            }

        df_combined_base_num = pd.DataFrame.from_dict(
            combined_base_dict, orient="index"
        )[COMBINED_COLS]
        combined_pack = (combined_sft, df_combined_base_num)

    return (dpo_sft, df_dpo_base_num), (rm_sft, df_rm_base_num), combined_pack


def generate_run_name(config):
    acq = ACQ_MAP.get(config.get("acquisition_function_type"), "Unknown")
    beta = config.get("acquisition_function.beta")
    decay = config.get("enn.regularization.exponential_decay_base")
    rb = config.get("replay_buffer_factor")
    name = f"{acq} ($\\beta={beta}$, $d={decay}$, $rb={rb}$)"
    return name


def filter_top_n_runs(runs_data, sort_key_fn, top_n):
    """
    Groups runs by acquisition function, sorts them using sort_key_fn,
    and keeps the top N per group.
    """
    if top_n is None:
        return runs_data

    grouped = defaultdict(list)
    for r in runs_data:
        grouped[r["acq_group"]].append(r)

    filtered = []
    for acq, group_list in grouped.items():
        # Sort descending by the provided key function
        # We handle NaNs/None by using a very small number
        group_list.sort(
            key=lambda x: (
                sort_key_fn(x)
                if sort_key_fn(x) is not None and not np.isnan(sort_key_fn(x))
                else -float("inf")
            ),
            reverse=True,
        )
        filtered.extend(group_list[:top_n])
    return filtered


def fetch_wandb_runs(
    entity, project, sweep_id, acq_filter_list=None, top_n=None, combined=False
):
    api = wandb.Api()
    print(f"Fetching runs for Sweep: {sweep_id}...")
    runs = api.runs(f"{entity}/{project}", filters={"sweep": sweep_id})

    # Collect all parsed runs first
    parsed_runs = []

    for run in runs:
        run_acq_type = run.config.get("acquisition_function_type")
        if acq_filter_list and run_acq_type not in acq_filter_list:
            continue

        name = generate_run_name(run.config)
        acq_group_name = ACQ_MAP.get(run_acq_type, "Unknown")

        # --- EXTRACT DPO ---
        dpo_row = {}
        has_dpo = False
        for col in DPO_COLS:
            keys = [
                KEY_MAPPING["DPO"][col],
                f"DPO/{col}",
                f"DPO/{col.replace(' ', '')}",
            ]
            score = next((run.summary[k] for k in keys if k in run.summary), None)
            if score is not None:
                has_dpo = True
                dpo_row[col] = score - SFT_BASE["DPO"][col]
            else:
                dpo_row[col] = np.nan

        # --- EXTRACT RM ---
        rm_row = {}
        has_rm = False
        for col in RM_COLS:
            keys = [
                KEY_MAPPING["RM"][col],
                f"Rewardbench/{col}",
                f"Rewardbench/{col.lower()}",
            ]
            score = next((run.summary[k] for k in keys if k in run.summary), None)
            if score is not None:
                has_rm = True
                rm_row[col] = score - SFT_BASE["RM"][col]
            else:
                rm_row[col] = np.nan

        parsed_runs.append(
            {
                "name": name,
                "acq_group": acq_group_name,
                "dpo_row": dpo_row if has_dpo else None,
                "rm_row": rm_row if has_rm else None,
            }
        )

    # =========================================================
    # INDEPENDENT FILTERING
    # =========================================================

    # 1. DPO LIST
    dpo_candidates = [r for r in parsed_runs if r["dpo_row"] is not None]
    dpo_final = filter_top_n_runs(
        dpo_candidates, lambda x: x["dpo_row"].get("Mean"), top_n
    )

    # 2. RM LIST
    rm_candidates = [r for r in parsed_runs if r["rm_row"] is not None]
    rm_final = filter_top_n_runs(
        rm_candidates, lambda x: x["rm_row"].get("Mean"), top_n
    )

    # 3. COMBINED LIST (Optional)
    combined_final = []
    if combined:
        # Candidate must have both DPO and RM data
        combined_candidates = [
            r
            for r in parsed_runs
            if r["dpo_row"] is not None and r["rm_row"] is not None
        ]

        # Helper to compute combined mean
        def get_combined_mean(r):
            dpo_m = r["dpo_row"].get("Mean")
            rm_m = r["rm_row"].get("Mean")
            if (
                dpo_m is not None
                and rm_m is not None
                and not np.isnan(dpo_m)
                and not np.isnan(rm_m)
            ):
                return (dpo_m + rm_m) / 2
            return np.nan

        combined_final = filter_top_n_runs(
            combined_candidates, get_combined_mean, top_n
        )

    # =========================================================
    # BUILD DATAFRAMES
    # =========================================================

    # Build DPO DataFrame
    dpo_dict = {r["name"]: r["dpo_row"] for r in dpo_final}
    df_dpo_wandb = pd.DataFrame.from_dict(dpo_dict, orient="index")
    if not df_dpo_wandb.empty:
        df_dpo_wandb = df_dpo_wandb[DPO_COLS]
        df_dpo_wandb = df_dpo_wandb.sort_index()

    # Build RM DataFrame
    rm_dict = {r["name"]: r["rm_row"] for r in rm_final}
    df_rm_wandb = pd.DataFrame.from_dict(rm_dict, orient="index")
    if not df_rm_wandb.empty:
        df_rm_wandb = df_rm_wandb[RM_COLS]
        df_rm_wandb = df_rm_wandb.sort_index()

    # Build Combined DataFrame
    df_combined_wandb = pd.DataFrame()
    if combined:
        combined_data = {}
        for r in combined_final:
            dpo_m = r["dpo_row"]["Mean"]
            rm_m = r["rm_row"]["Mean"]
            avg = (dpo_m + rm_m) / 2
            combined_data[r["name"]] = {
                "DPO Mean": dpo_m,
                "RM Mean": rm_m,
                "Average": avg,
            }
        df_combined_wandb = pd.DataFrame.from_dict(combined_data, orient="index")
        if not df_combined_wandb.empty:
            df_combined_wandb = df_combined_wandb[COMBINED_COLS]
            df_combined_wandb = df_combined_wandb.sort_index()

    return df_dpo_wandb, df_rm_wandb, df_combined_wandb


def write_latex_table(f, title, df_sft, df_base_fmt, df_wandb_fmt, cols):
    num_cols = len(cols)
    col_fmt = "l" + "c" * num_cols

    f.write(f"\\section*{{{title}}}\n")
    f.write("\\begin{table}[h]\n\\centering\n")
    f.write("\\resizebox{\\textwidth}{!}{\n")
    f.write(f"\\begin{{tabular}}{{{col_fmt}}}\n")
    f.write("\\toprule\n")

    # 1. SFT Row
    header_tex = df_sft.to_latex(header=True, index=True)
    lines = header_tex.splitlines()
    f.write(lines[2] + "\n")
    f.write("\\midrule\n")
    f.write(lines[4] + "\n")

    # 2. Baselines
    f.write("\\midrule\n")
    if not df_base_fmt.empty:
        body = df_base_fmt.to_latex(header=False, index=True, escape=False)
        for line in body.splitlines()[2:-2]:
            f.write(line + "\n")
            if line.strip().startswith("MaxMin"):
                f.write("\\midrule\n")

    # 3. WandB Runs
    f.write("\\midrule\n")
    if not df_wandb_fmt.empty:
        body = df_wandb_fmt.to_latex(header=False, index=True, escape=False)
        for line in body.splitlines()[2:-2]:
            f.write(line + "\n")
    else:
        f.write("% No WandB runs found\n")

    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("}\n")
    f.write(
        f"\\caption{{{title} (SFT: Absolute, Others: Deltas. Top {args.top_n if args.top_n else 'All'} runs per Acq sorted by Mean. Best in bold.)}}\n"
    )
    f.write("\\end{table}\n\n")


def save_latex(filename, dpo_pack, rm_pack, combined_pack=None):
    dpo_sft, dpo_base_fmt, dpo_wandb_fmt = dpo_pack
    rm_sft, rm_base_fmt, rm_wandb_fmt = rm_pack

    with open(filename, "w") as f:
        f.write("% ========================================================\n")
        f.write("% REQUIRED PACKAGES: \\usepackage{booktabs}, \\usepackage{graphicx}\n")
        f.write("% ========================================================\n\n")

        write_latex_table(
            f, "DPO Evaluation Results", dpo_sft, dpo_base_fmt, dpo_wandb_fmt, DPO_COLS
        )
        write_latex_table(
            f,
            "Reward Model Evaluation Results",
            rm_sft,
            rm_base_fmt,
            rm_wandb_fmt,
            RM_COLS,
        )

        if combined_pack:
            comb_sft, comb_base_fmt, comb_wandb_fmt = combined_pack
            write_latex_table(
                f,
                "Combined Average Scores",
                comb_sft,
                comb_base_fmt,
                comb_wandb_fmt,
                COMBINED_COLS,
            )

    print(f"Successfully generated {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=str, required=True)
    parser.add_argument("--entity", type=str, default="ActiveUF")
    parser.add_argument("--project", type=str, default="loop")
    parser.add_argument("--output", type=str, default="tables.tex")
    parser.add_argument(
        "--acq_type",
        type=str,
        nargs="+",
        default=None,
        help="Filter results by one or more acquisition function types (e.g. dts infomax)",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=None,
        help="If provided, output only the top N runs (independently sorted by DPO Mean, RM Mean, and Combined Mean) per acquisition function.",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="If set, generate a third table averaging the DPO Mean and RM Mean scores.",
    )

    args = parser.parse_args()

    (dpo_sft, dpo_base_num), (rm_sft, rm_base_num), combined_data = get_static_data(
        args.combined
    )

    dpo_wandb_num, rm_wandb_num, combined_wandb_num = fetch_wandb_runs(
        args.entity,
        args.project,
        args.sweep_id,
        args.acq_type,
        args.top_n,
        args.combined,
    )

    dpo_base_fmt = process_section_dataframe(dpo_base_num, DPO_COLS)
    rm_base_fmt = process_section_dataframe(rm_base_num, RM_COLS)
    dpo_wandb_fmt = process_section_dataframe(dpo_wandb_num, DPO_COLS)
    rm_wandb_fmt = process_section_dataframe(rm_wandb_num, RM_COLS)

    combined_pack_fmt = None
    if args.combined:
        comb_sft_num, comb_base_num = combined_data
        comb_base_fmt = process_section_dataframe(comb_base_num, COMBINED_COLS)
        comb_wandb_fmt = process_section_dataframe(combined_wandb_num, COMBINED_COLS)
        combined_pack_fmt = (comb_sft_num, comb_base_fmt, comb_wandb_fmt)

    save_latex(
        args.output,
        (dpo_sft, dpo_base_fmt, dpo_wandb_fmt),
        (rm_sft, rm_base_fmt, rm_wandb_fmt),
        combined_pack_fmt,
    )
