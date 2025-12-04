import wandb
import pandas as pd
import argparse
import sys
import numpy as np
import pprint  # Added for pretty printing

# ==============================================================================
#                               CONFIGURATION
# ==============================================================================

DPO_COLS = ["GSM8K", "IF Eval", "Minerva Math", "Truthful QA", "Mean"]
RM_COLS = ["Factuality", "Focus", "Math", "Precise IF", "Safety", "Ties", "Mean"]

# 1. ABSOLUTE BASELINE (SFT Base Model)
SFT_BASE = {
    "DPO": {
        "GSM8K": 0.758,
        "IF Eval": 0.713,
        "Minerva Math": 0.309,
        "Truthful QA": 0.468,
        "Mean": 0.562,
    },
    "RM": {
        "Factuality": 0.316,
        "Focus": 0.277,
        "Math": 0.445,
        "Precise IF": 0.261,
        "Safety": 0.347,
        "Ties": 0.095,
        "Mean": 0.290,
    },
}

# 2. OTHER BASELINES (Deltas)
OTHER_BASELINES_DELTAS = {
    "DPO": {
        "Random": [+0.048, -0.011, +0.033, +0.053, +0.032],
        "UltraFeedback": [+0.045, -0.083, +0.039, +0.046, +0.012],
        "MaxMin": [+0.025, -0.018, +0.089, +0.152, +0.062],
        "DeltaQwen": [+0.064, +0.041, +0.068, +0.126, +0.074],
        "DeltaQwenBig": [+0.002, +0.013, +0.002, +0.013, +0.008],
    },
    "RM": {
        "Random": [+0.409, +0.252, +0.164, +0.104, +0.422, +0.228, +0.264],
        "Ultrafeedback": [+0.392, +0.183, +0.200, +0.064, +0.405, +0.288, +0.256],
        "MaxMin": [+0.373, +0.392, +0.156, +0.164, +0.367, +0.366, +0.303],
        "DeltaQwen": [+0.183, -0.050, +0.030, +0.077, +0.209, +0.109, +0.093],
        "DeltaQwenBig": [+0.137, +0.008, +0.071, +0.095, +0.077, -0.183, +0.034],
    },
}

KEY_MAPPING = {
    "DPO": {
        "GSM8K": "DPO/GSM8K",
        "IF Eval": "DPO/IF Eval",
        "Minerva Math": "DPO/Minerva Math",
        "Truthful QA": "DPO/Truthful QA",
        "Mean": "DPO/Mean",
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


def format_delta(val, is_best=False):
    if val is None or np.isnan(val):
        return "-"
    sign = "+" if val >= 0 else ""
    text = f"{sign}{val:.3f}"
    if is_best:
        return f"\\textbf{{{text}}}"
    return text


def format_sft(val):
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
            is_best = (val == max_val) and (not np.isnan(val))
            df_formatted.at[idx, col] = format_delta(val, is_best)

    return df_formatted


def get_static_data():
    dpo_sft = pd.DataFrame([SFT_BASE["DPO"]], index=["SFT Base Model"])
    for c in dpo_sft.columns:
        dpo_sft[c] = dpo_sft[c].apply(format_sft)

    rm_sft = pd.DataFrame([SFT_BASE["RM"]], index=["SFT Base Model"])
    for c in rm_sft.columns:
        rm_sft[c] = rm_sft[c].apply(format_sft)

    dpo_base_dict = {}
    for name, deltas in OTHER_BASELINES_DELTAS["DPO"].items():
        dpo_base_dict[name] = dict(zip(DPO_COLS, deltas))
    df_dpo_base_num = pd.DataFrame.from_dict(dpo_base_dict, orient="index")[DPO_COLS]

    rm_base_dict = {}
    for name, deltas in OTHER_BASELINES_DELTAS["RM"].items():
        rm_base_dict[name] = dict(zip(RM_COLS, deltas))
    df_rm_base_num = pd.DataFrame.from_dict(rm_base_dict, orient="index")[RM_COLS]

    return (dpo_sft, df_dpo_base_num), (rm_sft, df_rm_base_num)


def fetch_wandb_runs(entity, project, sweep_id):
    api = wandb.Api()
    print(f"Fetching runs for Sweep: {sweep_id}...")
    runs = api.runs(f"{entity}/{project}", filters={"sweep": sweep_id})

    dpo_data = {}
    rm_data = {}

    for i, run in enumerate(runs):
        # ---------------------------------------------------------
        # DEBUG: PRINT CONFIG
        # ---------------------------------------------------------
        print(f"\n[{i}] Processing Run: {run.name}")
        # pprint.pprint(run.config) # Uncomment to see full config again if needed
        # ---------------------------------------------------------

        config = run.config

        # 1. Outer Loop Batch Size (Top level)
        obs = config.get("outer_loop_batch_size", "?")

        # 2. Effective Batch Size (Inside 'enn' dict or flat key)
        # Try finding it in config['enn']['effective_batch_size']
        if "enn" in config and isinstance(config["enn"], dict):
            ebs = config["enn"].get("effective_batch_size", "?")
        else:
            # Fallback to dot notation or flat key if W&B flattened it
            ebs = config.get(
                "enn.effective_batch_size", config.get("effective_batch_size", "?")
            )

        # 3. Regularization Params (Inside 'enn' -> 'regularization')
        iv = "?"
        edb = "?"

        # Access path: enn -> regularization
        if "enn" in config and isinstance(config["enn"], dict):
            enn_cfg = config["enn"]
            if "regularization" in enn_cfg and isinstance(
                enn_cfg["regularization"], dict
            ):
                reg_cfg = enn_cfg["regularization"]
                iv = reg_cfg.get("initial_value", "?")
                edb = reg_cfg.get("exponential_decay_base", "?")

        # Fallback for flattened keys (often happens in sweeps)
        if iv == "?":
            iv = config.get("enn.regularization.initial_value", "?")
        if edb == "?":
            edb = config.get("enn.regularization.exponential_decay_base", "?")

        # Construct readable name
        name = f"OBS:{obs}, EBS:{ebs}, IV:{iv}, EDB:{edb}"
        name = name.replace("_", "\\_")  # Escape for LaTeX

        # --- DPO Data Collection ---
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
        if has_dpo:
            dpo_data[name] = dpo_row

        # --- RM Data Collection ---
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
        if has_rm:
            rm_data[name] = rm_row

    df_dpo_wandb = pd.DataFrame.from_dict(dpo_data, orient="index")
    if not df_dpo_wandb.empty:
        df_dpo_wandb = df_dpo_wandb[DPO_COLS]
        df_dpo_wandb.sort_index(inplace=True)

    df_rm_wandb = pd.DataFrame.from_dict(rm_data, orient="index")
    if not df_rm_wandb.empty:
        df_rm_wandb = df_rm_wandb[RM_COLS]
        df_rm_wandb.sort_index(inplace=True)

    return df_dpo_wandb, df_rm_wandb


def write_latex_table(f, title, df_sft, df_base_fmt, df_wandb_fmt):
    num_cols = len(DPO_COLS) if "DPO" in title else len(RM_COLS)
    col_fmt = "l" + "c" * num_cols

    f.write(f"\\section*{{{title}}}\n")
    f.write("\\begin{table}[h]\n\\centering\n")
    f.write("\\resizebox{\\textwidth}{!}{\n")
    f.write(f"\\begin{{tabular}}{{{col_fmt}}}\n")
    f.write("\\toprule\n")

    # SFT
    header_tex = df_sft.to_latex(header=True, index=True)
    lines = header_tex.splitlines()
    f.write(lines[2] + "\n")
    f.write("\\midrule\n")
    f.write(lines[4] + "\n")

    # Baselines
    f.write("\\midrule\n")
    if not df_base_fmt.empty:
        body = df_base_fmt.to_latex(header=False, index=True, escape=False)
        for line in body.splitlines()[2:-2]:
            f.write(line + "\n")
            if line.strip().startswith("MaxMin"):
                f.write("\\midrule\n")

    # WandB
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
    f.write(f"\\caption{{{title} (SFT: Absolute, Others: Deltas. Best in bold.)}}\n")
    f.write("\\end{table}\n\n")


def save_latex(filename, dpo_pack, rm_pack):
    dpo_sft, dpo_base_fmt, dpo_wandb_fmt = dpo_pack
    rm_sft, rm_base_fmt, rm_wandb_fmt = rm_pack

    with open(filename, "w") as f:
        f.write("% ========================================================\n")
        f.write("% REQUIRED PACKAGES: \\usepackage{booktabs}, \\usepackage{graphicx}\n")
        f.write("% ========================================================\n\n")

        write_latex_table(
            f, "DPO Evaluation Results", dpo_sft, dpo_base_fmt, dpo_wandb_fmt
        )
        write_latex_table(
            f, "Reward Model Evaluation Results", rm_sft, rm_base_fmt, rm_wandb_fmt
        )

    print(f"Successfully generated {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=str, required=True)
    parser.add_argument("--entity", type=str, default="ActiveUF")
    parser.add_argument("--project", type=str, default="loop")
    parser.add_argument("--output", type=str, default="tables.tex")
    args = parser.parse_args()

    (dpo_sft, dpo_base_num), (rm_sft, rm_base_num) = get_static_data()

    dpo_wandb_num, rm_wandb_num = fetch_wandb_runs(
        args.entity, args.project, args.sweep_id
    )

    dpo_base_fmt = process_section_dataframe(dpo_base_num, DPO_COLS)
    rm_base_fmt = process_section_dataframe(rm_base_num, RM_COLS)
    dpo_wandb_fmt = process_section_dataframe(dpo_wandb_num, DPO_COLS)
    rm_wandb_fmt = process_section_dataframe(rm_wandb_num, RM_COLS)

    save_latex(
        args.output,
        (dpo_sft, dpo_base_fmt, dpo_wandb_fmt),
        (rm_sft, rm_base_fmt, rm_wandb_fmt),
    )
