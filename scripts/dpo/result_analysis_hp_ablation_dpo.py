import os
import json
import argparse
import pandas as pd
import yaml
import numpy as np

# --- CONFIGURATION ---
SFT_SCORES = {
    "gsm8k": 0.7589,
    "ifeval": 0.7153,
    "truthfulqa": 0.4676,
    "minerva_math": 0.3092,
}


def load_configs(models_dir):
    """
    Reads dpo_training.yaml from every folder in models_dir.
    Returns: { folder_name: {'Epochs': int, 'Seed': int} }
    """
    config_map = {}
    if not os.path.exists(models_dir):
        print(f"Error: Models dir {models_dir} not found.")
        return config_map

    print(f"--- 1. Scanning Configs in {models_dir} ---")

    for folder_name in os.listdir(models_dir):
        folder_path = os.path.join(models_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        yaml_path = os.path.join(folder_path, "dpo_training.yaml")
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, "r") as f:
                    data = yaml.safe_load(f)

                # 1. Get Seed
                seed = data.get("seed", -1)

                # 2. Get Epochs (Look in 'training' block first, then top level)
                epochs = None
                if "training" in data and isinstance(data["training"], dict):
                    epochs = data["training"].get("num_train_epochs")

                if epochs is None:
                    epochs = data.get("num_train_epochs")

                # Fallback to num_epochs if specific key missing
                if epochs is None:
                    epochs = data.get("num_epochs")

                if epochs is not None:
                    config_map[folder_name] = {
                        "Epochs": int(epochs)
                        if float(epochs).is_integer()
                        else float(epochs),
                        "Seed": int(seed),
                    }
            except Exception as e:
                pass  # Silently skip malformed yaml

    print(f"-> Loaded configs for {len(config_map)} folders.\n")
    return config_map


def load_results(results_dir, config_map):
    """
    Iterates tasks in results_dir, matches exact folder names to config_map.
    """
    records = []
    if not os.path.exists(results_dir):
        print(f"Error: Results dir {results_dir} not found.")
        return pd.DataFrame()

    print(f"--- 2. Scanning Results in {results_dir} ---")

    task_folders = [
        d
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ]

    for task_folder in task_folders:
        clean_task = task_folder.replace("_tulu", "").replace("::tulu", "")
        if clean_task not in SFT_SCORES:
            continue

        task_path = os.path.join(results_dir, task_folder)

        for model_folder in os.listdir(task_path):
            # STRICT MATCHING: Folder name in results must match folder name in models
            if model_folder not in config_map:
                continue

            cfg = config_map[model_folder]
            model_path = os.path.join(task_path, model_folder)

            # Find metrics.json in subfolders
            subdirs = [
                d
                for d in os.listdir(model_path)
                if os.path.isdir(os.path.join(model_path, d))
            ]
            if not subdirs:
                continue

            subdirs.sort()  # Get latest timestamp
            metrics_path = os.path.join(model_path, subdirs[-1], "metrics.json")

            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, "r") as f:
                        data = json.load(f)

                    if "all_primary_scores" in data:
                        for s_str in data["all_primary_scores"]:
                            parts = s_str.split()
                            if len(parts) >= 2:
                                val = float(parts[-1])
                                delta = val - SFT_SCORES[clean_task]

                                records.append(
                                    {
                                        "Folder": model_folder,
                                        "Epochs": cfg["Epochs"],
                                        "Seed": cfg["Seed"],
                                        "Task": clean_task,
                                        "Score": delta,
                                    }
                                )
                                break
                except Exception:
                    pass

    return pd.DataFrame(records)


def write_latex(f, df, caption, is_summary=False):
    # --- CRITICAL FIX: Deduplicate columns ---
    df = df.loc[:, ~df.columns.duplicated()]
    cols = list(df.columns)

    # Layout
    col_def = "l" * (len(cols) - 4) + "c" * 4
    if len(cols) < 4:
        col_def = "l" * len(cols)

    f.write("\\begin{table}[h]\n\\centering\n\\resizebox{\\textwidth}{!}{\n")
    f.write(f"\\begin{{tabular}}{{{col_def}}}\n")
    f.write("\\toprule\n")

    headers = [c.title() for c in cols]
    f.write(" & ".join(headers) + " \\\\\n")
    f.write("\\midrule\n")

    for _, row in df.iterrows():
        line = []
        for c in cols:
            val = row[c]

            # --- CRITICAL FIX: Handle Series collision ---
            if isinstance(val, pd.Series):
                val = val.iloc[0]

            if isinstance(val, (float, int)) and not isinstance(val, bool):
                if is_summary:
                    # If it's the Epochs column in summary, we want int formatting
                    if c == "Epochs":
                        line.append(f"{val:.0f}")
                    else:
                        # Otherwise it's a string "Mean +/- Std" (which shouldn't be float type here)
                        # or a raw mean value if formatting missed it.
                        # Usually summary strings enter 'else' block below.
                        line.append(str(val))
                else:
                    # Individual Table Formatting
                    if c in ["Epochs", "Seed"]:
                        line.append(f"{val:.0f}")
                    else:
                        line.append(f"{val:+.3f}")
            elif pd.isna(val):
                line.append("-")
            else:
                line.append(str(val))
        f.write(" & ".join(line) + " \\\\\n")

    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n}\n")
    f.write(f"\\caption{{{caption}}}\n")
    f.write("\\end{table}\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--output_file", default="dpo_summary.tex")
    args = parser.parse_args()

    # 1. Load Data
    config_map = load_configs(args.models_dir)
    if not config_map:
        print("No configs found. Exiting.")
        return

    df = load_results(args.results_dir, config_map)
    if df.empty:
        print("No matching results found.")
        return

    # 2. Pivot
    # One row per (Folder, Epochs, Seed)
    pivot_df = df.pivot_table(
        index=["Folder", "Epochs", "Seed"],
        columns="Task",
        values="Score",
        aggfunc="first",
    ).reset_index()

    # 3. Calculate Mean
    # Ensure distinct task columns
    task_cols = sorted(list(set([c for c in SFT_SCORES if c in pivot_df.columns])))

    if not task_cols:
        print("Error: No valid task columns found.")
        return

    pivot_df["Mean"] = pivot_df[task_cols].mean(axis=1)

    print(f"--- Data Ready: {len(pivot_df)} Rows ---")
    print(pivot_df[["Epochs", "Seed", "Mean"]])

    with open(args.output_file, "w") as f:
        f.write("% DPO Summary Tables\n")

        # --- Table 1: Individual Runs ---
        ind_df = pivot_df.sort_values(by=["Epochs", "Folder"])
        ind_cols = ["Epochs", "Seed"] + task_cols + ["Mean"]

        # Ensure columns exist before slicing
        ind_cols = [c for c in ind_cols if c in ind_df.columns]

        write_latex(f, ind_df[ind_cols], "Individual Runs", is_summary=False)

        # --- Table 2: Aggregated Summary ---
        agg_df = (
            pivot_df.groupby("Epochs")[task_cols + ["Mean"]]
            .agg(["mean", "std"])
            .reset_index()
        )

        summary_rows = []
        for _, row in agg_df.iterrows():
            new_row = {"Epochs": row["Epochs"]}
            for t in task_cols + ["Mean"]:
                m = row[(t, "mean")]
                s = row[(t, "std")]
                # Handle single-run cases where std is NaN
                if pd.isna(s):
                    s = 0.0
                new_row[t] = f"{m:+.3f}_{{ \\pm {s:.3f} }}"
            summary_rows.append(new_row)

        summary_df = pd.DataFrame(summary_rows)
        summary_cols = ["Epochs"] + task_cols + ["Mean"]

        write_latex(
            f,
            summary_df[summary_cols],
            "Aggregated Results (Mean $\\pm$ Std)",
            is_summary=True,
        )

    print(f"Done. Saved to {args.output_file}")


if __name__ == "__main__":
    main()
