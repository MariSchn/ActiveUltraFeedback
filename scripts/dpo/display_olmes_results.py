from argparse import ArgumentParser
import os
import json


def collect_results(results_dir):
    tasks = [
        d
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ]
    print(tasks)
    task_keys = ["_".join(d.split("_")[:-1]) + "::tulu" for d in tasks]
    print(task_keys)
    results_dict = {}
    for idx, task in enumerate(tasks):
        task_dir = os.path.join(results_dir, task)
        results = os.listdir(task_dir)
        for result in results:
            if result not in results_dict:
                results_dict[result] = {}
            result_path = os.path.join(task_dir, result)
            subresult = os.listdir(result_path)
            result_path = os.path.join(result_path, subresult[0])
            result_path = os.path.join(result_path, "metrics.json")
            if os.path.exists(result_path):
                with open(result_path, "r") as f:
                    metrics = json.load(f)
                    for scores in metrics["all_primary_scores"]:
                        if task_keys[idx] in scores:
                            results_dict[result]["_".join(task.split("_")[:-1])] = (
                                scores.split(" ")[-1]
                            )
            else:
                results_dict[result]["_".join(task.split("_")[:-1])] = "-"

    baselines = {
        "Llama-3.1-Tulu-3-8B-SFT": {
            "ifeval": "0.715342",
            "truthfulqa": "0.467636",
            "minerva_math": "0.309218",
            "gsm8k": "0.758908",
        },
        "combined_MaxMin": {
            "ifeval": "0.726",
            "truthfulqa": "0.617",
            "minerva_math": "0.348",
            "gsm8k": "0.800",
        },
        "combined_Random": {
            "ifeval": "0.738",
            "truthfulqa": "0.520",
            "minerva_math": "0.327",
            "gsm8k": "0.794",
        },
        "combined_Ultrafeedback": {
            "ifeval": "0.729",
            "truthfulqa": "0.519",
            "minerva_math": "0.329",
            "gsm8k": "0.791",
        },
        "Llama-3.1-Tulu-3-8B-DPO": {
            "ifeval": "0.815157",
            "truthfulqa": "0.561898",
            "minerva_math": "0.425645",
            "gsm8k": "0.842305",
        },
    }

    return results_dict, baselines


def process_name(name):
    splitted = name.split("_")

    name = ""
    for idx, part in enumerate(splitted):
        if part in ["qwen"]:
            continue
        part = part.replace("-", ".")
        if "." in part and int(part.split(".")[-1]) == 0:
            part = part.split(".")[0]
        try:
            float(part)
            continue
        except ValueError:
            pass
        i = 0
        for char in part:
            if char.isdigit():
                break
            i += 1
        partFirst = part[:i]
        if partFirst == "rgl":
            partFirst = "lambda"
        elif partFirst == "wdcb":
            partFirst = "base"
        elif partFirst == "obs":
            partFirst = "batch"
        name += (
            partFirst
            + (":" + part[i:] if i < len(part) else "")
            # + ("," if idx != len(splitted) - 1 else "")
            + " "
        )

    return name


def display_results(results_dict, baselines, delta=False):
    if delta:
        # overwrite result_dict scores with deltas from the SFT model scores.
        sft_model = "Llama-3.1-Tulu-3-8B-SFT"
        sft_scores = baselines[sft_model]
        for model, results in results_dict.items():
            for task, score in results.items():
                if score != "-":
                    results_dict[model][task] = float(score) - float(sft_scores[task])
        for model, results in baselines.items():
            if model == sft_model:
                continue
            for task, score in results.items():
                baselines[model][task] = float(score) - float(sft_scores[task])

    baseline = baselines["Llama-3.1-Tulu-3-8B-SFT"]
    tex_string = f"""
\\begin{{table*}}[ht]
\\captionsetup{{font={{\\fontsize{{9.5}}{{11}}\\selectfont}}}}
\\caption{{result table}}
\\label{{tab:delta-results}}
\\vspace{{0.1em}}
\\small
\\setlength{{\\tabcolsep}}{{5pt}}
\\renewcommand{{\\arraystretch}}{{1.15}}
\\begin{{tabular}}{{@{{}}l{"c" * (len(baselines["Llama-3.1-Tulu-3-8B-SFT"]) + 1)}@{{}}}}
\\toprule
\\textbf{{Model}} {" ".join(["& \\textbf{" + "\_".join(task.split("_")) + "}" for task in baseline.keys()])} & \\textbf{{Average}} \\\\
\\midrule
\\multicolumn{{{(len(baselines["Llama-3.1-Tulu-3-8B-SFT"]) + 2)}}}{{l}}{{ \\textbf{{Active Models}} }} \\\\
\\midrule
"""
    # sort according to key names
    results_dict = dict(
        sorted(results_dict.items(), key=lambda item: process_name(item[0]))
    )
    for model, results in results_dict.items():
        name = process_name(model)
        tex_string += f"""
{name} {" ".join([f"& {'+' if delta and score != '-' and float(score) > 0 else ''}{'-' if score == '-' else f'{float(score):.3f}'}" for score in results.values()])} & {sum(float(score) for score in results.values() if score != "-") / len([res for res in results.values() if res != "-"]):.3f} \\\\"""
    tex_string += f"""
\\midrule
\\multicolumn{{{(len(baselines["Llama-3.1-Tulu-3-8B-SFT"]) + 2)}}}{{l}}{{ \\textbf{{Baseline Models}} }} \\\\
\\midrule
"""
    for model, results in baselines.items():
        if model == "Llama-3.1-Tulu-3-8B-SFT":
            continue
        tex_string += f"""
{"\_".join(model.split("_"))} {" ".join([f"& {'+' if delta and float(score) > 0 else ''}{float(score):.3f}" for score in results.values()])} & {sum(float(score) for score in results.values()) / len(results):.3f} \\\\"""
    tex_string += """
\\bottomrule
\\end{tabular}
\\end{table*}"""
    print(tex_string)
    with open("results_table_delta.tex" if delta else "results_table.tex", "w") as f:
        f.write(tex_string)


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--delta", action="store_true", help="Use delta scores")
    args.add_argument(
        "--results_directory",
        type=str,
        default="/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/olmes/run/active_new_centered_cosine_big_batches",
        help="Path to the results directory",
    )
    args = args.parse_args()
    results_directory = args.results_directory
    results_dict, baselines = collect_results(results_directory)
    display_results(results_dict, baselines, delta=args.delta)
