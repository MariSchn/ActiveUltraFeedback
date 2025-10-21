from argparse import ArgumentParser
from datasets import Dataset, load_from_disk
import glob
import os
import torch

# python -m activeuf.new_loop.combine_dataset_with_features --inputs_path datasets/combined_with_smqwen_3_235b --feature_partials_path datasets/combined_with_smqwen_3_235b-feature_partials

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--inputs_path", help="Path to the dataset with prompts and response texts."
    )
    parser.add_argument(
        "--feature_partials_path",
        help="Path to the dir of precomputed features for the dataset.",
    )
    args = parser.parse_args()

    temp_ids = []
    features = []
    filepaths = list(glob.glob(os.path.join(args.feature_partials_path, "*.pt")))
    for filepath in filepaths:
        x = torch.load(filepath)
        temp_ids.extend(x["temp_ids"])
        features.append(x["features"])
    features = torch.vstack(features)

    temp_ids = torch.tensor(temp_ids)
    sort_keys = temp_ids[:, 0] * (temp_ids[:, 1].max() + 1) + temp_ids[:, 1]
    sort_idxs = sort_keys.argsort()
    features = features[sort_idxs]

    n_completions_per_prompt = max(temp_ids[:, 1]) + 1
    features = features.view(-1, n_completions_per_prompt, features.size(-1))
    features = Dataset.from_dict({"features": features})

    dataset = load_from_disk(args.inputs_path).select(range(len(features)))
    dataset = dataset.add_column("features", features["features"])
    dataset.save_to_disk(args.inputs_path + "-features")
