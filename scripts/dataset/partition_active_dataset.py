import argparse
import os
from datasets import load_from_disk


def generate_subsets():
    parser = argparse.ArgumentParser(
        description="Generate subsets of a HF dataset based on row counts."
    )

    # Required arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the source dataset (load_from_disk format).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the resulting subsets will be saved.",
    )
    parser.add_argument(
        "--output_name_replacement",
        type=str,
        default="",
        help="Optional string to replace in the output dataset names.",
    )

    # Optional list of integers.
    # 1000 has been removed from defaults.
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000],
        help="List of integer subset sizes.",
    )

    args = parser.parse_args()

    # 1. Validation and Setup
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path '{args.dataset_path}' does not exist.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Load Dataset
    print(f"Loading dataset from: {args.dataset_path}")
    try:
        ds = load_from_disk(args.dataset_path)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    total_rows = len(ds)
    print(f"Total rows in source dataset: {total_rows}")

    # 3. Determine Basename
    if args.output_name_replacement:
        dataset_basename = args.output_name_replacement
    else:
        dataset_basename = os.path.basename(os.path.normpath(args.dataset_path))

    # 4. Prepare Sizes List
    target_sizes = args.sizes.copy()

    # NOTE: The total dataset size is always added to the list of targets,
    # ensuring the full dataset is generated as one of the outputs.
    if total_rows not in target_sizes:
        target_sizes.append(total_rows)

    # Sort sizes
    sorted_sizes = sorted(target_sizes)
    print(f"Generating subsets for sizes: {sorted_sizes}")

    # 5. Loop and Generate
    for size in sorted_sizes:
        if size > total_rows:
            print(
                f"[SKIP] Size {size} is larger than total rows ({total_rows}). Skipping."
            )
            continue

        # Take the first 'size' elements
        subset = ds.select(range(size))

        # Construct new name: originalname_size
        new_name = f"{dataset_basename}_{size}"
        save_path = os.path.join(args.output_dir, new_name)

        print(f"Processing {new_name}...")

        # Save to disk
        subset.save_to_disk(save_path)
        print(f" -> Saved to {save_path}")

    print("\nAll tasks completed.")


if __name__ == "__main__":
    generate_subsets()
