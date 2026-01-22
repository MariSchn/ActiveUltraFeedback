import argparse
import os
import tempfile
import shutil
from huggingface_hub import HfApi, create_repo
from datasets import load_from_disk, Dataset, DatasetDict

DROP_COLS = [
    "input_ids_chosen",
    "input_ids_rejected",
    "attention_mask_chosen",
    "attention_mask_rejected",
]

def _prepare_parquet_folder(src_folder: str) -> str:
    """
    Load HF dataset saved with save_to_disk, drop tokenization columns,
    and write each split to a parquet file in a temp directory.
    Returns the temp directory path.
    """
    ds = load_from_disk(src_folder)
    tmpdir = tempfile.mkdtemp(prefix="hf_parquet_")

    def clean_and_save(split, name):
        to_drop = [c for c in DROP_COLS if c in split.column_names]
        cleaned = split.remove_columns(to_drop) if to_drop else split
        out_path = os.path.join(tmpdir, f"{name}.parquet")
        cleaned.to_parquet(out_path)

    if isinstance(ds, DatasetDict):
        for split_name, split in ds.items():
            clean_and_save(split, split_name)
    elif isinstance(ds, Dataset):
        clean_and_save(ds, "data")
    else:
        raise RuntimeError("Unsupported dataset type loaded from disk.")

    # Optional: copy README.md if present
    readme_src = os.path.join(src_folder, "README.md")
    if os.path.isfile(readme_src):
        shutil.copy2(readme_src, os.path.join(tmpdir, "README.md"))

    return tmpdir

def main():
    parser = argparse.ArgumentParser(description="Upload dataset folder (cleaned to Parquet) and add to HF Collection")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to local HF dataset saved via save_to_disk")
    parser.add_argument("--repo_name", type=str, required=True, help="Name for the new HF repo (e.g., tulu-3-dts-run1)")
    parser.add_argument("--org", type=str, required=True, help="Your HF Organization username (e.g., ActiveUltraFeedback)")
    parser.add_argument("--collection", type=str, default="Experimental Runs", help="Name of the Collection to add this to")
    parser.add_argument("--private", action="store_true", help="Make the dataset private")
    parser.add_argument("--keep_tmp", action="store_true", help="Keep the temporary parquet folder after upload")
    args = parser.parse_args()

    api = HfApi()

    # Convert to parquet and drop columns
    print(f"🔧 Converting to Parquet and dropping columns: {', '.join(DROP_COLS)}")
    parquet_dir = _prepare_parquet_folder(args.folder_path)

    repo_id = f"{args.org}/{args.repo_name}"
    print(f"🚀 Preparing to upload '{parquet_dir}' to '{repo_id}'...")

    try:
        create_repo(repo_id=repo_id, repo_type="dataset", private=args.private, exist_ok=True)
        print(f"✅ Repository ensured: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"❌ Error creating repo: {e}")
        if not args.keep_tmp:
            shutil.rmtree(parquet_dir, ignore_errors=True)
        return

    print("⏳ Uploading Parquet files...")
    api.upload_folder(folder_path=parquet_dir, repo_id=repo_id, repo_type="dataset")
    print("✅ Upload complete!")

    print(f"🔍 Looking for collection: '{args.collection}' in org '{args.org}'...")
    my_collections = api.list_collections(owner=args.org)
    target_collection = next((c for c in my_collections if c.title == args.collection), None)

    if not target_collection:
        print(f"   Collection not found. Creating '{args.collection}'...")
        target_collection = api.create_collection(title=args.collection, namespace=args.org, private=False)

    print(f"➕ Adding '{repo_id}' to collection '{args.collection}'...")
    try:
        api.add_collection_item(collection_slug=target_collection.slug, item_id=repo_id, item_type="dataset", exists_ok=True)
        print(f"🎉 Success! View here: https://huggingface.co/collections/{target_collection.slug}")
    except Exception as e:
        print(f"❌ Error adding to collection: {e}")

    if not args.keep_tmp:
        shutil.rmtree(parquet_dir, ignore_errors=True)
        print("🧹 Cleaned up temporary Parquet folder.")

if __name__ == "__main__":
    main()