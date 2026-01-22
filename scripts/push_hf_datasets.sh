#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --org ORG [--collection SLUG] [--private] [--dry-run] \
      (--dir PATH --name DATASET_NAME | --root ROOT_DIR [--prefix PREFIX])

Options:
  --org ORG            Hugging Face organization name (required)
  --collection SLUG    Collection slug to add each dataset to (optional)
                       Example: your-org/my-collection-64f9a55bb3115b4f513ec026
  --private            Create repos as private (default: public)
  --dry-run            Print actions without performing uploads
  --dir PATH           Local dataset folder to upload
  --name DATASET_NAME  Dataset repo name under the org (with --dir)
  --root ROOT_DIR      Bulk mode: upload each first-level subdirectory in ROOT_DIR
  --prefix PREFIX      Prefix to add to repo names in bulk mode (optional)
  --help               Show this help

Examples:
  # Single dataset
  $(basename "$0") --org your-org \
    --dir datasets/active_preference_dataset \
    --name active_preference_dataset \
    --collection your-org/my-collection-64f9a55bb3115b4f513ec026

  # Bulk: all subfolders in datasets/ become datasets under your-org
  $(basename "$0") --org your-org --root datasets --prefix proj_

Environment:
  HF_TOKEN             If set, used for authentication. Otherwise, if a file
                       'huggingface/token' exists, it will be used.
EOF
}

ORG=""
COLLECTION_SLUG=""
DIR=""
NAME=""
ROOT=""
PREFIX=""
PRIVATE="false"
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --org) ORG="${2:-}"; shift 2;;
    --collection) COLLECTION_SLUG="${2:-}"; shift 2;;
    --dir) DIR="${2:-}"; shift 2;;
    --name) NAME="${2:-}"; shift 2;;
    --root) ROOT="${2:-}"; shift 2;;
    --prefix) PREFIX="${2:-}"; shift 2;;
    --private) PRIVATE="true"; shift 1;;
    --dry-run) DRY_RUN="true"; shift 1;;
    --help|-h) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 2;;
  esac
done

if [[ -z "$ORG" ]]; then
  echo "Error: --org is required" >&2
  usage
  exit 2
fi

# Mode validation
if [[ -n "$DIR" || -n "$NAME" ]]; then
  if [[ -z "$DIR" || -z "$NAME" ]]; then
    echo "Error: --dir and --name must be provided together" >&2
    usage
    exit 2
  fi
  if [[ -n "$ROOT" ]]; then
    echo "Error: Use either --dir/--name or --root, not both" >&2
    usage
    exit 2
  fi
else
  if [[ -z "$ROOT" ]]; then
    echo "Error: Provide either --dir/--name or --root" >&2
    usage
    exit 2
  fi
fi

# Token detection
if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -f "huggingface/token" ]]; then
    export HF_TOKEN="$(cat huggingface/token)"
  fi
fi

# Ensure huggingface_hub is available
if ! python - <<'PY' >/dev/null 2>&1
import sys
import importlib
sys.exit(0 if importlib.util.find_spec('huggingface_hub') else 1)
PY
then
  echo "Installing huggingface_hub..." >&2
  pip install -q --user huggingface_hub
fi

# Optional: perform CLI login if token present and not already logged in
if [[ -n "${HF_TOKEN:-}" ]]; then
  if ! huggingface-cli whoami >/dev/null 2>&1; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential >/dev/null 2>&1 || true
  fi
fi

bool_py() { [[ "$1" == "true" ]] && echo True || echo False; }

upload_one() {
  local local_dir="$1"
  local dataset_name="$2"
  local repo_id="$ORG/$dataset_name"

  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] Would create repo: $repo_id (dataset, private=$(bool_py "$PRIVATE"))"
    echo "[DRY-RUN] Would upload folder: $local_dir -> $repo_id"
    if [[ -n "$COLLECTION_SLUG" ]]; then
      echo "[DRY-RUN] Would add $repo_id to collection $COLLECTION_SLUG"
    fi
    return 0
  fi

  python - "$local_dir" "$repo_id" "$COLLECTION_SLUG" "$(bool_py "$PRIVATE")" <<'PY'
import os, sys
from huggingface_hub import HfApi, create_repo, upload_folder

local_dir = sys.argv[1]
repo_id = sys.argv[2]
collection_slug = sys.argv[3]
private_str = sys.argv[4]
private = (private_str == 'True')

token = os.environ.get('HF_TOKEN')
api = HfApi(token=token)

# 1) Create (or ensure) dataset repo
create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)

# 2) Upload folder (idempotent overwrite of same paths)
upload_folder(
    repo_id=repo_id,
    repo_type="dataset",
    folder_path=local_dir,
    token=token,
    commit_message=f"Upload {os.path.basename(local_dir)} via push_hf_datasets.sh",
)

# 3) Optionally add to collection
if collection_slug:
    api.add_collection_item(
        collection_slug=collection_slug,
        item_id=repo_id,
        item_type="dataset",
        exists_ok=True,
    )

print(f"Uploaded {repo_id}")
if collection_slug:
    print(f"Added to collection: {collection_slug}")
PY
}

if [[ -n "$DIR" ]]; then
  if [[ ! -d "$DIR" ]]; then
    echo "Error: --dir '$DIR' not found or not a directory" >&2
    exit 2
  fi
  upload_one "$DIR" "$NAME"
else
  if [[ ! -d "$ROOT" ]]; then
    echo "Error: --root '$ROOT' not found or not a directory" >&2
    exit 2
  fi
  shopt -s nullglob
  for d in "$ROOT"/*; do
    [[ -d "$d" ]] || continue
    base="$(basename "$d")"
    name="$PREFIX$base"
    upload_one "$d" "$name"
  done
fi

echo "Done."
