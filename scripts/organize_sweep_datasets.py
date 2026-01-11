#!/usr/bin/env python3
"""
Organize datasets from wandb sweep by acquisition function type.
Reads wandb sweep runs, extracts acquisition function from config,
and copies datasets to organized output directories.
"""

import argparse
import os
import shutil
import wandb
from datasets import load_from_disk
from pathlib import Path


def get_sweep_runs(sweep_id, entity="ActiveUF", project="loop"):
    """Get all runs from a wandb sweep."""
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    return sweep.runs


def organize_datasets(
    sweep_id,
    loop_base_dir,
    output_base_dir,
    model_type,
    entity="ActiveUF",
    project="loop",
    dry_run=False,
):
    """
    Organize datasets from wandb sweep by acquisition function.
    
    Args:
        sweep_id: wandb sweep ID (e.g., "3e7zl14s")
        loop_base_dir: Base directory containing sweep datasets (e.g., "$SCRATCH/ActiveUltraFeedback/datasets/loop/3e7zl14s")
        output_base_dir: Output directory (e.g., "$SCRATCH/ActiveUltraFeedback/datasets/skywork/actives")
        model_type: "dpo" or "rm"
        entity: wandb entity
        project: wandb project
        dry_run: If True, just print actions without executing
    """
    print(f"🔍 Fetching runs from sweep {sweep_id}...")
    runs = get_sweep_runs(sweep_id, entity, project)
    
    print(f"Found {len(runs)} runs in sweep")
    
    # Mapping of acquisition functions to dataset paths
    datasets_by_acq = {}
    
    for run in runs:
        run_id = run.id
        config = run.config
        
        # Extract acquisition function from config (key is 'acquisition_function_type')
        acq_fn = config.get("acquisition_function_type", None)
        
        # Handle case where acq_fn might be a dict with 'value' key
        if isinstance(acq_fn, dict):
            acq_fn = acq_fn.get("value", None)
        
        if not acq_fn:
            print(f"⚠️  Run {run_id}: No acquisition_function_type in config, skipping")
            continue
        
        # Construct dataset path
        dataset_path = os.path.join(loop_base_dir, run_id)
        
        if not os.path.exists(dataset_path):
            print(f"⚠️  Run {run_id} ({acq_fn}): Dataset not found at {dataset_path}")
            continue
        
        # Store the mapping
        if acq_fn not in datasets_by_acq:
            datasets_by_acq[acq_fn] = []
        
        datasets_by_acq[acq_fn].append({
            "run_id": run_id,
            "path": dataset_path,
            "config": config
        })
        
        print(f"✅ Run {run_id}: {acq_fn} -> {dataset_path}")
    
    # Now organize and save datasets
    print("\n" + "="*80)
    print(f"Organizing datasets by acquisition function...")
    print("="*80)
    
    for acq_fn, dataset_list in datasets_by_acq.items():
        print(f"\n📦 Processing {acq_fn} ({len(dataset_list)} run(s))...")
        
        # Use the first (or best) dataset for each acquisition function
        # You might want to add logic here to select the best one based on some metric
        selected = dataset_list[0]
        
        if len(dataset_list) > 1:
            print(f"   ⚠️  Multiple datasets found for {acq_fn}:")
            for d in dataset_list:
                print(f"      - {d['run_id']}")
            print(f"   Using first one: {selected['run_id']}")
        
        # Construct output path
        output_path = os.path.join(output_base_dir, model_type, acq_fn)
        
        print(f"   Source: {selected['path']}")
        print(f"   Target: {output_path}")
        
        if dry_run:
            print(f"   [DRY RUN] Would copy dataset here")
            continue
        
        # Load and save dataset
        try:
            print(f"   Loading dataset...")
            dataset = load_from_disk(selected['path'])
            print(f"   Loaded {len(dataset)} samples")
            
            print(f"   Saving to {output_path}...")
            dataset.save_to_disk(output_path)
            print(f"   ✅ Saved successfully")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "="*80)
    print("✨ Done!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Organize wandb sweep datasets by acquisition function"
    )
    
    parser.add_argument(
        "--sweep_id",
        type=str,
        required=True,
        help="Wandb sweep ID (e.g., '3e7zl14s')"
    )
    
    parser.add_argument(
        "--loop_base_dir",
        type=str,
        required=True,
        help="Base directory containing sweep run datasets (e.g., '$SCRATCH/ActiveUltraFeedback/datasets/loop/3e7zl14s')"
    )
    
    parser.add_argument(
        "--output_base_dir",
        type=str,
        required=True,
        help="Output base directory (e.g., '$SCRATCH/ActiveUltraFeedback/datasets/skywork/actives')"
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["dpo", "rm"],
        help="Model type: 'dpo' or 'rm'"
    )
    
    parser.add_argument(
        "--entity",
        type=str,
        default="ActiveUF",
        help="Wandb entity (default: ActiveUF)"
    )
    
    parser.add_argument(
        "--project",
        type=str,
        default="loop",
        help="Wandb project (default: loop)"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print actions without executing"
    )
    
    args = parser.parse_args()
    
    # Expand environment variables
    loop_base_dir = os.path.expandvars(args.loop_base_dir)
    output_base_dir = os.path.expandvars(args.output_base_dir)
    
    organize_datasets(
        sweep_id=args.sweep_id,
        loop_base_dir=loop_base_dir,
        output_base_dir=output_base_dir,
        model_type=args.model_type,
        entity=args.entity,
        project=args.project,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
