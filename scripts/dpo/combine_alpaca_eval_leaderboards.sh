#!/bin/bash
# Combine multiple alpaca_eval leaderboard CSVs into a single CSV
# Each CSV's first column (second row) will be replaced with the model path from the file path

set -e

# Default paths
BASE_DIR="${BASE_DIR:-results/alpaca_eval/dpo}"
OUTPUT_FILE="${OUTPUT_FILE:-results/alpaca_eval/combined_leaderboard.csv}"
PATTERN="*/weighted_alpaca_eval_vllm_llama3_70b/leaderboard.csv"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--base-dir DIR] [--output FILE]"
            echo ""
            echo "Combines CSV files from: BASE_DIR/*/*/weighted_alpaca_eval_vllm_llama3_70b/leaderboard.csv"
            echo ""
            echo "Options:"
            echo "  --base-dir DIR    Base directory to search (default: results/alpaca_eval/dpo)"
            echo "  --output FILE     Output file path (default: results/alpaca_eval/combined_leaderboard.csv)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Normalize BASE_DIR (remove trailing slash if present)
BASE_DIR="${BASE_DIR%/}"

# Find all matching CSV files
echo "Searching for CSV files in ${BASE_DIR}..."
csv_files=()
while IFS= read -r -d '' file; do
    csv_files+=("$file")
done < <(find "$BASE_DIR" -path "*/weighted_alpaca_eval_vllm_llama3_70b/leaderboard.csv" -type f -print0 | sort -z)

if [ ${#csv_files[@]} -eq 0 ]; then
    echo "Error: No CSV files found matching pattern: ${BASE_DIR}/*/*/weighted_alpaca_eval_vllm_llama3_70b/leaderboard.csv"
    exit 1
fi

echo "Found ${#csv_files[@]} CSV file(s)"

# Create output directory if it doesn't exist
output_dir=$(dirname "$OUTPUT_FILE")
mkdir -p "$output_dir"

# Process first file to get header
first_file="${csv_files[0]}"
echo "Processing: $first_file"

# Extract model path from first file (e.g., "vuyw61pe/w4zu9uh1" from "results/alpaca_eval/dpo/vuyw61pe/w4zu9uh1/...")
# Remove BASE_DIR prefix and extract the two directory levels
rel_path="${first_file#${BASE_DIR}/}"
model_path=$(echo "$rel_path" | cut -d'/' -f1-2)

# Read first file
header=$(head -n 1 "$first_file")
data_row=$(sed -n '2p' "$first_file")

# Replace first column of data row with model path
# Remove everything before the first comma, then prepend model path
updated_row="${model_path},${data_row#*,}"

# Write header and first row to output
{
    echo "$header"
    echo "$updated_row"
} > "$OUTPUT_FILE"

# Process remaining files (skip header, only use data row)
for csv_file in "${csv_files[@]:1}"; do
    echo "Processing: $csv_file"
    
    # Extract model path (same logic as above)
    rel_path="${csv_file#${BASE_DIR}/}"
    model_path=$(echo "$rel_path" | cut -d'/' -f1-2)
    
    # Get data row (second line)
    data_row=$(sed -n '2p' "$csv_file")
    
    # Replace first column with model path
    updated_row="${model_path},${data_row#*,}"
    
    # Append to output file
    echo "$updated_row" >> "$OUTPUT_FILE"
done

echo ""
echo "Combined ${#csv_files[@]} CSV files into: $OUTPUT_FILE"

