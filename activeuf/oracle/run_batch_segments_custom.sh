#!/bin/bash

# Usage: ./run_batch_segments_custom.sh <segment_size> <total_size> <output_base> [start_index] <other_args...>
# Example: ./run_batch_segments_custom.sh 2500 60829 /path/to/out 0 --direct_output


SEGMENT_SIZE="$1"
TOTAL_SIZE="$2"
OUTPUT_BASE="$3"
START_INDEX=0
if [[ "$4" =~ ^[0-9]+$ ]]; then
    START_INDEX="$4"
    shift 1
fi
DRY_RUN="${DRY_RUN:-true}"
shift 3

NUM_SEGMENTS=$(( (TOTAL_SIZE - START_INDEX + SEGMENT_SIZE - 1) / SEGMENT_SIZE ))

for ((i=0; i<NUM_SEGMENTS; i++)); do
    BATCH_START=$((START_INDEX + i * SEGMENT_SIZE))
    BATCH_END=$((START_INDEX + (i + 1) * SEGMENT_SIZE))
    if [[ $BATCH_END -gt $TOTAL_SIZE ]]; then
        BATCH_END=$TOTAL_SIZE
    fi
    SEGMENT_OUTPUT_BASE="${OUTPUT_BASE}_incase_batch_${BATCH_START}_${BATCH_END}"
    if [ -d "$SEGMENT_OUTPUT_BASE" ]; then
        echo "Skipping segment $i: $BATCH_START to $BATCH_END, output directory exists: $SEGMENT_OUTPUT_BASE"
        continue
    fi
    echo "Ready to submit segment $i: $BATCH_START to $BATCH_END, output_base: $SEGMENT_OUTPUT_BASE"
    # if [[ "$DRY_RUN" == "false" ]]; then
    ./activeuf/oracle/run_annotations_binarized.sh --enable_reasoning --batch_start $BATCH_START --batch_end $BATCH_END --output "$SEGMENT_OUTPUT_BASE" "$@"
    # fi
done
