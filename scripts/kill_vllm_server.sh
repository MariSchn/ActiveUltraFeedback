#!/bin/bash

echo "Searching for active vllm server processes..."
PIDS=$(pgrep -f "vllm serve")

if [ -z "$PIDS" ]; then
    echo "No active vllm server processes found."
    exit 0
else
    echo "Found vllm server processes with PIDs: $PIDS"
    echo "Attempting to terminate them..."
    pkill -f "vllm serve"
    if [ $? -eq 0 ]; then
        echo "Successfully terminated vllm server processes."
    else
        echo "Failed to terminate some vllm server processes."
    fi
fi