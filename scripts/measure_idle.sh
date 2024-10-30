#!/bin/bash

# activate the Python environment
conda activate green_lm_inference

echo "wait for the consumption to normalize"
sleep 30

echo "recording idle measurements..."
sleep 610
python ../utils.py "idle state"

# start the Docker daemon
"C:\Program Files\Docker\Docker\Docker Desktop.exe" 
sleep 60

echo "recording measurements while Docker is running..."
sleep 610
python ../utils.py "Docker running"
