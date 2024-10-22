#!/bin/bash

# define arrays for model and dataset names
model_names=("gte-large" "gte-base" "gte-small")
dataset_names=("nfcorpus" "arguana" "cqadupstack-webmasters")

# loop through each model name
for model in "${model_names[@]}"; do
    # loop through each dataset name
    for dataset in "${dataset_names[@]}"; do
        echo "$model" "$dataset"
        # run the Python script with current parameters
        python ../embeddings/create_embeddings.py "$model" "$dataset" "1" "False"
    done
done