#!/bin/bash

# index or query 10 times each dataset of embeddings using Milvus DB

# get the operation type (index or query) from the command line
operation_type=$1

# set the results path
if [ "$operation_type" = "index" ]; then
    results_path="rq1_indexing"
elif [ "$operation_type" = "query" ]; then
    results_path="rq1_querying"
fi

# define arrays with the embedding datasets and run numbers
embedding_datasets=("gte_base_arguana"
                    "gte_base_cqadupstack_webmasters"
                    "gte_base_nfcorpus"
                    "gte_large_arguana"
                    "gte_large_cqadupstack_webmasters"
                    "gte_large_nfcorpus"
                    "gte_small_arguana"
                    "gte_small_cqadupstack_webmasters"
                    "gte_small_nfcorpus")
run_numbers=($(seq 1 10))

# loop through each dataset
for embedding_dataset in "${embedding_datasets[@]}"; do
    # repeat the experiment for each run number
    for run_number in "${run_numbers[@]}"; do
        echo "$embedding_dataset" "$run_number"
        # run the Python script with current parameters
        python ../vector_databases/db_operations.py "$operation_type" "milvus" "$embedding_dataset" "$run_number" "True" "$results_path"
    done
done