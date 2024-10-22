#!/bin/bash

# index 10 times using each of the three data bases the cqadupstack-webmasters dataset vectorized with gte-base

# define arrays with the name of the vector databases and run numbers
vector_databases=("milvus" "qdrant" "weaviate")
run_numbers=($(seq 1 10))

# loop through each vector database
for vector_database in "${vector_databases[@]}"; do
    # repeat the experiment for each run number
    for run_number in "${run_numbers[@]}"; do
        echo "$vector_database" "$run_number"
        # run the Python script with current parameters
        python ../vector_databases/index_embeddings.py "$vector_database" "gte_base_cqadupstack_webmasters" "$run_number" "True" "rq2_indexing"
    done
done