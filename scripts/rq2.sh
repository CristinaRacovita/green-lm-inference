#!/bin/bash

# index or query 10 times the cqadupstack-webmasters dataset vectorized with gte-base using each of the three data bases

# get the operation type (index or query) from the command line
operation_type=$1

# set the results path
if [ "$operation_type" = "index" ]; then
    results_path="rq2_indexing"
elif [ "$operation_type" = "query" ]; then
    results_path="rq2_querying"
fi

# define arrays with the name of the vector databases and run numbers
vector_databases=("qdrant" "milvus" "weaviate")
run_numbers=($(seq 1 10))

# loop through each vector database
for vector_database in "${vector_databases[@]}"; do
    # repeat the experiment for each run number
    for run_number in "${run_numbers[@]}"; do
        echo "$vector_database" "$run_number"
        # run the Python script with current parameters
        python ../vector_databases/db_operations.py "$operation_type" "$vector_database" "gte_base_cqadupstack_webmasters" "$run_number" "True" "$results_path"
    done
done