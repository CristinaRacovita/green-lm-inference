#!/bin/bash

# index or query 10 times the cqadupstack-webmasters dataset vectorized with gte-base using each of the three data bases

# get the operation type (index or query) from the command line
operation_type=$1

# start the Docker daemon
"C:\Program Files\Docker\Docker\Docker Desktop.exe" 
sleep 30

# set the results path
if [ "$operation_type" = "index" ]; then
    results_path="rq2_indexing"
    # initialize the needed Docker containers and then stop the containers
    docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
    sleep 30
    docker stop qdrant
    sleep 5
    docker run -d --name weaviate -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.26.5
    sleep 30
    docker stop weaviate
    sleep 5
elif [ "$operation_type" = "query" ]; then
    results_path="rq2_querying"
fi

# define arrays with the name of the vector databases and run numbers
vector_databases=("qdrant" "milvus" "weaviate")
run_numbers=($(seq 1 10))

# activate the Python environment
conda activate green_lm_inference

# loop through each vector database
for vector_database in "${vector_databases[@]}"; do

    # start the docker container of the vector DB
    if [ "$vector_database" = "qdrant" ]; then
        docker start qdrant
    elif [ "$vector_database" = "milvus" ]; then
        docker-compose up -d
    elif [ "$vector_database" = "weaviate" ]; then
        docker start weaviate
    fi
    sleep 30    

    # repeat the experiment for each run number    
    for run_number in "${run_numbers[@]}"; do
        echo "$vector_database" "$run_number"
        # run the Python script with current parameters
        python ../vector_databases/db_operations.py "$operation_type" "$vector_database" "gte_base_cqadupstack_webmasters" "$run_number" "True" "$results_path"
    done

    # stop the docker container of the vector DB
    if [ "$vector_database" = "qdrant" ]; then
        docker stop qdrant
    elif [ "$vector_database" = "milvus" ]; then
        docker-compose down
    elif [ "$vector_database" = "weaviate" ]; then
        docker stop weaviate
    fi
    sleep 30

done