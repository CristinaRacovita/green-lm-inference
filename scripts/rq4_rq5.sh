#!/bin/bash

# run the RAG system using the language model of choice with various numbers of retrieved texts included in the prompt

# get the language model (hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_0 or gemma2:2b) from the command line
lm_name=$1

# start the Docker daemon
"C:\Program Files\Docker\Docker\Docker Desktop.exe" 
sleep 30

# start the qdrant container
docker start qdrant
sleep 30

# get the short lm name that will be included in the directory with the results
if [ "$lm_name" = "gemma2:2b" ]; then
    lm_short_name="gemma"
elif [ "$lm_name" = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_0" ]; then
    lm_short_name="llama"
fi

# store the data
echo "Store the embeddings in the database"
python ../vector_databases/db_operations.py "index" "qdrant" "gte_base_nfcorpus" "1" "False" ""

# define how many documents will be retrieved from the db and included in the prompt
retrieved_documents_values=(1 2 3 4 5)

# loop through each number of retrieved documents
for retrieved_documents in "${retrieved_documents_values[@]}"; do
    # run the Python script with current parameters
    echo "$lm_name" "$retrieved_documents"
    python ../retrieval_augmented_generation/run_rag.py "gte-base" "$lm_name" "nfcorpus" "qdrant" "$retrieved_documents" "rq4_rq5_$lm_short_name"
    # unload model
    ollama stop "$lm_name"
done

# stop the qdrant container
docker stop qdrant
