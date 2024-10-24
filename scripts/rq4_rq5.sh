#!/bin/bash

# run the RAG system using the language model of choice with various numbers of retrieved texts included in the prompt

# get the language model (qwen2.5:0.5b or gemma2:2b) from the command line
lm_model_name=$1

# define how many documents will be retrieved from the db and included in the prompt
retrieved_documents_values=($(seq 1 15))

# define how many times each experiment will be executed
run_numbers=($(seq 1 10))

# loop through each number of retrieved documents
for retrieved_documents in "${retrieved_documents_values[@]}"; do
    # repeat the experiment for each run number
    for run_number in "${run_numbers[@]}"; do
        echo "$lm_model_name" "$run_number"
        # run the Python script with current parameters
        python ../retrieval_augmented_generation/run_rag.py "gte-base" "$lm_model_name" "nfcorpus" "qdrant" "$retrieved_documents" "$run_number" "rq4_rq5"
        # unload model
        ollama stop "$lm_model_name"
    done
done