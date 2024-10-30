
#!/bin/bash

# This script initially collects the idle measurements and then runs the scripts associated with the research questions.
# After calling each script, a 60 second pause is inserted to allow time for previously used resources to be released.

# collect idle measurements
./measure_idle.sh 
sleep 60

# create corpus and query embeddings for each dataset with each embedding model
./create_all_embeddings.sh`
sleep 60

# index and then query 10 times each dataset of embeddings using Milvus DB
./rq1.sh index
sleep 60

./rq1.sh query
sleep 60

# index and then query query 10 times the cqadupstack-webmasters dataset vectorized with gte-base using each of the three data bases
./rq2.sh index
sleep 60

./rq2.sh query
sleep 60

# benchmark the embedding generation which creates embeddings for nfcorpus dataset 10 times with each embedding model
./rq3.sh
sleep 60

# run the RAG system to answer 100 questions with the language model of choice, gte-base embedding model, qdrant vector DB
# and a various number of retrieved texts included from the arguana dataset in the prompt (1 up to 5):
./rq4_rq5.sh gemma2:2b
sleep 60

./rq4_rq5.sh hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_0
