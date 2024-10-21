# Arrays for model names and dataset names
model_names=("gte-large" "gte-base" "gte-small")
dataset_names=("nfcorpus" "arguana" "cqadupstack-webmasters")

# Initialize run number
run_number=1

# Loop through each model name
for model in "${model_names[@]}"; do
    # Loop through each dataset name
    for dataset in "${dataset_names[@]}"; do
        # Run the Python script with current parameters
        python ./embeddings/create_embeddings.py "$model" "$dataset" "$run_number"
        
        # Increment run number
        ((run_number++))
    done
done