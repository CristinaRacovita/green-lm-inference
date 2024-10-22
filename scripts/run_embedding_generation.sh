# define arrays for model names and run number
model_names=("gte-large" "gte-base" "gte-small")
run_numbers=($(seq 1 10))

# loop through each model name
for model in "${model_names[@]}"; do
    # repeat the experiment for each run number
    for run_number in "${run_numbers[@]}"; do
        echo "$model" "nfcorpus" "$run_number"
        # run the Python script with current parameters
        python ../embeddings/create_embeddings.py "$model" "nfcorpus" "$run_number" "True"
    done
done