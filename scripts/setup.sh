#!/bin/bash

# create the directories for storing the results and unzips the original results
tar -xf results.zip

# install the Python packages in a new Conda environment
conda env create -f ../environment.yml

# make conda command available in the terminal
conda init

# pull the language models
echo "pulling the language models..."
ollama pull gemma2:2b
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_0