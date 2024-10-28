echo "recording idle measurements..."
sleep 310
python ../utils.py "idle state"

# start the Docker daemon
"C:\Program Files\Docker\Docker\Docker Desktop.exe" 
sleep 30

echo "recording measurements while Docker is running..."
sleep 310
python ../utils.py "Docker running"

# pull the language models
echo "pulling the language models..."
ollama pull gemma2:2b
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_0