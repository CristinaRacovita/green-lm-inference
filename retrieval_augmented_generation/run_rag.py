import sys

sys.path.append("../")

from utils import load_query_text
from vector_databases.db_operations import get_client, get_prompt_docs
from embeddings.create_embeddings import get_device, get_model_and_tokenizer, create_embedding

import ollama

DIRECTORY_PATH = "../data/datasets"
VECTOR_DB = "qdrant"
DATASET_NAME = "arguana"
EMBEDDING_MODEL_NAME = "gte-large"
MODEL_NAME = "gemma2:2b"


def ask_model(model_name, prompt):
    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response


query_texts = load_query_text(DIRECTORY_PATH + f"/{DATASET_NAME}/queries.jsonl", 10)
client = get_client(VECTOR_DB)

device = get_device()
model, tokenizer = get_model_and_tokenizer(EMBEDDING_MODEL_NAME, device)

ollama.pull(MODEL_NAME)
ask_model(MODEL_NAME, "Hello! I'm loading the model now...")


embedding = create_embedding(device, model, tokenizer, [query_texts[0]], 1, False)
print(len(embedding))
similar_retrieved_texts = get_prompt_docs(client, DATASET_NAME, 10, [0])
print(len(similar_retrieved_texts))

# for query_text in query_texts:
#     embedding = create_embedding(device, model, tokenizer, [query_text], 1, False)
#     similar_retrieved_texts = get_prompt_docs(client, DATASET_NAME, 10, [0])

# create prompt

# ask LM


# load embedding model, language model, vector db

# get queries

# for each query: embed it, get from the vector db top k answers, create the prompt, generate answer

# before benchmarking answer to one question
