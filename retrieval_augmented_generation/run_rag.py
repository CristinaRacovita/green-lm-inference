import sys
from datetime import datetime

sys.path.append("../")
sys.path.append("../vector_databases/")

from utils import load_query_text, store_rag_timestamps
from vector_databases.db_operations import get_client, get_prompt_docs
from embeddings.create_embeddings import get_device, get_model_and_tokenizer, create_embedding

import ollama
from tqdm import tqdm

DIRECTORY_PATH = "../data/datasets"
MAX_RETRIEVED_TEXT_LENGTH = 2000


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


def setup_rag(db_name, dataset_name, embedding_model_name, language_model_name, collection_name):
    query_texts = load_query_text(DIRECTORY_PATH + f"/{dataset_name}/queries.jsonl", 10)
    client = get_client(db_name)

    if db_name == "milvus":
        client.load_collection(collection_name)

    # initialize the embedding model
    device = get_device()
    embedding_model, tokenizer = get_model_and_tokenizer(embedding_model_name, device)
    create_embedding(device, embedding_model_name, embedding_model, tokenizer, [["Test"]], 1, False)

    # initialize the language model
    ollama.pull(language_model_name)
    ask_model(language_model_name, "Hello! I'm loading the model now...")

    return query_texts, client, device, embedding_model, tokenizer


def get_length(text, tokenizer, device):
    tokenized_text = tokenizer(
        text, max_length=512, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    return tokenized_text["input_ids"].size()[1] - 2


def create_prompt(similar_retrieved_texts, query_text):
    prompt = """Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).\n\n"""

    for text_index, similar_retrieved_text in enumerate(similar_retrieved_texts):
        prompt += f"Document [{text_index}]: {similar_retrieved_text[:MAX_RETRIEVED_TEXT_LENGTH]}\n"

    prompt += f"\nQuestion: {query_text}\nAnswer:"
    return prompt


def ask_query(models, query_texts, vector_db_name, device, k):
    embedding_model, embedding_model_name, language_model_name, tokenizer = models

    timestamps = {}
    timestamps["embedding_start_time"] = []
    timestamps["embedding_end_time"] = []
    timestamps["retrieval_end_time"] = []
    timestamps["ask_model_end_time"] = []
    timestamps["prompt_length"] = []
    timestamps["answer_tokens_no"] = []

    for query_text in tqdm(query_texts):
        embedding_start_time = datetime.now()
        embedding = create_embedding(
            device, embedding_model_name, embedding_model, tokenizer, [[query_text]], 1, False
        )
        embedding_end_time = datetime.now()

        similar_retrieved_texts = get_prompt_docs(
            vector_db_name, embedding, client, collection_name, k
        )
        retrieval_end_time = datetime.now()
        prompt = create_prompt(similar_retrieved_texts, query_text)

        answer = ask_model(language_model_name, prompt)["message"]["content"]

        ask_model_end_time = datetime.now()

        prompt_length = get_length(prompt, tokenizer, device)
        answer_length = get_length(answer, tokenizer, device)

        timestamps["embedding_start_time"].append(embedding_start_time)
        timestamps["embedding_end_time"].append(embedding_end_time)
        timestamps["retrieval_end_time"].append(retrieval_end_time)
        timestamps["ask_model_end_time"].append(ask_model_end_time)
        timestamps["prompt_length"].append(prompt_length)
        timestamps["answer_tokens_no"].append(answer_length)

    return timestamps


if __name__ == "__main__":
    # parse arguments
    if len(sys.argv) != 8:
        print("Please provide: ", end="")
        print(
            "embedding and language, dataset, db, number of retrieved docs, run index, and results path."
        )
        sys.exit(1)

    embedding_model_name = sys.argv[1]
    language_model_name = sys.argv[2]
    dataset_name = sys.argv[3]
    db_name = sys.argv[4]
    k = int(sys.argv[5])
    run_index = int(sys.argv[6])
    results_path = f"../results/{sys.argv[7]}"

    collection_name = f"{embedding_model_name}_{dataset_name}".replace("-", "_") + "_1"

    query_texts, client, device, embedding_model, tokenizer = setup_rag(
        db_name, dataset_name, embedding_model_name, language_model_name, collection_name
    )

    models = [embedding_model, embedding_model_name, language_model_name, tokenizer]

    timestamps = ask_query(models, query_texts, db_name, device, k)

    if db_name == "milvus":
        client.release_collection(collection_name)

    client.close()

    lm_name = language_model_name.split(":")[0]
    timestamps_path = f"{results_path}/timestamps_{dataset_name}_{embedding_model_name}_{lm_name}_{db_name}_{k}.csv"
    store_rag_timestamps(timestamps_path, timestamps, run_index)
