"""
This script creates embeddings with a selected model for each text from a corpus and stores the 
results in a JSON file, where each entry contains the text and the corresponding embeddings.
"""

import sys
import json
import warnings
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")


def load_data(dataset_name, subset_type, batch_size):
    # compose the file name
    file_name = f"../data/datasets/{dataset_name}/{subset_type}.jsonl"

    # store the titles together with texts or the queries in a list
    with open(file_name, "r") as json_file:
        data = [
            "\n".join(
                {k: v for k, v in json.loads(line).items() if k != "_id"}.values()
            ).strip("\n")
            for line in json_file
        ]
    # split the list in a list of sublists, each having 'batch_size' items
    batched_data = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    return batched_data


def get_model_and_tokenizer(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

    return model, tokenizer


def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def embed_corpus(dataset_name, model_name, model, tokenizer, device, batch_size):
    # set the maximum input length
    if "v1.5" in model_name:
        max_length = 8192
    else:
        max_length = 512

    # load the corpus
    corpus = load_data(dataset_name, "corpus", batch_size)

    corpus_embeddings = []

    # get each batch of texts
    for texts in tqdm(corpus):
        # tokenize the input texts
        batch_dict = tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        # get embeddings
        outputs = model(**batch_dict)

        if "v1.5" in model_name:
            embeddings = outputs.last_hidden_state[:, 0]
        else:
            embeddings = average_pool(
                outputs.last_hidden_state, batch_dict["attention_mask"]
            )

        # add each text with the corresponding embedding in a separate
        detached_embeddings = embeddings.cpu().detach().tolist()

        corpus_embeddings.append(
            [
                {"text": texts[i], "embedding": detached_embeddings[i]}
                for i in range(batch_size)
            ]
        )

        # free the memory
        torch.cuda.empty_cache()
        del batch_dict, outputs, embeddings, detached_embeddings

    # store the data
    with open(f"./data/embeddings/{model_name}_{dataset_name}.json", "w") as json_file:
        json.dump(corpus_embeddings, json_file, indent=4)

    # free the memory
    del corpus, corpus_embeddings


def main(model_name, dataset_name, batch_size):
    # get the available device for running the embedding model on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_id = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(0)
    print(f"Current device: {device_id} ({device}) -> {device_name}")

    # define available models and datasets
    model_names = [
        "thenlper/gte-large",
        "thenlper/gte-base",
        "thenlper/gte-small",
    ]
    dataset_names = ["nfcorpus", "scifact", "arguana", "cqadupstack-webmasters"]

    # check if the model and dataset are available
    if model_name not in model_names or dataset_name not in dataset_names:
        raise NameError("The model or the dataset is not available.")
    model, tokenizer = get_model_and_tokenizer(model_name, device)
    embed_corpus(
        dataset_name, model_name.split("/")[-1], model, tokenizer, device, batch_size
    )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Please provide the model and dataset names with the batch size.")
        sys.exit(1)

    model_name = sys.argv[1]
    dataset_name = sys.argv[2]
    batch_size = int(sys.argv[3])

    main(model_name, dataset_name, batch_size=batch_size)
