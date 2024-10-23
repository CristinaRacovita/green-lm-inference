"""
This script creates embeddings with a model for each text from a corpus or set of queries and stores
the results in a JSON file, where each entry contains the text and the corresponding embedding.
"""

import sys
import json
import warnings
from datetime import datetime

sys.path.append("../")

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from utils import store_timestamps

warnings.filterwarnings("ignore")


def load_data(dataset_name, subset_type, batch_size):
    # compose the file name
    file_name = f"../data/datasets/{dataset_name}/{subset_type}.jsonl"

    # store the titles together with texts or the queries in a list
    with open(file_name, "r") as json_file:
        data = [
            "\n".join({k: v for k, v in json.loads(line).items() if k != "_id"}.values()).strip(
                "\n"
            )
            for line in json_file
        ]
    # split the list in a list of sublists, each having 'batch_size' items
    batched_data = [data[i : i + batch_size] for i in range(0, len(data), batch_size)][:10]

    return batched_data


def get_model_and_tokenizer(model_name, device):
    complete_model_name = f"thenlper/{model_name}"
    tokenizer = AutoTokenizer.from_pretrained(complete_model_name)
    model = AutoModel.from_pretrained(complete_model_name, trust_remote_code=True).to(device)

    return model, tokenizer


def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def embed_data(chosen_dataset, model_name, model, tokenizer, device, run_index, store_flag):
    batch_size = 1
    dataset_name, dataset_component = chosen_dataset

    # set the path to the results file
    stored_model_name = model_name.replace("-", "_")
    stored_dataset_name = dataset_name.replace("-", "_")
    results_path = "../results/rq3"
    timestamps_path = f"{results_path}/timestamps_{stored_model_name}_{stored_dataset_name}.csv"

    # set the maximum input length
    if "v1.5" in model_name:
        max_length = 8192
    else:
        max_length = 512

    # load the data
    data = load_data(dataset_name, dataset_component, batch_size)

    # record the start date and time of embedding generation
    start_time = datetime.now()
    data_embeddings = []

    # get each batch of texts
    for texts in tqdm(data):
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
            embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])

        # add each text with the corresponding embedding in a separate
        detached_embeddings = embeddings.cpu().detach().tolist()

        for i in range(batch_size):
            data_embeddings.append({"text": texts[i], "embedding": detached_embeddings[i]})

        # free the memory
        torch.cuda.empty_cache()
        del batch_dict, outputs, embeddings, detached_embeddings

    # record the end date and time of embedding generation
    end_time = datetime.now()

    if store_flag:
        store_timestamps(timestamps_path, run_index, start_time, end_time)

    # store the data
    embeddings_file_name = f"../data/embeddings/{stored_model_name}_{stored_dataset_name}"

    if dataset_component == "queries":
        embeddings_file_name += "_queries.json"
    else:
        embeddings_file_name += ".json"

    with open(embeddings_file_name, "w") as json_file:
        json.dump(data_embeddings, json_file, indent=4)

    # free the memory
    del data, data_embeddings


def main(model_name, dataset_name, dataset_component, run_index, store_flag):
    # get the available device for running the embedding model on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_id = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(0)
    print(f"Current device: {device_id} ({device}) -> {device_name}")

    # define available models and datasets
    model_names = ["gte-large", "gte-base", "gte-small"]
    dataset_names = ["nfcorpus", "arguana", "cqadupstack-webmasters"]

    # check if the model and dataset are available
    if model_name not in model_names or dataset_name not in dataset_names:
        raise NameError("The model or the dataset is not available.")

    model, tokenizer = get_model_and_tokenizer(model_name, device)
    dataset_info = [dataset_name, dataset_component]
    embed_data(dataset_info, model_name, model, tokenizer, device, run_index, store_flag)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Please provide the model, dataset name, dataset component, run id and store flag.")
        sys.exit(1)

    model_name = sys.argv[1]
    dataset_name = sys.argv[2]
    dataset_component = sys.argv[3]
    run_index = int(sys.argv[4])
    store_flag = sys.argv[5] == "True"

    main(model_name, dataset_name, dataset_component, run_index, store_flag)
