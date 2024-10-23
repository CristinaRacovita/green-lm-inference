import os
import json
import random

import pandas as pd


def read_json_file(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)

    return data


def read_from_directory(directory_path, file_name=None):
    json_data = []
    collection_names = []

    if file_name:
        file_path = os.path.join(directory_path, file_name)
        file_name_without_type = file_name.replace(".json", "")
        return [read_json_file(file_path)], [file_name_without_type]

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        collection_names.append(file_path.split("\\")[1].replace(".json", ""))
        json_data.append(read_json_file(file_path))

    return json_data, collection_names


def store_timestamps(timestamps_path, run_index, start_time, end_time):
    new_experiment_data = pd.DataFrame(
        {
            "run_number": run_index,
            "start_timestamp": start_time,
            "end_timestamp": end_time,
        },
        index=[0],
    )

    # if previous info about the experiment has been saved, then append new info to it
    if os.path.exists(timestamps_path):
        existing_experiment_data = pd.read_csv(timestamps_path)
        experiment_data = pd.concat([existing_experiment_data, new_experiment_data])

    else:
        experiment_data = new_experiment_data

    # store the updated info data
    experiment_data.to_csv(timestamps_path, index=None)


def load_query_embeddings(file_path, queries_number):
    # set the seed to get the same queries every time
    random.seed(42)

    # get the embeddings of queries in a list and shuffle the list
    query_data = read_json_file(file_path)
    queries = [query["embedding"] for query in query_data]
    random.shuffle(queries)

    # get the first queries_number embedded queries
    return queries[:queries_number]


def load_query_text(file_path, queries_number):
    random.seed(42)
    with open(file_path, "r") as json_file:
        queries = [json.loads(line)["text"] for line in json_file]

    random.shuffle(queries)

    return queries[:queries_number]
