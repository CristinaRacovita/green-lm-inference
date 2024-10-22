import sys
from datetime import datetime

sys.path.append("../")

from qdrant_db import (
    create_collections as qdrant_create_collections,
    connnect as qdrant_connect,
    add_objects_for_collection as qdrant_store_embeddings,
    count_collection_entries as qdrant_count_entries,
)

from milvus_db import (
    create_collections as milvus_create_collections,
    connnect as milvus_connect,
    add_objects_for_collection as milvus_store_embeddings,
    count_collection_entries as milvus_count_entries,
)
from weaviate_db import (
    create_collections as weaviate_create_collections,
    connnect as weaviate_connect,
    add_objects_for_collection as weaviate_store_embeddings,
    count_collection_entries as weaviate_count_entries,
)
from utils import read_from_directory, store_timestamps


def get_client(vector_db_name):
    if vector_db_name == "milvus":
        return milvus_connect()
    elif vector_db_name == "qdrant":
        return qdrant_connect()
    else:
        return weaviate_connect()


def get_entries_number(vector_db_name, collection_name, client):
    if vector_db_name == "milvus":
        number_of_stored_embeddings = milvus_count_entries(collection_name, client)
    elif vector_db_name == "qdrant":
        number_of_stored_embeddings = qdrant_count_entries(collection_name, client)
    else:
        number_of_stored_embeddings = weaviate_count_entries(collection_name, client)

    print(f"Number of stored embeddings: {number_of_stored_embeddings}")


def main(vector_db_name, dataset_name, run_index, store_flag, results_directory):
    directory_path = "../data/embeddings"
    results_path = f"../results/{results_directory}"

    # load data and get the vector db client
    json_data, collection_names = read_from_directory(directory_path, dataset_name + ".json")
    collection_names = [collection_name + f"_{run_index}" for collection_name in collection_names]

    client = get_client(vector_db_name)

    # record the start date and time of embedding generation
    start_time = datetime.now()

    # store the embeddings
    if vector_db_name == "milvus":
        milvus_create_collections(collection_names, client, json_data)
        milvus_store_embeddings(collection_names[0], json_data[0], client)

    elif vector_db_name == "qdrant":
        qdrant_create_collections(collection_names, client, json_data)
        qdrant_store_embeddings(collection_names[0], client, json_data[0])

    else:
        weaviate_create_collections(collection_names, client)
        weaviate_store_embeddings(collection_names[0], client, json_data[0])

    # record the end date and time of embedding generation
    end_time = datetime.now()

    if store_flag:
        timestamps_path = f"{results_path}/timestamps_{dataset_name}_{vector_db_name}.csv"
        store_timestamps(timestamps_path, run_index, start_time, end_time)

    # check how many entries have been stored in the vector db (for debugging purposes)
    # get_entries_number(vector_db_name, collection_names[0], client)

    client.close()


if len(sys.argv) != 6:
    print("Please provide the db name, embeddings filename, run id, store flag and results dir.")
    sys.exit(1)


vector_db_name = sys.argv[1]
dataset_name = sys.argv[2]
run_index = int(sys.argv[3])
store_flag = sys.argv[4] == "True"
results_directory = sys.argv[5]

main(vector_db_name, dataset_name, run_index, store_flag, results_directory)
