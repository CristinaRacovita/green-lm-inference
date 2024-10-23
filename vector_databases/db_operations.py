import sys
from datetime import datetime

sys.path.append("../")

from qdrant_db import (
    create_collections as qdrant_create_collections,
    connnect as qdrant_connect,
    add_objects_for_collection as qdrant_store_embeddings,
    count_collection_entries as qdrant_count_entries,
    get_k_most_similar as qdrant_get_k_most_similar,
)
from milvus_db import (
    create_collections as milvus_create_collections,
    connnect as milvus_connect,
    add_objects_for_collection as milvus_store_embeddings,
    count_collection_entries as milvus_count_entries,
    get_k_most_similar as milvus_get_k_most_similar,
)
from weaviate_db import (
    create_collections as weaviate_create_collections,
    connnect as weaviate_connect,
    add_objects_for_collection as weaviate_store_embeddings,
    count_collection_entries as weaviate_count_entries,
    get_k_most_similar as weaviate_get_k_most_similar,
)
from utils import read_from_directory, store_timestamps, load_query_embeddings

DIRECTORY_PATH = "../data/embeddings"


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


def index_data(vector_db_name, dataset_name, run_index):
    # load data and get the vector db client
    json_data, collection_names = read_from_directory(DIRECTORY_PATH, dataset_name + ".json")
    collection_names = [collection_name + f"_{run_index}" for collection_name in collection_names]

    client = get_client(vector_db_name)

    # record the start date and time
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

    # record the end date and time
    end_time = datetime.now()

    # check how many entries have been stored in the vector db (for debugging purposes)
    # get_entries_number(vector_db_name, collection_names[0], client)

    client.close()

    return start_time, end_time


def query_db(vector_db_name, dataset_name, number_of_queries=500, top_k_entries=10):
    # load queries and get the vector db client
    data_path = f"{DIRECTORY_PATH}/{dataset_name}_queries.json"
    queries = load_query_embeddings(data_path, number_of_queries)

    client = get_client(vector_db_name)

    # record the start date and time
    start_time = datetime.now()

    # search results for each query
    collection_name = dataset_name.replace(".json", "") + "_1"

    if vector_db_name == "milvus":
        get_k_most_similar = milvus_get_k_most_similar
    elif vector_db_name == "qdrant":
        get_k_most_similar = qdrant_get_k_most_similar
    else:
        get_k_most_similar = weaviate_get_k_most_similar

    for query in queries:
        results = get_k_most_similar(query, client, collection_name, top_k_entries)

    # record the end date and time
    end_time = datetime.now()

    print(len(results))

    client.close()

    return start_time, end_time


if __name__ == "__main__":
    # parse arguments
    if len(sys.argv) != 7:
        print("Please provide: ", end="")
        print("operation type, db name, embeddings filename, run id, store flag and results dir.")
        sys.exit(1)

    operation_type = sys.argv[1]
    vector_db_name = sys.argv[2]
    dataset_name = sys.argv[3]
    run_index = int(sys.argv[4])
    store_flag = sys.argv[5] == "True"
    results_path = f"../results/{sys.argv[6]}"

    # run the chosen operation
    if operation_type == "index":
        start_time, end_time = index_data(vector_db_name, dataset_name, run_index)
    elif operation_type == "query":
        start_time, end_time = query_db(vector_db_name, dataset_name)
    else:
        print("Please use a valid operation type: index or query.")
        sys.exit(1)

    # store the recorded timestamps
    if store_flag:
        timestamps_path = f"{results_path}/timestamps_{dataset_name}_{vector_db_name}.csv"
        store_timestamps(timestamps_path, run_index, start_time, end_time)
