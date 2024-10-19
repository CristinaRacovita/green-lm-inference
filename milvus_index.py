from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
from read_files import read_from_directory


PARTITION_NAME = "Milvus_Partition"

# similarity metric is set when searching
# https://medium.com/@tspann/partitioning-collections-by-name-395eb48a2238


def create_collections(collection_names, milvus_client, json_data):
    my_id = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True)
    text = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=25000)

    for index, collection_name in enumerate(collection_names):
        embedding = FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=len(json_data[index][0]["embedding"]),
        )
        schema = CollectionSchema(fields=[my_id, text, embedding], auto_id=True)
        if milvus_client.has_collection(collection_name):
            milvus_client.drop_collection(collection_name)
        milvus_client.create_collection(collection_name=collection_name, schema=schema)


def add_objects_for_collection(collection_name, input_data, client):
    data = map(
        lambda el: {"text": el["text"], "embedding": el["embedding"]},
        input_data,
    )

    index_params = [
        {
            "field_name": "embedding",
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": len(input_data[0]["embedding"])},
        }
    ]

    client.create_index(collection_name, index_params)

    client.create_partition(
        collection_name=collection_name, partition_name=PARTITION_NAME
    )

    client.insert(
        collection_name=collection_name, partition_name=PARTITION_NAME, data=list(data)
    )


def get_all_data_from_collection(collection_name, milvus_client):
    milvus_client.load_partitions(
        collection_name=collection_name, partition_names=[PARTITION_NAME]
    )
    res = milvus_client.get_load_state(collection_name=collection_name)
    print(res)
    count = milvus_client.query(
        collection_name=collection_name,
        output_fields=["count(*)"],
        # limit=100
    )

    print(count)


def get_k_most_similar(embedding, milvus_client, collection_name, k):
    milvus_client.load_partitions(
        collection_name=collection_name, partition_names=[PARTITION_NAME]
    )
    search_params = {
        "metric_type": "COSINE",
    }
    results = milvus_client.search(
        collection_name=collection_name,
        data=[embedding],
        parsearch_paramsam=search_params,
        limit=k,
    )
    item_ids = get_ids(results)
    text_items = get_item_texts(milvus_client, collection_name, item_ids)
    print(text_items)
    return text_items


def get_item_texts(milvus_client, collection_name, item_ids):
    items = milvus_client.get(collection_name=collection_name, ids=item_ids)
    text_items = []
    for item in items:
        text_items.append(item["text"])
    return text_items


def get_ids(results):
    item_ids = []
    for query_response in results[0]:
        item_ids.append(query_response["id"])
    return item_ids


if __name__ == "__main__":
    client = MilvusClient(uri="http://localhost:19530")
    DIRECTORY_PATH = "./data/embeddings"
    try:
        json_data, collection_names = read_from_directory(DIRECTORY_PATH)
        create_collections(collection_names, client, json_data)
        for index, collection_name in enumerate(collection_names):
            add_objects_for_collection(collection_name, json_data[index], client)
            # get_all_data_from_collection(collection_name, client)
            get_k_most_similar(
                [0.12, 0.87, -0.44, 0.66, -0.01, 0.23, 0.99, -0.78, 0.11, 0.34],
                client,
                collection_name,
                2,
            )

    except Exception as error:
        print(error)
    finally:
        client.close()
