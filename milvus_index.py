from pymilvus import (
    MilvusClient,
    CollectionSchema,
    FieldSchema,
    DataType,
)
from read_files import read_from_directory


def create_collections(collection_names, milvus_client):
    my_id = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True)
    text = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024)
    embedding = FieldSchema(
        name="embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=10,
    )
    schema = CollectionSchema(fields=[my_id, text, embedding], auto_id=True)
    for collection_name in collection_names:
        if milvus_client.has_collection(collection_name):
            milvus_client.drop_collection(collection_name)
        milvus_client.create_collection(collection_name=collection_name, schema=schema)


if __name__ == "__main__":
    client = MilvusClient(uri="http://localhost:19530")
    DIRECTORY_PATH = "./data/embeddings"
    try:
        json_data, collection_names = read_from_directory(DIRECTORY_PATH)
        create_collections(collection_names, client)
        for index, collection_name in enumerate(collection_names):
            data = map(
                    lambda el: {"text": el["text"], "embedding": el["embedding"]},
                    json_data[index],
                )

            res = client.insert(
                collection_name=collection_name,
                data=list(data)
            )
            # print(res)
    except Exception as error:
        print(error)
    finally:
        client.close()
