from pymilvus import (
    MilvusClient,
    CollectionSchema,
    FieldSchema,
    DataType,
)
from read_files import read_from_directory

# similarity metric is set when searching 
def create_collections(collection_names, milvus_client, vector_size = 10):
    my_id = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True)
    text = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024)
    embedding = FieldSchema(
        name="embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=vector_size,
    )
    schema = CollectionSchema(fields=[my_id, text, embedding], auto_id=True)
    for collection_name in collection_names:
        if milvus_client.has_collection(collection_name):
            milvus_client.drop_collection(collection_name)
        milvus_client.create_collection(collection_name=collection_name, schema=schema)

def add_objects_for_collection(collection_name, input_data):
    data = map(
            lambda el: {"text": el["text"], "embedding": el["embedding"]},
            input_data,
        )

    client.insert(
        collection_name=collection_name,
        data=list(data)
    )

def get_all_data_from_collection(collection_name, client):
    print(client.describe_collection(collection_name=collection_name))
                

if __name__ == "__main__":
    client = MilvusClient(uri="http://localhost:19530")
    DIRECTORY_PATH = "./data/embeddings"
    try:
        json_data, collection_names = read_from_directory(DIRECTORY_PATH)
        create_collections(collection_names, client)
        for index, collection_name in enumerate(collection_names):
            add_objects_for_collection(collection_name, json_data[index])
            get_all_data_from_collection(collection_name, client)


    except Exception as error:
        print(error)
    finally:
        client.close()
