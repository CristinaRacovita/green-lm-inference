import sys
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from read_files import read_from_directory


def create_collections(collection_names, q_client, json_data):
    for index, collection_name in enumerate(collection_names):
        if collection_exists(collection_name, q_client):
            q_client.delete_collection(collection_name=collection_name)
        q_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=len(json_data[index][0]["embedding"]), distance=Distance.COSINE
            ),
        )


def collection_exists(collection_name, q_client):
    collections = q_client.get_collections().collections
    return any(collection.name == collection_name for collection in collections)


def add_objects_for_collection(collection_name, q_client, data):
    for index, element in enumerate(data):
        operation_info = q_client.upsert(
            collection_name=collection_name,
            wait=True,
            points=[
                PointStruct(
                    id=index + 1,
                    vector=element["embedding"],
                    payload={"text": element["text"]},
                ),
            ],
        )


def get_all_data_from_collection(collection_name, q_client):
    print(
        q_client.scroll(
            collection_name=collection_name,
            with_payload=True,
            with_vectors=True,
        )
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide file that contains embeddings.")
        sys.exit(1)

    file_name = sys.argv[1]

    client = QdrantClient(url="http://localhost:6333")
    DIRECTORY_PATH = "./data/embeddings"
    try:
        json_data, collection_names = read_from_directory(DIRECTORY_PATH, file_name)
        create_collections(collection_names, client, json_data)
        for index, collection_name in enumerate(collection_names):
            add_objects_for_collection(collection_name, client, json_data[index])
            # get_all_data_from_collection(collection_name, client)
    except Exception as error:
        print(error)
    finally:
        client.close()
