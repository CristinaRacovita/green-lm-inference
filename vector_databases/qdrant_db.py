import sys

sys.path.append("../")

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from utils import read_from_directory


def connnect():
    return QdrantClient(url="http://localhost:6333")


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


def get_k_most_similar(embedding, q_client, collection_name, k):
    items = []
    response = q_client.query_points(
        collection_name=collection_name,
        query=embedding,
        limit=k,
    )
    for query_response in response.points:
        items.append(query_response.payload["text"])
    print(items)
    return items


def count_collection_entries(collection_name, client):
    collection_data = client.scroll(
        collection_name=collection_name,
        with_payload=True,
        with_vectors=True,
    )

    return len(collection_data[0])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide file that contains embeddings.")
        sys.exit(1)

    file_name = sys.argv[1]

    client = connnect()
    DIRECTORY_PATH = "../data/embeddings"
    try:
        json_data, collection_names = read_from_directory(DIRECTORY_PATH, file_name)
        create_collections(collection_names, client, json_data)
        for index, collection_name in enumerate(collection_names):
            embedding_size = len(json_data[index][0]["embedding"])
            add_objects_for_collection(collection_name, client, json_data[index])
            # get_all_data_from_collection(collection_name, client)
            get_k_most_similar([0] * embedding_size, client, collection_name, 2)
    except Exception as error:
        print(error)
    finally:
        client.close()
