import sys

sys.path.append("../")

import weaviate
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.classes.data import DataObject
from utils import read_from_directory

# If not specified explicitly, the default distance metric in Weaviate is cosine


def connnect():
    return weaviate.connect_to_local()


def create_collections(collection_names, weaviate_client):
    for collection_name in collection_names:
        if weaviate_client.collections.exists(collection_name):
            collection = weaviate_client.collections.get(collection_name)
            collection.data.delete_many(where=Filter.by_property("text").like("*"))

        else:
            weaviate_client.collections.create(
                collection_name,
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                ],
                vectorizer_config=Configure.Vectorizer.none(),
            )


def add_objects_for_collection(collection_name, weaviate_client, data):
    data_objs = []
    for d in data:
        data_objs.append(
            DataObject(
                properties={
                    "text": d["text"],
                },
                vector=d["embedding"],
            )
        )

    collection = weaviate_client.collections.get(collection_name)
    collection.data.insert_many(data_objs)


def get_all_data_from_collection(collection_name, client):
    collection = client.collections.get(collection_name)
    for item in collection.iterator():
        print(item.uuid, item.properties, item.vector)


def get_k_most_similar(embedding, weaviate_client, collection_name, k):
    collection = weaviate_client.collections.get(collection_name)
    response = collection.query.near_vector(
        near_vector=embedding, limit=k, return_metadata=MetadataQuery(certainty=True)
    )
    items = []
    for query_response in response.objects:
        items.append(query_response.properties["text"])

    return items


def count_collection_entries(collection_name, client):
    collection = client.collections.get(collection_name)
    counter = 0

    for _ in collection.iterator():
        counter += 1

    return counter


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide file that contains embeddings.")
        sys.exit(1)

    file_name = sys.argv[1]

    client = connnect()
    DIRECTORY_PATH = "../data/embeddings"

    if client.is_ready():
        try:
            json_data, collection_names = read_from_directory(DIRECTORY_PATH, file_name)
            create_collections(collection_names, client)
            for index, collection_name in enumerate(collection_names):
                embedding_size = len(json_data[index][0]["embedding"])
                add_objects_for_collection(collection_name, client, json_data[index])
                # get_all_data_from_collection(collection_name, client)
                print(get_k_most_similar([0] * embedding_size, client, collection_name, 2))
        except Exception as error:
            print(error)
        finally:
            client.close()
