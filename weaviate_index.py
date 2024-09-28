import weaviate
import json
import os
from weaviate.classes.config import Property, DataType
from weaviate.classes.query import Filter


def weaviate_connect():
    return weaviate.connect_to_local()


def read_json_file(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)

    return data


def read_from_directory(directory_path):
    json_data = []
    collection_names = []

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        collection_names.append(file_path.split("\\")[1].replace(".json", ""))
        json_data.append(read_json_file(file_path))
    return json_data, collection_names


def create_collections(collection_names, weaviate_client):
    for collection_name in collection_names:
        if not weaviate_client.collections.exists(collection_name):
            weaviate_client.collections.create(
                collection_name,
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="embedding", data_type=DataType.NUMBER_ARRAY),
                ],
            )


def add_objects_for_collection(collection_name, weaviate_client, data):
    collection = weaviate_client.collections.get(collection_name)
    collection.data.delete_many(where=Filter.by_property("text").like("*"))
    for element in data:
        collection.data.insert(
            {"text": element["text"], "embedding": element["embedding"]}
        )


if __name__ == "__main__":
    client = weaviate_connect()
    DIRECTORY_PATH = "./data/embeddings"

    if client.is_ready():
        try:
            json_data, collection_names = read_from_directory(DIRECTORY_PATH)
            create_collections(collection_names, client)
            for index, collection_name in enumerate(collection_names):
                add_objects_for_collection(collection_name, client, json_data[index])
        except Exception as error:
            print(error)
        finally:
            client.close()
