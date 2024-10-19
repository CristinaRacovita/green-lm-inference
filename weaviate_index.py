import weaviate
from weaviate.classes.config import Property, DataType
from weaviate.classes.query import Filter
from read_files import read_from_directory

# If not specified explicitly, the default distance metric in Weaviate is cosine

def weaviate_connect():
    return weaviate.connect_to_local()


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

def get_all_data_from_collection(collection_name, client):
    collection = client.collections.get(collection_name)
    for item in collection.iterator():
        print(item.uuid, item.properties)

if __name__ == "__main__":
    client = weaviate_connect()
    DIRECTORY_PATH = "./data/embeddings"

    if client.is_ready():
        try:
            json_data, collection_names = read_from_directory(DIRECTORY_PATH)
            create_collections(collection_names, client)
            for index, collection_name in enumerate(collection_names):
                add_objects_for_collection(collection_name, client, json_data[index])
                get_all_data_from_collection(collection_name, client)
        except Exception as error:
            print(error)
        finally:
            client.close()
