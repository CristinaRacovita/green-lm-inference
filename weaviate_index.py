import weaviate
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.classes.data import DataObject
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
                ],
                vectorizer_config=Configure.Vectorizer.none(),
            )


def add_objects_for_collection(collection_name, weaviate_client, data):
    collection = weaviate_client.collections.get(collection_name)
    collection.data.delete_many(where=Filter.by_property("text").like("*"))
    data_objs = []
    for d in data:
        data_objs.append(DataObject(
            properties={
                "text": d["text"],
            },
            vector=d["embedding"]
        ))

    collection = weaviate_client.collections.get(collection_name)
    collection.data.insert_many(data_objs)
    # with collection.batch.dynamic() as batch:
    #     for element in data:
    #         batch.add_object(
    #             properties={"text": element["text"]},
    #             vector=element["embedding"]
    #         )
    # if len(collection.batch.failed_objects) > 0:
    #     print(f"Failed to import {len(collection.batch.failed_objects)} objects")

def get_all_data_from_collection(collection_name, client):
    collection = client.collections.get(collection_name)
    for item in collection.iterator():
        print(item.uuid, item.properties, item.vector)

def get_k_most_similar(weaviate_client, collection_name, embedding, k):
    collection = weaviate_client.collections.get(collection_name)
    response = collection.query.near_vector(
        near_vector=embedding,
        limit=k,
        return_metadata=MetadataQuery(certainty=True)
    )

    print(response)

if __name__ == "__main__":
    client = weaviate_connect()
    DIRECTORY_PATH = "./data/embeddings"

    if client.is_ready():
        try:
            json_data, collection_names = read_from_directory(DIRECTORY_PATH)
            create_collections(collection_names, client)
            for index, collection_name in enumerate(collection_names):
                add_objects_for_collection(collection_name, client, json_data[index])
                # get_all_data_from_collection(collection_name, client)
                get_k_most_similar(client, collection_name, [
                        0.12,
                        0.87,
                        -0.44,
                        0.66,
                        -0.01,
                        0.23,
                        0.99,
                        -0.78,
                        0.11,
                        0.34
                    ], 1)
        except Exception as error:
            print(error)
        finally:
            client.close()
