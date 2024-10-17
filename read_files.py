import json
import os


def read_json_file(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)

    return data


def read_from_directory(directory_path, file_name):
    json_data = []
    collection_names = []

    if file_name:
        file_path = os.path.join(directory_path, file_name)
        file_name_without_type = file_name.replace(".json", "")
        return [read_json_file(file_path)], [file_name_without_type]

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        collection_names.append(file_path.split("\\")[1].replace(".json", ""))
        json_data.append(read_json_file(file_path))
    return json_data, collection_names
