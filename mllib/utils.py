import json


def load_json(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def save_json(obj, filepath):
    with open(filepath, "w") as f:
        json.dump(obj, f)
