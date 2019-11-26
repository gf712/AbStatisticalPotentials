import json
import numpy as np


def serialize_numpy_to_string(array):
    result = []
    for val in array:
        if val == np.inf:
            result.append("np.inf")
        else:
            result.append(str(val))
    result[0] = "np.array([" + result[0]
    result[-1] = result[-1] + "])"
    return ','.join(result)

def serialize_json(var, name):
    """Helper function to serialise a Python dictionary in
    JSON format. Handles opening and closing file.
    """
    with open(name, "w") as f:
        json.dump(var, f)


def deserialize_json(path):
    result = {}
    with open(path, "r") as f:
        result = json.load(f)
    return result
