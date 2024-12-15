"""
save_data.py - Utility functions for saving and retrieving data from a JSON file.
"""

import json
import os
from typing import Any


def init_save_file(save_path: str, count_pieces: int):
    """
    Initialize a JSON file with a structure for storing data for each piece.

    Args:
        save_path (str): The directory where the `data.json` file will be saved.
        count_pieces (int): The number of pieces to initialize in the file.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    json_save_path = os.path.join(save_path, "data.json")
    initial_data = {f"piece_{i}": {} for i in range(0, count_pieces)}

    with open(json_save_path, mode="w") as json_file:
        json.dump(initial_data, json_file, indent=4)

    print(f"Initialized JSON file at {json_save_path}")


def save_data_for_piece(
    save_path: str,
    piece_number: int,
    variable_name: str,
    data: Any,
    avoid_print: bool = False,
):
    """
    Save a specific variable for a given piece in the JSON file.

    Args:
        save_path (str): The directory containing the `data.json` file.
        piece_number (int): The name of the piece (e.g., 'piece_1').
        variable_name (str): The name of the variable to save.
        data (Any): The data to save under the variable.
    """
    json_save_path = os.path.join(save_path, "data.json")

    if not os.path.exists(json_save_path):
        raise FileNotFoundError(f"No data.json found at {json_save_path}")

    with open(json_save_path, mode="r") as json_file:
        contents = json.load(json_file)

    piece = f"piece_{piece_number}"
    if piece not in contents:
        raise KeyError(f"{piece}  not initialized in data.json")

    contents[piece][variable_name] = data

    with open(json_save_path, mode="w") as json_file:
        json.dump(contents, json_file, indent=4)

    if not avoid_print:
        print(f"Saved data for {piece}: {variable_name} = {data}")


def get_data_for_piece(save_path: str, piece_number: int, variable_name: str) -> Any:
    """
    Retrieve a specific variable's data for a given piece from the JSON file.

    Args:
        save_path (str): The directory containing the `data.json` file.
        piece_number (int): The name of the piece (e.g., 'piece_1').
        variable_name (str): The name of the variable to retrieve.

    Returns:
        Any: The data stored under the specified variable.
    """
    json_save_path = os.path.join(save_path, "data.json")

    if not os.path.exists(json_save_path):
        raise FileNotFoundError(f"No data.json found at {json_save_path}")

    with open(json_save_path, mode="r") as json_file:
        contents = json.load(json_file)

    piece = f"piece_{piece_number}"
    if piece not in contents or variable_name not in contents[piece]:
        raise KeyError(f"Data for {piece} with variable '{variable_name}' not found.")

    return contents[piece][variable_name]
