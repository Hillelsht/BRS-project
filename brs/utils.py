########################
# Author: Hillel Shtein
# Date: 05/09/2023
########################

import yaml
import json
import pandas as pd
import pickle

from pathlib import Path
from datetime import datetime

def read_yaml(file_path: Path) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
    


def write_yaml(file_path: Path, dictionary: dict):
    """
    Write dict to yaml
    Input: file_path - path where to save the file
           dictionary - dict to be saved
    """
    with open(file_path, 'w') as file:
        yaml.dump(dictionary, file)


def write_json(obj: object, file_path: Path, ensure_ascii=False, indent=2):
    with open(f"{file_path}", "w", encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=ensure_ascii, indent=2)


def read_json(file_path: Path) -> dict:
    with open(f"{file_path}", "r", encoding='utf-8') as f:
        return json.load(f)
    


def read_pickle(file_path: Path) -> dict:
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def write_pickle(file_path: Path, model):
    """
    Write pickle file
    Input: file_path - path where to save the file
           model - model to be saved
    """
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)


def print_time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        print(f'{func.__module__}.{func.__name__} started at {start_time}')
        result = func(*args, **kwargs)
        print(f'{func.__module__}.{func.__name__} completed at {datetime.now().strftime("%H:%M:%S")}, it took {datetime.now() - start_time}')
        return result
    return wrapper