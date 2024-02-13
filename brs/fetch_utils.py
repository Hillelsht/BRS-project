########################
# Author: Hillel Shtein
# Date: 05/09/2023
########################

import argparse
import pandas as pd
from pydantic import BaseModel, validator
from pathlib import Path
from brs.utils import read_yaml
from typing import Any, List, Literal, Optional


def get_args() -> argparse.Namespace:
    """
    This function defines the arguments need to be passed to fetch_data
    Output: args - an object that contains all collected arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperparams_path', nargs='?', type=str)
    args = parser.parse_args()
    return args




class LabelingParams(BaseModel):

    to_label: bool
    label_date: str



class LearningParams(BaseModel):
    time_seperator: list
    #imputer_type: Literal['MeanMedianImputer']
    #imputation_method: Literal['mean', 'median']
    #normalization_type: Literal['MinMaxScaler']
    model_params: dict


class Hyperparams(BaseModel):
    """
    This class defines the schema for hyperparameters using pydantic library
    """

    output_dir: str
    apis: list
    instruments: dict
    to_fetch: bool
    fetch_date: str
    historical_params: Any
    to_preprocess: bool
    features_list: list
    labeling: LabelingParams
    learning: LearningParams


def get_hyperparams(file_path: Path) -> Hyperparams:
    """
    This function loads the hyperparameters from yaml file into the schema defined using pydantic library
    Input file_path - path to yaml file with the hyperparameter
    Output - a dictionary that allows hinting
    """
    hyperparams = read_yaml(file_path)
    hyperparams = Hyperparams(**hyperparams)
    return hyperparams