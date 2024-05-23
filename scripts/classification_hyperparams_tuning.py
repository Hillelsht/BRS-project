###################
# Author: Hillel Shtein
# Date: 21/12/2023
###################

import optuna
import pandas as pd
import os
import shutil
import sys

from pathlib import Path
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from optuna.samplers import TPESampler

sys.path.append(str(Path().resolve()))
from brs.fetch_utils import get_hyperparams, Hyperparams
from brs.learning import Learner
from brs.utils import write_yaml


class HyperparamsTuning:

    def __init__(self, learn_date: str, hyperparams: Hyperparams, model_params: dict, tune_params: dict, output_dir: str, n_trials: int) -> None:
        self.learn_date = learn_date
        self.hyperparams = hyperparams
        self.model_params = model_params
        self.tune_params = tune_params
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_trials = n_trials
        self.label_type = None

    def __get_data(self) -> tuple:
        '''This function handles the process of loading a dataset, partitioning it into subsets, and 
        preprocessing the data.
        Output:  x_train - A dataframe comprising the features of the training data.
                y_train - A dataframe that houses the labels corresponding to the training data.
                x_valid - A dataframe comprising the features of the validation data.
                y_valid - A dataframe that houses the labels corresponding to the validation data.'''

        # 1. load dataset
        dataset = pd.read_parquet(Path(self.hyperparams.output_dir,
                                  'labeling', f'labeling_{self.learn_date}', 'df_labeled.parquet').resolve())

        # Omit lines with empty label 
        dataset = dataset.dropna(subset='cost_label').reset_index(drop=True)

        learner = CostLearner(dataset, self.hyperparams, None)

        self.cols_dtypes, dataset = learner._Learner__get_col_dtypes(dataset)

        # 2. get training data
        x_train, y_train = split_df_to_features_and_label(
            learner.df.loc[learner.df.SetType == 'train'], self.label_type)

        # 3. preprocess training data
        # Create a dictionary to store the machine learning and preprocessing models.
        models = {'preproc': {}, 'ml': {}}
        x_train, y_train, models['preproc'] = learner.preprocess_data(
            x_train, y_train, models['preproc'], 'train')

        # 4. get validation data
        x_valid, y_valid = split_df_to_features_and_label(
            learner.df.loc[learner.df.SetType == 'val'], self.label_type)

        # 5. preprocess validation data
        x_valid, y_valid = learner.preprocess_data(
            x_valid, y_valid, models['preproc'], 'valid')

        return x_train, y_train, x_valid, y_valid

    def __learn(self, x_train: pd.DataFrame, y_train: pd.DataFrame, x_valid: pd.DataFrame, y_valid: pd.DataFrame, model_name: str, model_params: dict):
        '''This function manages the process of loading a dataset, training a machine learning model, and 
        evaluating its performance.
        Input:  x_train - A dataframe comprising the features of the training data.
                y_train - A dataframe that houses the labels corresponding to the training data.
                x_valid - A dataframe comprising the features of the validation data.
                y_valid - A dataframe that houses the labels corresponding to the validation data.
                model_name - A string representing the name of the model to be trained.
                model_params - A dictionary containing the parameters required for training the 
                specified model.
        Output  AUC - The function returns the AUC score for the given model on the validation set.'''

        # 1. train models
        model = train_single_model(x_train, y_train, model_params)

        # 2. execute validation
        results_dict = dict()
        # make prediction
        y_valid_pred = model.predict(x_valid.values)

        results_dict[model_name] = calc_cost_model_results(model_name, y_valid.values.flatten(), y_valid_pred, 'valid', self.output_dir)
            #model_name, y_valid.values, y_valid_pred, y_valid_prob, 'valid', self.output_dir)

        # Optimize for AUC, it can be change to other metrics
        return results_dict[model_name]["valid_MAPE"]

    # Define the objective function for Optuna optimization
    def __objective(self, trial, model, x_train, y_train, x_valid, y_valid):
        '''This function yields the value we aim to optimize, either through minimization 
        or maximization. In this particular instance, our objective is to maximize the AUC 
        (Area Under the Curve) metric.
        Input:  trial - Optuna Trial object, which is used to suggest hyperparameters.
                model - This is a string specifying the name of the model being optimized.
        Output: valid_result - The metric we aim to optimize.'''

        # store the hyperparameters for the model being trained and their suggested values
        # for each round of the hyperparameter tuning.
        params = {
            param: trial.suggest_categorical(param, values)
            for param, values in self.tune_params[model].items()
        }

        # merge with default parameters, `params` takes precedence if a key is present in both
        model_params = {**self.model_params[model], 'train_params': {
            **self.model_params[model]['train_params'], **params}}

        valid_result = self.__learn(
            x_train, y_train, x_valid, y_valid, model, model_params)
        return valid_result

    def optimize(self) -> None:
        '''This function is responsible for performing hyperparameter tuning for each specified machine 
        learning model in order to find the optimal set of hyperparameters that maximize the model's performance.'''

        # preprocess the datasets
        x_train, y_train, x_valid, y_valid = self.__get_data()

        # optimization and hyperparameter tuning for each model
        for model in self.tune_params.keys():
            # for a model's accuracy/AUC/AP, you would choose 'maximize', for loss, you would choose 'minimize'.
            # path to save the study object
            abs_path = os.path.abspath(self.output_dir.parent)

            # Replace backslashes with forward slashes
            abs_path = abs_path.replace('\\', '/')

            file_path = f'sqlite:///{abs_path}/hyperparams_tune/study_file.db'
            study = optuna.create_study(storage=file_path,
                                        direction='minimize', sampler=TPESampler())

            # Increase/reduce the number of trials as needed
            study.optimize(lambda trial: self.__objective(
                trial, model, x_train, y_train, x_valid, y_valid), n_trials=self.n_trials)

            # Get the best parameters
            print(f"{model} best parameters:", study.best_params)

            # save all trials params and results
            running_params = dict()
            for i, trial in enumerate(study.trials):

                # Add trial parameters to the dictionary
                running_params[trial.number] = {
                    "value": trial.value, "params": trial.params}

            # Save the running parameters in YAML format
            file_path = Path(self.output_dir.parent, 'hyperparams_tune',
                             f'{model}_running_params.yaml')
            write_yaml(file_path, running_params)
            # 'study' is the optimized Optuna study
            best_trial_dict = dict()
            best_trial = study.best_trial

            best_trial_dict['trial_number'] = best_trial.number
            best_trial_dict['AUC'] = best_trial.value
            best_trial_dict['best_params'] = best_trial.params

            # Save the best parameters in YAML format
            file_path = Path(self.output_dir.parent, 'hyperparams_tune',
                             f'{model}_best_params.yaml')
            write_yaml(file_path, best_trial_dict)

        # delete results directory
        file_path = Path(Path(self.output_dir.parent),
                         'hyperparams_tune', 'results')
        if os.path.exists(file_path):
            shutil.rmtree(file_path)


def main():
    # *******change parameters - learning date and hyperparams path *******
    learn_date = '2024-05-23'
    n_trials = 300  # The number of different sets of hyperparameters to test.
    hyperparams = get_hyperparams(Path('configs/hyperparams.yml'))
    model_params = hyperparams.learning.model_params
    output_dir = f'data/learning/learning_{learn_date}/results/hyperparams_tune'

    # Define the parameter tuning options for each model.
    tune_params = {
        'xgboost': {
            'learning_rate': [0.05, 0.075, 0.1, 1, 10, 40],  # Slightly narrow down to focus on optimal range
            'n_estimators': [1, 10, 100, 200, 300 ],  # Lower numbers may help prevent overfitting
            'max_depth': [1, 6, 8, 10],  # Reduce maximum depth to prevent overfitting
            'min_child_weight': [0.001, 1, 2, 3, 10],  # Slight increase to encourage more conservative models
            'reg_alpha': [0, 0.5, 1, 5],  # Increase L1 regularization slightly
            'reg_lambda': [1, 2, 3, 10],  # Increase L2 regularization to encourage more generalization
            'gamma': [ 0.5, 1, 1.5, 10],  # Adjust to explore more conservative tree splits
            'subsample': [0.6, 0.7],  # Narrow down based on previous best
            'colsample_bytree': [0.6, 0.65, 0.7],  # Narrow down based on previous best
            'max_delta_step': [0, 1],  # Keep as is, given no clear direction on its impact
        }

        #'lightgbm': {
        #    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        #    'n_estimators': [50, 100, 500, 1000, 1500],
        #    'max_depth': [8, 10, 12, 15, 20, 30],
            # sets the maximum number of leaves (or terminal nodes) that can be created in any tree.
        #    'num_leaves': [31, 63, 127, 255],
            # tells the model the minimum number of records a leaf node can have.
        #    'min_child_samples': [10, 20, 30, 40, 50],
        #    'reg_alpha': [0, 1],
        #    'reg_lambda': [0, 1],
        #    'subsample': [0.8, 0.9, 1.0],
        #    'colsample_bytree': [0.7, 0.8, 0.9]
        #}
    }
    # ***************************************************************************

    hyper_tune = HyperparamsTuning(
        learn_date, hyperparams, model_params, tune_params, output_dir, n_trials)
    hyper_tune.optimize()
    return


if __name__ == '__main__':
    main()