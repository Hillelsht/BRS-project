import pandas as pd
import numpy as np
import importlib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, DetCurveDisplay, RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix, det_curve, roc_curve, \
    precision_recall_curve, average_precision_score, recall_score, precision_score, auc

from brs.utils import print_time_decorator, write_json, write_pickle
from brs.fetch_utils import Hyperparams
from datetime import datetime
from pathlib import Path
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import xgboost




class Learner:

    def __init__(self, df: pd.DataFrame, hyperparams: Hyperparams, label: str)  -> None:
        self.df = df
        self.hyperparams = hyperparams
        self.label = label
        # Split df into Sets
        self.df = self.__split_data_to_sets(df)

    def __split_data_to_sets(self, df: pd.DataFrame) -> pd.DataFrame:
        '''This function takes the dataframe resulting from the preprocessing and labeling
        process and splits it into train, valid, and test sets. The splitting operation is
        performed based on the time when the patient enters the hospital.
        input:  df - A dataframe after preprocessing and labeling
        output: df - A dataframe containing the "SetType" column indicating whether each
                entry belongs to the train, validation, or test set.
                The function additionally saves the split dataset.'''

        # Integrate 'SetType' columns into the DataFrame to accommodate the assignment of data into training, validation,
        # and testing subsets.
        df['SetType'] = np.nan

        # Split the dataset into train, valid, and test sets based on their arrival time to the hospital and 'medical_center' is in 'train/test _med_center' list
        df.loc[df['Datetime'] <
               self.hyperparams.learning.time_seperator[0], 'SetType'] = 'train'
        df.loc[df['Datetime'].between(
            self.hyperparams.learning.time_seperator[0], self.hyperparams.learning.time_seperator[1], inclusive='left'), 'SetType'] = 'val'
        df.loc[df['Datetime'].between(
            self.hyperparams.learning.time_seperator[1], self.hyperparams.learning.time_seperator[2], inclusive='left'), 'SetType'] = 'test'

        return df


    def split_df_to_features_and_label(self, df) -> tuple:
        '''This function partitions the input dataframe into two distinct dataframes, one for features (x) and another for labels (y).
        Input: df - a dataframe comprising both feature and label columns
                label_type - any_time_in_the_futre/survival etc
        Output: x_df - a dataframe encapsulating all features, excluding label columns
                y_df - a dataframe exclusively consisting of label columns, devoid of any feature columns'''

        # remove all columns that start with 'Label_' - naming convention, labels start with 'Label_'
        x_df = df.loc[:, ~df.columns.str.startswith('Label_')]

        y_df = df[["Datetime",
                "Instrument", "Market", self.label]].copy()

        return x_df, y_df
    
    def train_models(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> dict:
        '''The function is training the machine learning models using the training data.
        input:  x_train - Dataframe containing the feature of the training data.
                y_train - Dataframe containing the labels for the training data
        output: ml_models - A dictionary, post-population with instances of the machine learning models.
        The train_models function is not an internal function of the Learn class, as it is also used by other
        functions outside of the Learn class.'''

        ml_models = dict()
        # 1. Train models
        for model_name, model_params in self.hyperparams.learning.model_params.items():
            print(
                f'Initiate the training process for the model {model_name} at {datetime.now().strftime("%H:%M:%S")}')
            model = self.train_single_model(x_train, y_train, model_params)
            ml_models[model_name] = model
        return ml_models
    
        
    def calc_model_results(self, model_name: str, y_df: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, set_type: str, output_dir: Path) -> pd.Series:
        '''This function is designed to calculate various metrics, generate visualizations, and store resulting figures.
        Input:  model_name - the name of the machine learning model
                y_df - a dataframe featuring the ground truth labels for training data
                y_pred - a dataframe with predicted labels for training data
                y_prob - a dataframe holding predicted probabilities for training data labels
                set_type - train, val or test
                output_dir - The dictionary where the resultant data is stored.
        Output: model_results - a series gathering all calculated results.'''
        model_results = pd.Series(dtype=object)
        y_df = y_df.astype(int)
        y_pred = y_pred.astype(int)

        # 1. Calculate and display graphs
        fig, ax = plt.subplots(2, 2, figsize=(14, 12))

        # Confusion matrix
        cm = confusion_matrix(y_df, y_pred)
        ConfusionMatrixDisplay(cm).plot(ax=ax[0, 0])

        # Det curve
        fpr_det, fnr_det, _ = det_curve(y_df, y_prob)
        DetCurveDisplay(fpr=fpr_det, fnr=fnr_det,
                        estimator_name=model_name).plot(ax=ax[0, 1])
        # ROC curve
        fpr, tpr, _ = roc_curve(y_df, y_prob)
        roc_auc = auc(fpr, tpr)
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                        estimator_name=model_name).plot(ax=ax[1, 0])

        # Precision recall
        precision, recall, _ = precision_recall_curve(y_df, y_prob)
        ap = average_precision_score(y_df, y_prob)
        PrecisionRecallDisplay(precision=precision, recall=recall,
                            average_precision=ap, estimator_name=model_name).plot(ax=ax[1, 1])

        # 2. Save parameters
        model_results[f"{set_type}_AUC"] = roc_auc
        model_results[f"{set_type}_AP"] = ap
        model_results[f"{set_type}_recall"] = recall_score(y_df, y_pred)
        model_results[f"{set_type}_precision"] = precision_score(y_df, y_pred)
        model_results = model_results.round(decimals=3)
        model_results['model_name'] = model_name
        model_results[f"{set_type}_y_pred"] = y_pred
        model_results[f"{set_type}_y_prob"] = y_prob
        model_results[f"{set_type}_y_df"] = y_df

        # 3. Save figures and corresponding label vectors
        results_path = Path(output_dir, "results")
        results_path.mkdir(parents=True, exist_ok=True)

        # Save label vectors
        y_dfs = {'y_df': y_df, 'y_pred': y_pred, 'y_prob': y_prob}

        for name, y in y_dfs.items():
            y = pd.DataFrame(y)
            y.columns = y.columns.astype(str)
            y.to_parquet(
                Path(results_path, f"{model_name}_{set_type}_{name}.parquet").resolve())

        # Save figures
        fig.savefig(
            str(Path(results_path, f"{model_name}_{set_type}_results.jpg").resolve()))
        plt.close(fig)

        return model_results

    
    def evaluate_models(self, x_df: pd.DataFrame, y_df: pd.DataFrame, ml_models: dict, set_type: str) -> pd.DataFrame:
        '''the function is evaluating the models on validation data
        input:  x_df - Dataframe containing the feature of the training data.
                y_df - Dataframe containing the labels for the training data
                ml_models - A dictionary, post-population with instances of the machine learning models.
                set_type - string suffix containing the values 'train', 'valid' or 'test'
        output: results_df - Amalgamate the training results from all models into a unified dataframe
        The evaluate_models function is not an internal function of the Learn class, as it is also used by other 
        functions outside of the Learn class.'''

        results_dict = dict()
        for model_name, model in ml_models.items():
            # predict probabilities, [:, 1] take the probabilities to predict 1
            y_prob = model.predict_proba(x_df.values)[:, 1]

            # make prediction
            y_pred = model.predict(x_df.values)

            results_dict[model_name] = self.calc_model_results(
                model_name, y_df.values, y_pred, y_prob, set_type, self.output_dir)

        # Aggregate the validation/test results from all models into a unified dataframe
        results_df = pd.concat(
            results_dict, axis=1, ignore_index=True).T.set_index('model_name', drop=True)
        return results_df
        
    def merge_features_and_label(self, x_df: pd.DataFrame, y_df: pd.DataFrame) -> pd.DataFrame:
        '''merge the two dataframes, features (x) and label (y) into a single dataframe
        input:  x_df - dataframe containing all the features without labels
                y_df - dataframe containing all the labels without features
        output: df - dataframe containing both features and label columns.'''
        df = x_df.join(y_df, how='left').reset_index(drop=False)
        return df
    

    def train_single_model(self, x_train: pd.DataFrame, y_train: pd.DataFrame, model_params: dict):
    
        '''This function is responsible for training a specific machine learning model on the training dataset.
        Input:  x_train - A dataframe comprising the features of the training data.
                y_train - A dataframe that houses the labels corresponding to the training data.
                model_name - The designation of the current machine learning model to be trained.
                model_params - The parameters affiliated with the current machine learning model.
        Output: model - The trained model post the fitting process.
        '''
        # Import the module and get the model class.
        module = importlib.import_module(model_params['module_path'])
        model_classifier = getattr(module, model_params['model_classifier'])

        # Instantiate the model with the training parameters.
        model = model_classifier(**model_params['train_params'])

        # train training data
        model.fit(x_train.values, y_train.values.astype(int).ravel(),
                **model_params['fit_params'])
        return model
    
        
    def save_learning_results(self, models: dict, x_train: pd.DataFrame, y_train: pd.DataFrame,
                            x_valid: pd.DataFrame, y_valid: pd.DataFrame,
                            x_test: pd.DataFrame, y_test: pd.DataFrame, results_df: pd.DataFrame, output_dir: Path) -> None:
        '''Save all results including models, metrics, and graphs
        input:  models - containing all models and results
                x_train - training data features dataframe
                y_train - training data labels dataframe
                x_valid - validation data features dataframe
                y_valid - validation data labels dataframe
                x_test - test data features dataframe
                y_test - test data labels dataframe

                results_df - A dataframe containing results from both training and validation processes
                output_dir - path to save the results
        output: No output is generated as the results are saved in the learning directory.'''
        print(
            f'Saving results started at {datetime.now().strftime("%H:%M:%S")}')
        output_folder = Path(output_dir, "results")
        results_df.to_csv(Path(output_folder, 'results_df.csv'))
        # Save the x_train.columns for later use in extracting feature importance.
        write_json(x_train.columns.to_list(), Path(
            output_folder, 'x_train_cols.json'))
        for model_name, model in models['ml'].items():
            write_pickle(Path(output_folder, f'{model_name}.pickle'), model)
        write_pickle(Path(output_folder, 'models.pickle'), models)
        self.merge_features_and_label(x_train, y_train).to_parquet(
            Path(output_folder, 'train_df.parquet'))
        self.merge_features_and_label(x_valid, y_valid).to_parquet(
            Path(output_folder, 'valid_df.parquet'))
        self.merge_features_and_label(x_test, y_test).to_parquet(
            Path(output_folder, 'test_df.parquet'))



    @print_time_decorator
    def learn(self)   -> pd.DataFrame:
        """
        '''This function performs the learning process of machine learning models using the labeled dataset and save the results.
        input: The df is self.df - The table consists of both the features and the labels, encompassing all the relevant information.
        output: No output is generated as the results are saved in the learning directory.'''
        """

        # 1. process training data
        x_train, y_train = self.split_df_to_features_and_label(self.df.loc[self.df.SetType == 'train'])

        # train models
        models = {'ml': {}}
        models['ml'] = self.train_models(x_train, y_train)

        train_results_df = self.evaluate_models(
            x_train, y_train, models['ml'], 'train')
        
        # 2. process validation data
        x_valid, y_valid = self.split_df_to_features_and_label(
            self.df.loc[self.df.SetType == 'val'])

        # get validation results
        valid_results_df = self.evaluate_models(
            x_valid, y_valid, models['ml'], 'valid')

        # 3. process test data
        x_test, y_test = self.split_df_to_features_and_label(
            self.df.loc[self.df.SetType == 'test'])

        # get test results
        test_results_df = self.evaluate_models(
            x_test, y_test, models['ml'], 'test')

        results_df = train_results_df.join(
            valid_results_df, how='inner').join(test_results_df, how='inner')

        # 4. Save all the results, including models, metrics, and graphs
        self.save_learning_results(models, x_train, y_train,
                              x_valid, y_valid, x_test, y_test, results_df, self.output_dir)




