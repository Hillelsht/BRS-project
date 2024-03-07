import pandas as pd
from brs.utils import write_json, read_json, print_time_decorator
from brs.fetch_utils import Hyperparams
from pathlib import Path
from datetime import date

class Preprocessor:
    """
    preprocess data after fetching
    """
    def __init__(self, df: pd.DataFrame, hyperparams: Hyperparams)  -> None:
        self.df = df
        self.hyperparams = hyperparams
        self.output_dir = Path(
            hyperparams.output_dir, 'preprocess', f"preprocess_{date.today()}")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def handle_missing_values_prep(self) -> None:
        """main method to clean the data
        handle missing values
        """
        # Handle missing values
        missing_values = self.df.isnull().sum()
        print("Missing values in each column:\n", missing_values)

        # Impute missing values

        # Handle missing values for the numerical columns
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        # TODO check what imputation method is the best for the performance: impute zeros, leave empty values, ffil or mean
        # try mean for the missing numeric values
        # self.df[numerical_cols] = self.df[numerical_cols].fillna(self.df[numerical_cols].mean())
        # try ffil for the missing numeric values
        # self.df[numerical_cols] = self.df[numerical_cols].fillna(method='ffill')
        # try zeros for the missing numeric values
        self.df[numerical_cols] = self.df[numerical_cols].fillna(0)


        # For categorical columns, you might replace missing values with the mode
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        self.df[categorical_cols] = self.df[categorical_cols].fillna(self.df[categorical_cols].mode().iloc[0])

        # Verify if there are any missing values left
        missing_values_after = self.df.isnull().sum()
        print("Missing values after imputation:\n", missing_values_after)



    def format_data_prep(self) -> None:
        """main method to format the data"""
        # process datatime columns
        self.df['Datetime'] = pd.to_datetime(self.df['Datetime'], utc=True).dt.tz_convert('America/New_York')


    def engineer_features_prep(self) -> None:
        """features engineering"""
        # create time-based features
        self.df['day_of_week'] = self.df['Datetime'].dt.dayofweek  # Monday=0, Sunday=6
        self.df['month'] = self.df['Datetime'].dt.month
        self.df['hour'] = self.df['Datetime'].dt.hour  # if the data includes time

        # create technical indicators
        # Calculate moving averages for the 'Close' column
        self.df['MA_5'] = self.df['Close'].rolling(window=5).mean()
        self.df['MA_10'] = self.df['Close'].rolling(window=10).mean()

        # Calculate percentage changes for the 'Close' and 'Volume' columns
        # The method pct_change computes the percentage change from the previous row by default
        self.df['Close_pct_change'] = self.df['Close'].pct_change() * 100  # multiply by 100 to convert to percentage
        self.df['Volume_pct_change'] = self.df['Volume'].pct_change() * 100  # multiply by 100 to convert to percentage




    # Define a function to remove outliers using IQR
    def remove_outliers(self, column_list) -> None:
        '''removes outliers'''
        outlier_indices = []
        print("Original DataFrame shape:", self.df.shape)
        for column in column_list:
            for instrument in self.df['Instrument'].unique():
                # Select the subset of the DataFrame for the current instrument
                instrument_df = self.df[self.df['Instrument'] == instrument]
                
                # Calculate the IQR for the current column in the subset
                Q1 = instrument_df[column].quantile(0.25)
                Q3 = instrument_df[column].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define the bounds for outliers
                lower_bound = Q1 - 50 * IQR # TODO evaluate different approaches, 1.4 coefficient for example
                upper_bound = Q3 + 50 * IQR # TODO evaluate different approaches, 1.4 coefficient for example
                
                # Find indices of outliers within the current instrument and column
                outlier_list = instrument_df[(instrument_df[column] < lower_bound) | (instrument_df[column] > upper_bound)].index
                
                # Append these indices to the list of all outliers
                outlier_indices.extend(outlier_list)

        # Remove duplicate indices to ensure each row is only considered once
        outlier_indices = list(set(outlier_indices))
        print(len(outlier_indices))

        # Drop the outliers from the DataFrame
        self.df = self.df.drop(index=outlier_indices)
        print("W/o outliers DataFrame shape:", self.df.shape)

    
    def save_data(self, df: pd.DataFrame):
        df.to_csv(Path(self.output_dir, 'df.csv'))
        df.to_parquet(Path(self.output_dir, 'df.parquet'))



    @print_time_decorator
    def preprocess(self)   -> pd.DataFrame:
        """preprocess main function"""

        # 1. clean
        # TODO revise the missing values handling method based on UAC influence
        self.handle_missing_values_prep()

        # 2 . format
        self.format_data_prep()

        # 2. engineer features
        self.engineer_features_prep()

        # 3. remove outliers
        self.remove_outliers(self.df.select_dtypes(include=['float64', 'int64']).columns)

        #  keep only relevant columns
        self.df = self.df[self.hyperparams.features_list]

        return self.df