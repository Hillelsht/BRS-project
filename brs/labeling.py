import pandas as pd
from brs.utils import print_time_decorator
from brs.fetch_utils import Hyperparams
from pathlib import Path
from datetime import date

class Labeler:
    """
    preprocess data after fetching
    """
    def __init__(self, df: pd.DataFrame, hyperparams: Hyperparams)  -> None:
        self.df = df
        self.hyperparams = hyperparams
        # set output directory
        self.output_dir = Path(
            hyperparams.output_dir, 'label', f"label_{date.today()}")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def create_labels(self) -> None:
        """
        create Binary Label for Price Direction
        """
        # Create continuous label
        # Shift the 'Close' column by -1 to compare the current closing price with the next hour's closing price
        self.df['Label_Future_Close'] = self.df['Close'].shift(-1)

        # Create binary label
        # Create a binary label where 1 indicates the price will go up, and 0 indicates the price will go down or stay the same
        self.df['Label_Price_Direction'] = (self.df['Label_Future_Close'] > self.df['Close']).astype(int)

        # Drop the last row as it will have NaN value for 'Future_Close'
        self.df = self.df[:-1]


    def save_data(self, df: pd.DataFrame):
        df.to_csv(Path(self.output_dir, 'df.csv'))
        df.to_parquet(Path(self.output_dir, 'df.parquet'))

    @print_time_decorator
    def label(self)   -> pd.DataFrame:
        """labeling main function
        creates 2 types of labels:

        1 Binary Label for Price Direction
        2 Continuous Label for Predicted Price

        input:
        df w/o label
        hyperparams

        output:
        df with label
        """

        # 1. create Binary Label for Price Direction
        self.create_labels()


        return self.df