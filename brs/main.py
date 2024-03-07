########################
# Author: Hillel Shtein
# Date: 05/09/2023
########################


import pandas as pd
import sys
from pathlib import Path
from datetime import date, datetime

sys.path.append(str(Path().resolve()))

from brs.fetch_utils import get_args, get_hyperparams
from brs.fetcher import BrsFetcher
from brs.preprocessing import Preprocessor
from brs.labeling import Labeler
from brs.learning import Learner

def main():
    start_time = datetime.now()
    print(f'Execution started at ', start_time)
    
    # 1. preparations (parameters definition)
    args = get_args()
    hyperparams = get_hyperparams(Path('configs', args.hyperparams_path))
    

    # 2. fetch or load data
    df = pd.DataFrame()
    if hyperparams.to_fetch:  # fetch data from brs API
 
        # Fetch historical data
        fetcher_historical = BrsFetcher(hyperparams, data_type="historical")
        df = fetcher_historical.fetch_data()
        fetcher_historical.save_data(df)
        
        # Fetch real-time data (online fetcher is temporary unavailable due to API provider issues)
        fetcher_real_time = BrsFetcher(hyperparams, data_type="real_time")
        #df_real_time = fetcher_real_time.fetch_data()
        #fetcher_historical.save_data(df_real_time)

    else:  # load
        print(
            f'The previously fetched data was loaded from the dataset dated {hyperparams.fetch_date}')
        df = pd.read_parquet(Path(hyperparams.output_dir, 'fetch',
                             f"fetch_{hyperparams.fetch_date}", 'df.parquet'))
        
    # 3. preprocessing
    if hyperparams.to_preprocess:  # run preprocess
        preprocessor = Preprocessor(df, hyperparams)
        df = preprocessor.preprocess()
        preprocessor.save_data(df)
    else:  # load previously saved data
        print(
            f'The previously preprocessed data was loaded from the dataset dated {hyperparams.preprocessing.preprocess_date}')
        df = pd.read_parquet(Path(hyperparams.output_dir, 'preprocessing',
                                  f"preprocess_{hyperparams.preprocessing.preprocess_date}", 'df_preprocessed.parquet'))

    # 4. labeling
    if hyperparams.labeling.to_label:  # run labeling
        labeler = Labeler(df, hyperparams)
        df = labeler.label()
        labeler.save_data(df)
    else:  # load previously saved data
        print(
            f'The previously labeled data was loaded from the dataset dated {hyperparams.labeling.label_date}')

        df = pd.read_pickle(Path(hyperparams.output_dir, 'labeling',
                                 f"labeling_{hyperparams.labeling.label_date}", 'df_labeled.pickle'))

    # 5. learning Label_Price_Direction - Classification Problem
    learner_direction = Learner(df, hyperparams, 'Label_Price_Direction')
    learner_direction.learn()

    # 6. learning Label_Future_Close - Regression Problem
    learner_close = Learner(df, hyperparams, 'Label_Future_Close')
    learner_close.learn()


    print(f'Execution ended at {datetime.now().strftime("%H:%M:%S")}, it took ', datetime.now() - start_time)
    return


if __name__ == '__main__':
    main()
