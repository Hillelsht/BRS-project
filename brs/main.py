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
        fetcher_historical = BrsFetcher(instrument_list=hyperparams.instruments, api_resource_list=hyperparams.apis, data_type="historical", period=hyperparams.historical_params['period'], interval=hyperparams.historical_params['interval'])
        df = fetcher_historical.fetch_data()
        df.to_csv('df.csv')

        # Fetch real-time data
        fetcher_real_time = BrsFetcher(instrument_list=hyperparams.instruments, api_resource_list=hyperparams.apis, data_type="real_time")
        df_online = fetcher_real_time.fetch_data()
        df_online.to_csv('df_online.csv')

    else:  # load
        print(
            f'The previously fetched data was loaded from the dataset dated {hyperparams.fetch_date}')
        df = pd.read_parquet(Path(hyperparams.output_dir, 'fetch',
                             f"fetch_{hyperparams.fetch_date}", 'df.parquet'))
        
    # 3. preprocessing

    # 4. labeling

    # 5. learning

    print(f'Execution ended at {datetime.now().strftime("%H:%M:%S")}, it took ', datetime.now() - start_time)
    return


if __name__ == '__main__':
    main()
