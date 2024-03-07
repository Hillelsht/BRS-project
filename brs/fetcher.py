from typing import Type, List, Any, Union
from pathlib import Path
from datetime import date

import pandas as pd

from API_bursas import *



class BrsFetcher:
    def __init__(self, 
                 hyperparams: dict, 
                 data_type: str = "historical"):
        self.data_type = data_type
        self.instrument_list = hyperparams.instruments
        self.api_resource_list=hyperparams.apis
        self.period=hyperparams.historical_params['period']
        self.interval=hyperparams.historical_params['interval']
        # set output directory
        self.output_dir = Path(
            hyperparams.output_dir, 'fetch', f"fetch_{date.today()}")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def fetch_data(self) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        all_data_for_all_apis = []
        
        for api_resource_str in self.api_resource_list:
            api_resource = eval(api_resource_str)
            merged_data = []
            
            if self.data_type == "historical":
                for symbol, market in self.instrument_list.items():
                    finance_data = api_resource(symbol, self.period, self.interval)
                    data = finance_data.get_historical_data()
                    data['Instrument'] = symbol
                    data['Market'] = market
                    data.reset_index(inplace=True)
                    merged_data.append(data)

                all_data_for_all_apis.append(pd.concat(merged_data, ignore_index=True))
        
            else:
                all_data = {}
                for symbol in self.instrument_list:
                    finance_data = api_resource(symbol)
                    data = finance_data.get_real_time_data()
                    all_data[symbol] = data
                real_time_df = pd.DataFrame(all_data).T.reset_index()
                real_time_df['Instrument'] = real_time_df['index']
                all_data_for_all_apis.append(real_time_df.drop('index', axis=1))

        return pd.concat(all_data_for_all_apis, ignore_index=True) if len(all_data_for_all_apis) > 1 else all_data_for_all_apis[0]

    def save_data(self, df: pd.DataFrame):
        df.to_csv(Path(self.output_dir, 'df.csv'))
        df.to_parquet(Path(self.output_dir, 'df.parquet'))
