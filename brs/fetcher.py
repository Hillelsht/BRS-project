from typing import Type, List, Any, Union
import pandas as pd

from API_bursas import *



class BrsFetcher:
    def __init__(self, 
                 instrument_list: List[str], 
                 api_resource_list: List[str], 
                 data_type: str = "historical", 
                 period: str = "1y", 
                 interval: str = "1d"):
        self.instrument_list = instrument_list
        self.api_resource_list = api_resource_list
        self.data_type = data_type
        self.period = period
        self.interval = interval

    def fetch_data(self) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        all_data_for_all_apis = []
        
        for api_resource_str in self.api_resource_list:
            api_resource = eval(api_resource_str)
            merged_data = []
            
            if self.data_type == "historical":
                for symbol in self.instrument_list:
                    finance_data = api_resource(symbol, self.period, self.interval)
                    data = finance_data.get_historical_data()
                    data['Instrument'] = symbol
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

