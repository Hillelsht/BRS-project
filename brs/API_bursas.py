import yfinance as yf

class YahooBursaFetcher:
    def __init__(self, stock_symbol: str, period: str ="1y", interval: str ="1d"):
        self.stock_symbol = stock_symbol
        self.period = period
        self.interval = interval
        self.stock = yf.Ticker(self.stock_symbol)

    def get_historical_data(self):
        historical_data = self.stock.history(period=self.period, interval=self.interval)
        return historical_data

    def get_real_time_data(self):
        real_time_data = self.stock.info
        return real_time_data