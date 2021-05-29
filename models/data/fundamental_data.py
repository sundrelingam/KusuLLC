import os
import pandas as pd
import simfin as sf


class FundamentalData:
    def __init__(self, dir: str = os.getcwd()):
        self._dir = dir
        sf.set_api_key('free')
        sf.set_data_dir(os.path.join(self._dir, 'simfin'))

        self._industries = sf.load_industries()
        self._prices = sf.load_shareprices(refresh_days=0)
        self._balance = sf.load_balance(variant = "quarterly")
        self._income = sf.load_income(variant = "quarterly")
        self._companies = sf.load_companies()

    def _get_latest_financials(self):
        self._balance = self._balance.sort_values(['Ticker', 'Report Date'],ascending=False)\
            .groupby(['Ticker'])\
            .first()\
            .reset_index()
        self._income = self._income.sort_values(['Ticker', 'Report Date'],ascending=False)\
            .groupby(['Ticker'])\
            .first()\
            .reset_index()

    def _join_datasets(self):
        self._data = self._prices\
            .join(self._balance.set_index('SimFinId'), on = 'SimFinId')\
            .join(self._income.set_index('SimFinId'), on = 'SimFinId', lsuffix = '_dup')\
            .join(self._companies.set_index('SimFinId'), on = 'SimFinId')\
            .join(self._industries, on = 'IndustryId')

    def _clean_columns(self):
        self._data = self._data\
            .drop(['Ticker', 'Open', 'Low', 'High', 'Adj. Close', 'Shares (Basic)', 'Shares (Diluted)', 'SimFinId', 'Currency', 'Fiscal Year', 'Fiscal Period', 'Publish Date', 'Restated Date', 'Company Name', 'IndustryId'], axis=1)\
            .reset_index()

        self._data = self._data.loc[:, ~self._data.columns.str.endswith('_dup')]
        self._data = self._data.loc[:,~self._data.columns.duplicated()]

    def _clean_delisted(self):
        max_date = self._data.Date.max()
        self._data = self._data[self._data.Date == max_date]

        print('Data up-to-date as of ', max_date)
        self._data = self._data.drop(['Date'], axis=1)

    def _get_dummies(self):
        self._data = pd.get_dummies(self._data, columns=['Sector', 'Industry'])

    def _calculate_market_cap(self):
        self._data['Market Capitalization'] = self._data.Close * self._data['Shares Outstanding']

        # if the target is null then we shouldn't use it to train
        self._data = self._data[pd.notnull(self._data['Market Capitalization'])]
        self._data = self._data.drop(['Close', 'Shares Outstanding'], axis=1)

    def save(self):
        self._data.to_csv(os.path.join(self._dir, 'fundamentals.csv'), index=False)

    def update(self):
        self._get_latest_financials()
        self._join_datasets()
        self._clean_columns()
        self._clean_delisted()
        self._get_dummies()
        self._calculate_market_cap()
        self.save()

if __name__ == '__main__':
    FundamentalData().update()
