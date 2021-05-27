# @TODO refactor

import simfin as sf
import pandas as pd
import os
print(os.getcwd())

# Set your API-key for downloading data. This key gets the free data.
sf.set_api_key('free')

# Set the local directory where data-files are stored.
# The directory will be created if it does not already exist.
sf.set_data_dir('~/simfin_data/')

industries = sf.load_industries()
prices = sf.load_shareprices()
balance = sf.load_balance(variant = "quarterly")
income = sf.load_income(variant = "quarterly")
companies = sf.load_companies()

balance = balance.sort_values(['Ticker', 'Report Date'],ascending=False).groupby(['Ticker']).first().reset_index()
income = income.sort_values(['Ticker', 'Report Date'],ascending=False).groupby(['Ticker']).first().reset_index()

data = prices.join(balance.set_index('SimFinId'), on = 'SimFinId')
data = data.join(income.set_index('SimFinId'), on = 'SimFinId', lsuffix = '_dup')
data = data.join(companies.set_index('SimFinId'), on = 'SimFinId')
data = data.join(industries, on = 'IndustryId')

data = data.drop(['SimFinId', 'Currency', 'Fiscal Year', 'Fiscal Period', 'Publish Date', 'Restated Date', 'Company Name', 'IndustryId'], axis = 1)
data = data.loc[:, ~data.columns.str.endswith('_dup')]
data = data.loc[:,~data.columns.duplicated()]

# create target
data['Market Capitalization'] = data.Close * data['Shares Outstanding']
data = data.drop(['Date', 'Open', 'Low', 'High', 'Close', 'Adj. Close', 'Shares Outstanding', 'Ticker.1', 'Shares (Basic)', 'Shares (Diluted)'], axis=1)

# if the target is null then we shouldn't use it to train
data = data[pd.notnull(data['Market Capitalization'])]

data = pd.get_dummies(data, columns=['Sector', 'Industry'])
data.to_csv('.\\fundamentals_preprocessed.csv', index=False)
