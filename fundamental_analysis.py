from models.data.fundamental_data import FundamentalData
import numpy as np
import os
import pandas as pd
import pickle


class Fundamentals:
    def __init__(
            self,
            model: str = os.path.join(os.getcwd(), 'models', 'pretrained_FA_model', 'rf_model.pkl'),
            imputer: str = os.path.join(os.getcwd(), 'models', 'pretrained_FA_model', 'imputer.pkl'),
            data: str = os.path.join(os.getcwd(), 'models', 'data', 'fundamentals.csv')
    ):
        with open(model, 'rb') as fp:
            self._model = pickle.load(fp)
        with open(imputer, 'rb') as fp:
            self._imputer = pickle.load(fp)

        self._data_path = data

    def _preprocess(self, data):
        data = data.drop(['Ticker', 'Market Capitalization'], axis=1)
        data = pd.DataFrame(self._imputer.transform(data), columns=data.columns)

        return data

    def analyze(self, ticker: str, update_data: bool = True):
        if update_data:
            print(f'### UPDATING DATA')
            FundamentalData(os.path.join(os.getcwd(), 'models', 'data')).update()
            self._data = pd.read_csv(self._data_path)

        print(f'### ANALYZING FUNDAMENTAL DATA FOR {ticker}')
        self._data = pd.read_csv(self._data_path)

        raw = self._data[self._data['Ticker'] == ticker]
        actual = raw['Market Capitalization']
        preprocessed = self._preprocess(raw)

        ratio = self._model.predict(preprocessed) / actual

        if ratio.item() > 1:
            diff = np.round(ratio.item() - 1, 4) * 100
        else:
            diff = np.round(1 - ratio.item(), 4) * 100

        print(f'Predicted Market Cap is {diff}% {"higher" if ratio.item() > 1 else "lower"} than actual Market Cap.')


if __name__ == '__main__':
    Fundamentals().analyze('PINS')
