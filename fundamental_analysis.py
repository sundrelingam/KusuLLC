import pandas as pd
import numpy as np
import pickle
import os


class Fundamental:
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
        self._data = pd.read_csv(data)

    @staticmethod
    def preprocess(data, imputer):
        data = data.drop(['Ticker', 'Market Capitalization'], axis=1)
        data = pd.DataFrame(imputer.transform(data), columns=data.columns)

        return data

    def analyze_fundamentals(self, ticker: str):
        print(f'### ANALYZING FUNDAMENTAL DATA FOR {ticker}')

        raw = self._data[self._data['Ticker'] == ticker]
        actual = raw['Market Capitalization']
        preprocessed = Fundamental.preprocess(raw, self._imputer)

        ratio = self._model.predict(preprocessed) / actual

        if ratio.item() > 1:
            diff = np.round(ratio.item() - 1, 4) * 100
        else:
            diff = np.round(1 - ratio.item(), 4) * 100

        print(f'Predicted Market Cap is {diff}% {"higher" if ratio.item() > 1 else "lower"} than actual Market Cap.')


if __name__ == '__main__':
    model = Fundamental()
    model.analyze_fundamentals('PINS')