import os
import pickle

import bs4 as bs
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ta import add_all_ta_features
from torch.utils.data import Dataset, DataLoader


class Technicals:
    def __init__(
            self,
            path_to_model: str = os.path.join(os.getcwd(), 'models', 'pretrained_TA_model', 'ta_model.pt'),
            path_to_scaler: str = os.path.join(os.getcwd(), 'models', 'pretrained_TA_model', 'scaler.pkl')
    ):
        self._path_to_model = path_to_model
        self._path_to_scaler = path_to_scaler

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = Classifier()
        self._model.to(self._device)

        if self._path_to_model is not None:
            self._model.load_state_dict(torch.load(self._path_to_model))

        with open(self._path_to_scaler, 'rb') as fp:
            self._scaler = pickle.load(fp)

    def _update_dataset(self, path_to_data: str = os.path.join(os.getcwd(), 'models', 'data', 'technicals.csv')):

        resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        table = soup.find('table', {'class': 'wikitable sortable'})

        data = pd.DataFrame()

        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text.strip()

            df = Technicals.get_features(ticker, period='9mo')

            if df is None:
                continue

            df['SevenDayDiff'] = df.Close.diff(periods=-7) > 0
            df = df.dropna(subset=["SevenDayDiff"])
            df.SevenDayDiff = df.SevenDayDiff.astype('int32')

            df = df.reset_index().drop(['Date', 'Close'], axis = 1)

            data = data.append(df)

        data.to_csv(path_to_data)
        return data

    @staticmethod
    def get_features(ticker, period='3mo'):
        prices = yf.Ticker(ticker).history(period=period)
        try:
            df = add_all_ta_features(
                prices, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
            )
        except:
            return

        df = df.drop(['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'], axis=1)

        return df

    def _create_dataloaders(self, data, batch_size: int = 64):
        y = data.loc[:, 'SevenDayDiff']
        X = data.loc[:, data.columns != 'SevenDayDiff']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=69)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        with open(self._path_to_scaler, 'wb') as fp:
            pickle.dump(scaler, fp)
        self._scaler = scaler

        X_test = scaler.transform(X_test)

        train_data = Data(torch.FloatTensor(X_train), torch.FloatTensor(y_train.to_numpy()))
        test_data = Data(torch.FloatTensor(X_test), torch.FloatTensor(y_test.to_numpy()))

        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size)

        return train_loader, test_loader

    @staticmethod
    def binary_acc(y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum / y_test.shape[0]
        acc = torch.round(acc * 100)

        return acc

    def _train_epoch(self, dataloader):
        train_loss = 0
        train_acc = 0

        self._model.train()
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(self._device), y_batch.to(self._device)
            self._optimizer.zero_grad()

            y_pred = self._model(X_batch)

            loss = self._criterion(y_pred, y_batch.unsqueeze(1))
            acc = Technicals.binary_acc(y_pred, y_batch.unsqueeze(1))

            loss.backward()
            self._optimizer.step()

            train_loss += loss.item()
            train_acc += acc.item()

        return train_loss, train_acc

    def _validation_epoch(self, dataloader):
        val_acc = 0

        self._model.eval()
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self._device)
                y_test_pred = self._model(X_batch)

                acc = Technicals.binary_acc(y_test_pred, y_batch.unsqueeze(1))

                val_acc += acc.item()

            return val_acc

    def train(self, epochs=50, learning_rate=0.001):
        data = self._update_dataset()
        train_loader, test_loader = self._create_dataloaders(data)

        self._criterion = nn.BCEWithLogitsLoss()
        self._optimizer = optim.Adam(self._model.parameters(), lr=learning_rate)

        for e in range(1, epochs + 1):
            train_loss, train_acc = self._train_epoch(train_loader)
            print(f'Epoch {e + 0:03}: | Loss: {train_loss / len(train_loader):.5f} | Acc: {train_acc / len(train_loader):.3f}')

            val_acc = self._validation_epoch(test_loader)
            print(f'    VAL Acc: {val_acc / len(test_loader):.3f}')

        torch.save(self._model.state_dict(), self._path_to_model)

    def analyze(self, ticker):
        print(f'### ANALYZING TECHNICALS FOR {ticker}')
        # get latest technical indicators
        x = Technicals.get_features(ticker).drop(['Close'], axis=1).iloc[[-1]]
        x = self._scaler.transform(x)

        self._model.eval()
        y = self._model(torch.from_numpy(x).float())

        indicator = 'UP' if torch.round(torch.sigmoid(y)) == 1.0 else 'DOWN'

        print(f'Technical Indicators suggest {ticker} will be {indicator} in 7 days.')

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Number of input features is 83
        self.layer_1 = nn.Linear(83, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x


class Data(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

if __name__ == '__main__':
    Technicals().analyze("PINS")