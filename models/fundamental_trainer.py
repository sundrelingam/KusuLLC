from .data.fundamental_data import FundamentalData
import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV


class FundamentalTrainer:
    def __init__(self, update_data: bool = False):
        if update_data:
            FundamentalData(os.path.join(os.getcwd(), 'data')).update()

        data = pd.read_csv(os.path.join(os.getcwd(), 'data', 'fundamentals.csv'))
        self._features = data.drop(['Ticker', 'Market Capitalization'], axis=1)
        self._labels = data['Market Capitalization']
        params_path = os.path.join(os.getcwd(), 'pretrained_FA_model', 'best_params.pkl')

        if os.path.exists(params_path):
            with open(params_path, 'rb') as fp:
                params = pickle.load(fp)
            self._model = RandomForestRegressor(**params)
        else:
            self._model = RandomForestRegressor()

        self._impute()

    def _impute(self):
        # knn imputation requires nomalization which has adverse effects on rf performance
        imputer = SimpleImputer(strategy='median')
        imputer.fit(self._features)
        with open(os.path.join(os.getcwd(), 'pretrained_FA_model', 'imputer.pkl'), 'wb') as fp:
            pickle.dump(imputer, fp)

        self._features = pd.DataFrame(imputer.transform(self._features), columns=self._features.columns)

    def tune(self,
             param_grid: dict = {
                'bootstrap': [False],
                'max_depth': [10, 100, None],
                'max_features': ['sqrt'],
                'min_samples_leaf': [2, 5, 10],
                'min_samples_split': [2, 5, 10],
                'n_estimators': [100, 500, 1000]
            }
             ):

        grid_search = GridSearchCV(estimator=self._model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(self._features, self._labels)
        print("Best parameters: ", grid_search.best_params_)

        with open(os.path.join(os.getcwd(), 'pretrained_FA_model', 'best_params.pkl'), 'wb') as fp:
            pickle.dump(grid_search.best_params_, fp)

        self._model = RandomForestRegressor(**grid_search.best_params_)

    def train(self):
        self._model.fit(self._features, self._labels)
        with open(os.path.join(os.getcwd(), 'pretrained_FA_model', 'rf_model.pkl'), 'wb') as fp:
            pickle.dump(self._model, fp)


if __name__ == '__main__':
    FundamentalTrainer().train()
