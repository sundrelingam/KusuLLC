# @TODO refactor
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.impute import SimpleImputer
import pickle

data = pd.read_csv('.\\data\\fundamentals.csv')

features = data.drop(['Ticker', 'Market Capitalization'], axis=1)
train_labels = data['Market Capitalization']

# knn imputation requires nomalization which has adverse effects on rf performance
imputer = SimpleImputer(strategy='median')
imputer.fit(features)
with open('.\\pretrained_FA_model\\imputer.pkl', 'wb') as fp:
    pickle.dump(imputer, fp)

train_features = pd.DataFrame(imputer.transform(features), columns=features.columns)

rf = RandomForestRegressor()

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(train_features, train_labels)

rf = RandomForestRegressor(**grid_search.best_params_)
rf.fit(train_features, train_labels)

with open('.\\pretrained_FA_model\\rf_model.pkl', 'wb') as fp:
    pickle.dump(rf, fp)
