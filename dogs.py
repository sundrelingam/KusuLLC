import yfinance as yf
import numpy as np
import pandas as pd

securities = pd.read_csv("securities.csv")
data = securities.copy()

def get_movement(x, days = 2):
    period = f"{days}d"
    yfdata = yf.download(tickers=x, period=period, interval='1d')
    try:
        res = yfdata.iloc[0,:].Close/yfdata.iloc[-1,:].Close - 1
    except:
        res = np.nan

    return(res)

data["pct_chg"] = data.apply(lambda row: get_movement(row["Ticker symbol"]), axis = 1)

def q10(x):
    return x.quantile(0.1)

dogs = data.groupby("GICS Sector").agg({'pct_chg': [np.mean, q10]})
dogs.columns = dogs.columns.droplevel()
dogs = dogs.eval("metric = mean - q10")

dogs = pd.eval("dogs[dogs['mean'] >= 0]")

dogs = dogs.sort_values(['metric'], ascending=[False])

print(dogs.head())
sector = dogs.first_valid_index()

def picks(df, sector):
    filtered = df[df["GICS Sector"] == sector]
    q = q10(filtered.pct_chg)
    res = filtered[filtered.pct_chg <= q][["Ticker symbol", "pct_chg"]]

    return(res)

print(picks(data, sector))