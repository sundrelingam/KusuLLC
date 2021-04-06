import yfinance as yf
import numpy as np
import pandas as pd

securities = pd.read_csv("securities.csv")
data = securities.iloc[:10,:].copy()

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

dogs = dogs.sort_values(['mean', 'q10'], ascending=[False, True])
sector = dogs.first_valid_index()

def picks(df, sector):
    filtered = df[df["GICS Sector"] == sector]
    q = q10(filtered.pct_chg)
    res = filtered[filtered.pct_chg <= q]["Ticker symbol"]

    return(res)

print(picks(data, sector))