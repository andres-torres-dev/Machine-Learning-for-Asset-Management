import pandas as pd
import numpy as np
from os import listdir

stocks = listdir("data/") 

if __name__ == "__main__":
    stockReturn = pd.DataFrame()
    for stock in stocks:
        df = pd.read_csv(f"data/{stock}")
        stockReturn["date"] = df["date"]
        openPrice = np.array(df["open"].values)
        closePrice = np.array(df["close"].values)
        diff = np.subtract(openPrice, closePrice)
        dailyReturn = np.divide(diff, openPrice)
        stockReturn[stock.strip(".csv")] = dailyReturn

    stockReturn.to_csv(f"data/dailyReturns.csv", index=False)
    

                   