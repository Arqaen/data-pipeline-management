import pandas as pd
import yfinance as yf

def get_cape_data():
    df = pd.read_excel(
        "ie_data.xls",
        sheet_name="Data",
        usecols="A,M",
        skiprows=128,
        engine="xlrd"
    )
    df.columns = ["Date", "CAPE"]
    df = df.dropna(subset=["Date"])
    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y.%m") + pd.offsets.MonthEnd(0)
    df.to_csv("data/cape_data.csv", index=False)
    print(df)

def get_spy_data():
    df = yf.download("SPY", period="max")
    df.reset_index(inplace=True)
    df.to_csv("data/sp500.csv", index=False)
    print(df)

def get_dxy_data():
    df = yf.download("DX-Y.NYB", period="max")
    df.reset_index(inplace=True)
    df.to_csv("data/dxy.csv", index=False)
    print(df)

if __name__ == "__main__":
    get_dxy_data()