import pandas as pd
import yfinance as yf
import cloudscraper
from datetime import datetime

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

def get_pmi_data():
    # import investpy
    # data = investpy.economic_calendar(
    #     from_date='01/01/2015',
    #     to_date='01/01/2026'
    # )
    # pmi = data[data['event'].str.contains('ISM Manufacturing PMI')]
    # print(pmi)    
    # import cloudscraper

    scraper = cloudscraper.create_scraper()
    url = "https://endpoints.investing.com/pd-instruments/v1/calendars/economic/events/173/occurrences?domain_id=1&limit=1000"
    response = scraper.get(url)
    print(response.status_code)
    data = response.json()
    print(data)
    rows = []
    for item in data["occurrences"]:
        date = datetime.fromisoformat(item["occurrence_time"].replace("Z", "+00:00"))

        rows.append({
            "observation_date": date.strftime("%Y-%m-%d"),
            "PMI": item["actual"]
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("observation_date")
    df.to_csv("data/pmi.csv", index=False)
    print(df)




# CPB Trade monitor
# https://www.cpb.nl/en/world-trade-monitor/cpb-world-trade-monitor-december-2025

if __name__ == "__main__":
    get_pmi_data()