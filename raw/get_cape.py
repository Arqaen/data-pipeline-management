import pandas as pd

df = pd.read_excel(
    "ie_data.xls",
    sheet_name="Data",
    usecols="A,M",
    skiprows=128,
    engine="xlrd"
)

df.columns = ["Date", "CAPE"]

# eliminar filas vacías
df = df.dropna(subset=["Date"])

# convertir a fecha
df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y.%m") + pd.offsets.MonthEnd(0)

# guardar a CSV
df.to_csv("cape_data.csv", index=False)

print(df)