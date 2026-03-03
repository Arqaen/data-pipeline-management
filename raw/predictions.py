from typing import List, Optional, Tuple
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from scipy.stats import spearmanr
from xgboost import XGBClassifier,XGBRegressor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# import yfinance as yf
# sp500 = yf.download("^GSPC", progress=False, period="max")
# sp500.reset_index(inplace=True)
# sp500.to_csv("sp500.csv")

def correlation_report(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    corr = df[cols].corr()
    print("\nCorrelation matrix:")
    print(corr.round(3))
    return corr

def _has_cols(df: pd.DataFrame, cols: List[str]) -> bool:
    return all(c in df.columns for c in cols)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close_col = "Close"
    global horizon
    h = horizon

    # EMAs adaptadas al horizonte
    short = max(3, h // 2)        
    mid   = h                   
    long  = h * 2              

    df[f"ema_{short}"] = df[close_col].ewm(span=short, adjust=False).mean()
    df[f"ema_{mid}"]   = df[close_col].ewm(span=mid, adjust=False).mean()
    df[f"ema_{long}"]  = df[close_col].ewm(span=long, adjust=False).mean()
    df[f"ema_{short}_dist"] = df["Close"] / df[f"ema_{short}"] - 1
    df[f"ema_{mid}_dist"] = df["Close"] / df[f"ema_{mid}"] - 1
    df[f"ema_{long}_dist"] = df["Close"] / df[f"ema_{long}"] - 1

    df["ema_spread"] = df[f"ema_{short}"] / df[f"ema_{long}"] - 1

    # RSI adaptado
    rsi_window = 14
    delta = df[close_col].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(rsi_window).mean()
    avg_loss = loss.rolling(rsi_window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[f"rsi_{rsi_window}"] = 100 - (100 / (1 + rs))

    # ROC alineado al horizonte
    df[f"roc_{h}"] = df[close_col].pct_change(h)

    return df

def to_monthly_last(df):
    """
    Convierte cualquier serie (diaria/mensual/trimestral)
    a frecuencia mensual usando el último valor disponible del mes.
    """
    df = df.sort_index()
    df = df.resample("M").last()
    df = df.ffill()
    return df


# CARGAR DATOS
sp500 = pd.read_csv("sp500.csv", parse_dates=["Date"], index_col="Date")       
sp500 = sp500.apply(pd.to_numeric, errors="coerce")
balance = pd.read_csv("balance_fed.csv", parse_dates=["observation_date"], index_col="observation_date")
corp_profit = pd.read_csv("corporate_profit.csv", parse_dates=["observation_date"], index_col="observation_date")
corp_spread = pd.read_csv("corporate_spread.csv", parse_dates=["observation_date"], index_col="observation_date")
fund_rate = pd.read_csv("fund_rate.csv", parse_dates=["observation_date"], index_col="observation_date")
gdp = pd.read_csv("gdp.csv", parse_dates=["observation_date"], index_col="observation_date")
hy_spread = pd.read_csv("high_yield_spread.csv", parse_dates=["observation_date"], index_col="observation_date")
unemp = pd.read_csv("unemployment.csv", parse_dates=["observation_date"], index_col="observation_date")

sp500 = sp500.drop(columns=["Price", "High", "Low", "Open", "Volume"])

# PASAR A MENSUAL (último dato del mes)
balance = to_monthly_last(balance)
sp500 = to_monthly_last(sp500)
corp_profit = to_monthly_last(corp_profit)
corp_spread = to_monthly_last(corp_spread)
fund_rate = to_monthly_last(fund_rate)
gdp = to_monthly_last(gdp)
hy_spread = to_monthly_last(hy_spread)
unemp = to_monthly_last(unemp)



# MERGE
df = sp500.join([
    balance,
    corp_profit,
    corp_spread,
    fund_rate,
    gdp,
    hy_spread,
    unemp
], how="left")





horizon = 10


# FEATURE ENGINEERING 
df["balance_yoy"] = df["WALCL"].pct_change(12)
df["sp500_12m"] = df["Close"].pct_change(12)
df["sp500_horizon"] = df["Close"].pct_change(horizon)
df["gdp_yoy"] = df["GDPC1"].pct_change(12)
df["unemp_change_12m"] = df["UNRATE"].diff(12)
df["fund_rate_change_12m"] = df["FEDFUNDS"].diff(12)

df = add_technical_indicators(df)

h = horizon
short = max(3, h // 2)       
mid   = h                     
long  = h * 2                 
features = [
    # ===== MACRO =====
    "balance_yoy",
    "sp500_12m",
    "sp500_horizon",
    "gdp_yoy",
    "unemp_change_12m",
    "fund_rate_change_12m",
    "BAMLC0A0CM",
    "BAMLH0A0HYM2",

    # ===== TECHNICAL =====
    f"ema_{short}_dist",
    f"ema_{mid}_dist",
    f"ema_{long}_dist",
    "rsi_14",
    f"roc_{horizon}",
]









   



# TARGET  
horizon = horizon
min_train_size = 180   
test_size = 48     



df["future_return"] = df["Close"].shift(-horizon) / df["Close"] - 1
df = df.dropna(subset=["future_return"])
# df["target"] = (df["future_return"] > 0.05).astype(int)
df["target"] = df["future_return"]


# EDA
correlation_report(df, features + ["target"])
# La señal lineal es débil.
# Eso es normal en mercados financieros.
# Ninguna variable tiene correlación fuerte (> 0.3).
# Eso es buena señal:
# 👉 No hay leakage obvio.

print(df.head())
# print(df.tail())




# OBJETIVO
fecha_objetivo = "2001-05-31"
fecha_objetivo = "2035-01-31"
df = df.loc[:fecha_objetivo].copy()















# Variables predictoras
X = df[features]
y = df["target"]


correlations = []
rank_correlations = []
r2_scores = []
mses = []

all_preds = []
all_actuals = []
all_dates = []

last_model = None

# Walk-forward validation
for start in range(min_train_size, len(df) - test_size, 6):

    purge = horizon
    X_train = X.iloc[:start - purge]
    y_train = y.iloc[:start - purge]

    X_test = X.iloc[start:start + test_size]
    y_test = y.iloc[start:start + test_size]

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=1,
        gamma=0,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0,
        reg_lambda=1,
        random_state=42,
        tree_method="hist",
        early_stopping_rounds=100
    )

    # Validation interna
    val_size = int(len(X_train) * 0.2)

    X_tr = X_train.iloc[:-val_size]
    y_tr = y_train.iloc[:-val_size]

    X_val = X_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    last_model = model

    preds = model.predict(X_test)

    all_preds.extend(preds)
    all_actuals.extend(y_test.values)
    all_dates.extend(y_test.index)

    # Métricas importantes en regresión financiera
    corr = np.corrcoef(preds, y_test)[0, 1]
    rank_corr = spearmanr(preds, y_test).correlation
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)

    correlations.append(corr)
    rank_correlations.append(rank_corr)
    r2_scores.append(r2)
    mses.append(mse)

print("Walk-forward Correlation promedio:", np.nanmean(correlations))
print("Walk-forward Rank Correlation promedio:", np.nanmean(rank_correlations))
print("Walk-forward R2 promedio:", np.nanmean(r2_scores))
print("Walk-forward MSE promedio:", np.nanmean(mses))

print()

if last_model is not None:
    last_X = X.iloc[[-1]]
    walk_pred = last_model.predict(last_X)[0]
    print("Predicción retorno 10 meses (walk):", walk_pred)

final_model= XGBRegressor(
    objective="reg:squarederror",
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=1,
    gamma=0,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0,
    reg_lambda=1,
    random_state=42,
    tree_method="hist",
)

final_model.fit(X, y)

last_X = X.iloc[[-1]]
final_pred = final_model.predict(last_X)[0]

print()
print("Última fecha:", X.index[-1])
print("Predicción retorno 10 meses:", final_pred)

# ── Gráfico Walk-Forward: Predicción vs Real ──
wf_df = pd.DataFrame({"date": all_dates, "predicted": all_preds, "actual": all_actuals})
wf_df = wf_df.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(wf_df["date"], wf_df["actual"],  label="Retorno real", linewidth=1.2, alpha=0.85)
ax.plot(wf_df["date"], wf_df["predicted"], label="Predicción", linewidth=1.2, alpha=0.85)
ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
ax.fill_between(wf_df["date"], wf_df["actual"], wf_df["predicted"], alpha=0.15, color="purple")
ax.set_title(f"Walk-Forward: Retorno {horizon}m — Predicción vs Real", fontsize=13)
ax.set_ylabel("Retorno")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
fig.autofmt_xdate()
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("walk_forward_predictions.png")