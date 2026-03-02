from typing import List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import numpy as np

# import yfinance as yf
# sp500 = yf.download("^GSPC", progress=False, period="max")
# sp500.reset_index(inplace=True)
# sp500.to_csv("sp500.csv")

@dataclass
class Config:
    input_csv: str = "data.csv"
    date_col: str = "date"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    volume_col: str = "volume"
    target_col: str = "close"
    feature_cols: Optional[List[str]] = None
    horizon: int = 1
    lags: int = 5
    ma_windows: Tuple[int, ...] = (5, 10, 20)
    test_size: float = 0.2
    random_state: int = 42

def correlation_report(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    corr = df[cols].corr()
    print("\nCorrelation matrix:")
    print(corr.round(3))
    return corr

def _has_cols(df: pd.DataFrame, cols: List[str]) -> bool:
    return all(c in df.columns for c in cols)

def add_technical_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    close_col = cfg.close_col if cfg.close_col in df.columns else cfg.target_col

    # EMA
    for w in cfg.ma_windows:
        df[f"{close_col}_ema_{w}"] = df[close_col].ewm(span=w, adjust=False).mean()

    df["ema_200_dist"] = df["Close"] / df["Close_ema_200"] - 1
    df["ema_50_dist"]  = df["Close"] / df["Close_ema_50"] - 1
    # RSI(14)
    delta = df[close_col].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # ROC(12)
    df["roc_12"] = 100 * (df[close_col] / df[close_col].shift(12) - 1)

    if _has_cols(df, [cfg.high_col, cfg.low_col, close_col]):
        high = df[cfg.high_col]
        low = df[cfg.low_col]
        close_prev = df[close_col].shift(1)
        tr = pd.concat(
            [(high - low), (high - close_prev).abs(), (low - close_prev).abs()],
            axis=1,
        ).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()

        plus_dm = (high.diff()).where((high.diff() > low.diff().abs()) & (high.diff() > 0), 0.0)
        minus_dm = (low.diff().abs()).where((low.diff().abs() > high.diff()) & (low.diff() < 0), 0.0)
        tr_smooth = tr.ewm(alpha=1 / 14, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / tr_smooth)
        minus_di = 100 * (minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / tr_smooth)
        dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
        df["adx_14"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    return df

# FUNCION GENERAL
def to_monthly_last(df):
    """
    Convierte cualquier serie (diaria/mensual/trimestral)
    a frecuencia mensual usando el último valor disponible del mes.
    """
    df = df.sort_index()
    # df = df.dropna()
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
# balance = to_monthly_last(balance)
sp500 = to_monthly_last(sp500)
corp_profit = to_monthly_last(corp_profit)
corp_spread = to_monthly_last(corp_spread)
fund_rate = to_monthly_last(fund_rate)
gdp = to_monthly_last(gdp)
hy_spread = to_monthly_last(hy_spread)
unemp = to_monthly_last(unemp)



# MERGE
df = sp500.join([
    # balance,
    corp_profit,
    corp_spread,
    fund_rate,
    gdp,
    hy_spread,
    unemp
], how="left")


# FEATURE ENGINEERING 
# df["balance_yoy"] = df["WALCL"].pct_change(12)
df["sp500_12m"] = df["Close"].pct_change(12)
df["gdp_yoy"] = df["GDPC1"].pct_change(12)
df["unemp_change_12m"] = df["UNRATE"].diff(12)
df["fund_rate_change_6m"] = df["FEDFUNDS"].diff(12)

features = [
    # ===== MACRO =====
    "sp500_12m",
    "gdp_yoy",
    "unemp_change_12m",
    "fund_rate_change_6m",
    "BAMLC0A0CM",
    "BAMLH0A0HYM2",

    # ===== TECHNICAL =====
    "ema_200_dist",
    "ema_50_dist",
    "rsi_14",
    "roc_12",
]

df = add_technical_indicators(df, Config(
    target_col="Close",
    close_col="Close",
    high_col=None,
    low_col=None,
    ma_windows=(20, 50, 200),
    horizon=10
))









   



# TARGET 10 meses 
horizon = 10
min_train_size = 180   
test_size = 48     



# TARGETS
df["future_return"] = df["Close"].shift(-horizon) / df["Close"] - 1
df["target"] = (df["future_return"] > 0).astype(int)
df = df.dropna(subset=features + ["future_return"])




# EDA
correlation_report(df, features + ["target"])
df = df.dropna(subset=["future_return", "target"])
# La señal lineal es débil.
# Eso es normal en mercados financieros.
# Ninguna variable tiene correlación fuerte (> 0.3).
# Eso es buena señal:
# 👉 No hay leakage obvio.

print(df.head())
# print(df.tail())




# OBJETIVO
fecha_objetivo = "2001-05-31"
# fecha_objetivo = "2035-01-31"
df = df.loc[:fecha_objetivo].copy()















# Variables predictoras
X = df[features]
y = df["target"]

aucs = []
accuracies = []
last_model = None
for start in range(min_train_size, len(df) - test_size,6):

    # Definir ventanas
    X_train = X.iloc[:start]
    y_train = y.iloc[:start]

    X_test = X.iloc[start:start + test_size]
    y_test = y.iloc[start:start + test_size]

    # Entrenar modelo nuevo cada vez
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist"
    )

    model.fit(X_train, y_train)
    last_model = model
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)

    # Solo calcular ROC si hay ambas clases
    if len(np.unique(y_test)) > 1:
        aucs.append(roc_auc_score(y_test, proba))

    accuracies.append(np.mean(pred == y_test))

print("Walk-forward Accuracy promedio:", np.mean(accuracies))
print("Walk-forward ROC AUC promedio:", np.mean(aucs))


print()
# Última observación disponible
if last_model is not None:
    last_X = X.iloc[[-1]]

    walk_pred = last_model.predict(last_X)[0]
    walk_proba = last_model.predict_proba(last_X)[:, 1][0]

    print("Predicción último modelo walk:", walk_pred)
    print("Probabilidad último modelo walk:", walk_proba)




# Predicción final
final_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist"
)

final_model.fit(X, y)

# Tomar última observación
last_X = X.iloc[[-1]]

# Predicción
final_pred = final_model.predict(last_X)[0]
final_proba = final_model.predict_proba(last_X)[:, 1][0]

print()
print("Última fecha:", X.index[-1])
print("Predicción clase (sube=1, baja=0):", final_pred)
print("Probabilidad subida 10 meses:", final_proba)