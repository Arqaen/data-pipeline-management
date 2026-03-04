from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

def correlation_report(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    corr = df[cols].corr()
    print("\nCorrelation matrix:")
    print(corr.round(3))
    return corr

def _has_cols(df: pd.DataFrame, cols: List[str]) -> bool:
    return all(c in df.columns for c in cols)


BASE_DIR = Path(__file__).resolve().parent
HORIZON = 10

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close_col = "Close"
    h = HORIZON

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


def load_series_csv(
    filename: str,
    *,
    date_col: str,
    index_name: Optional[str] = None,
    drop_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    file_path = BASE_DIR / filename
    df = pd.read_csv(file_path, parse_dates=[date_col], index_col=date_col)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.columns = df.columns.str.strip() 
    if index_name:
        df.index.name = index_name

    if drop_columns:
        existing = [column for column in drop_columns if column in df.columns]
        if existing:
            df = df.drop(columns=existing)

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
sp500 = load_series_csv(
    "sp500.csv",
    date_col="Date",
    drop_columns=["Price", "High", "Low", "Open", "Volume"],
)

series_map: Dict[str, pd.DataFrame] = {
    "vix": load_series_csv(
        "vix.csv",
        date_col="Date",
        drop_columns=["Price", "High", "Low", "Open", "Volume"],
    ).rename(columns={"Close": "VIX_Close"}),
    "balance": load_series_csv("balance_fed.csv", date_col="observation_date"),
    "corp_profit": load_series_csv("corporate_profit.csv", date_col="observation_date"),
    "corp_spread": load_series_csv("corporate_spread.csv", date_col="observation_date"),
    "fund_rate": load_series_csv("fund_rate.csv", date_col="observation_date"),
    "gdp": load_series_csv("gdp.csv", date_col="observation_date"),
    "hy_spread": load_series_csv("high_yield_spread.csv", date_col="observation_date"),
    "unemp": load_series_csv("unemployment.csv", date_col="observation_date"),
    "dfii10": load_series_csv("DFII10.csv", date_col="observation_date"),
    "dgs10": load_series_csv("DGS10.csv", date_col="observation_date"),
    "m2sl": load_series_csv("M2SL.csv", date_col="observation_date"),
    "nfci": load_series_csv("NFCI.csv", date_col="observation_date"),
    "permit": load_series_csv("PERMIT.csv", date_col="observation_date"),
    "t10y3m": load_series_csv("T10Y3M.csv", date_col="observation_date"),
    "t10yie": load_series_csv("T10YIE.csv", date_col="observation_date"),
    "sp500_pe_ratio": load_series_csv("sp-500-pe-ratio-price-to-earnings-chart.csv", date_col="date").rename(columns={"value": "sp500_pe_ratio"}),
    # "cape_ratio": load_series_csv("Historic-cape-ratios.csv", date_col="Date").rename(columns={"USA": "cape_ratio"}),
    "cape_data": load_series_csv("cape_data.csv", date_col="Date").rename(columns={"CAPE": "cape_data"}),
}


# PASAR A MENSUAL (último dato del mes)
sp500 = to_monthly_last(sp500)
for name, dataset in list(series_map.items()):
    series_map[name] = to_monthly_last(dataset)



# MERGE
df = sp500.join([
    series_map["vix"],
    series_map["balance"],
    series_map["corp_profit"],
    series_map["corp_spread"],
    series_map["fund_rate"],
    series_map["gdp"],
    series_map["hy_spread"],
    series_map["unemp"],
    series_map["dfii10"],
    series_map["dgs10"],
    series_map["m2sl"],
    series_map["nfci"],
    series_map["permit"],
    series_map["t10y3m"],
    series_map["t10yie"],
    series_map["sp500_pe_ratio"],
    # series_map["cape_ratio"],
    series_map["cape_data"],
], how="left")



print(df)

# FEATURE ENGINEERING 

# release lag aproximado
# df["GDPC1"] = df["GDPC1"].shift(2)
# df["UNRATE"] = df["UNRATE"].shift(1)
# df["PERMIT"] = df["PERMIT"].shift(1)
# df["M2SL"] = df["M2SL"].shift(1)

df["balance_yoy"] = df["WALCL"].pct_change(12)
df["sp500_12m"] = df["Close"].pct_change(12)
df["sp500_horizon"] = df["Close"].pct_change(12)
df["gdp_yoy"] = df["GDPC1"].pct_change(12)
df["unemp_change_12m"] = df["UNRATE"].diff(12)
df["fund_rate_change_3m"] = df["FEDFUNDS"].diff(3)
df["vix_level"] = df["VIX_Close"]
df["vix_3m_change"] = df["VIX_Close"].pct_change(3)
df["m2_yoy"] = df["M2SL"].pct_change(12)
df["permit_yoy"] = df["PERMIT"].pct_change(12)
df["curve_slope_3m_change"] = df["T10Y3M"].diff(3)
df["inflation_expectations_3m_change"] = df["T10YIE"].diff(3)
df["sp500_earnings_yield"] = 1 / df["sp500_pe_ratio"]
df["cape_earnings_yield"] = 1 / df["cape_data"]



df = add_technical_indicators(df)

h = HORIZON
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
    "fund_rate_change_3m",
    "BAMLC0A0CM",
    "BAMLH0A0HYM2",
    "vix_level",
    "vix_3m_change",
    "m2_yoy",
    "NFCI",
    "permit_yoy",
    "DFII10",
    "T10Y3M",
    "curve_slope_3m_change",
    "T10YIE",
    "inflation_expectations_3m_change",
    "sp500_earnings_yield",
    "cape_earnings_yield",

    # ===== TECHNICAL =====
    f"ema_{short}_dist",
    f"ema_{mid}_dist",
    f"ema_{long}_dist",
    "rsi_14",
    f"roc_{HORIZON}",
]




cols_to_drop = [
    "m2_yoy",
    "balance_yoy",
    "vix_3m_change",
    "inflation_expectations_3m_change",
    "DFII10",
    "ema_5_dist",   
    "ema_10_dist", 
    "sp500_12m"
]
df = df.drop(columns=cols_to_drop, errors="ignore")



features = [f for f in features if f not in cols_to_drop]




# usar solo variables con 70% historia
min_history = 0.7   
valid_features = [
    f for f in features
    if df[f].notna().mean() > min_history
]
dropped_features = [f for f in features if f not in valid_features]
print("Features eliminadas por tener poco historial:")
for f in dropped_features:
    print(f)
features = valid_features

   



# TARGET  
min_train_size = 180   
test_size = 48     

df["future_return"] = df["Close"].shift(-HORIZON) / df["Close"] - 1
df = df.dropna(subset=["future_return"])
# df["target"] = (df["future_return"] > 0.05).astype(int)
df["target"] = df["future_return"]
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["target"])
df = df.dropna(subset=features)



# EDA
 
correlation_report(df, features + ["target"])
# La señal lineal es débil.
# Eso es normal en mercados financieros.
# Ninguna variable tiene correlación fuerte (> 0.3).
# Eso es buena señal:
# 👉 No hay leakage obvio.



# =========================================================
# 1️⃣ Rank Correlation (Spearman) feature vs target
# Detecta relación monotónica (no necesariamente lineal)
# En mercados suele ser más informativo que Pearson
# =========================================================

from scipy.stats import spearmanr

print("Spearman rank correlation vs target\n")

for col in features:
    rc = spearmanr(df[col], df["target"]).correlation
    print(f"{col:35s} {rc:.3f}")

# Interpretación:
# > 0.05 consistente ya es interesante en finanzas
# Signo estable > magnitud


# =========================================================
# 2️⃣ Decile / Binning Analysis
# Mide si extremos de la variable predicen retornos distintos
# Es mucho más útil que mirar solo correlación
# =========================================================
feature_to_test = f"roc_{HORIZON}"   # cambia si quieres probar otra
df["bin"] = pd.qcut(df[feature_to_test], 10, labels=False, duplicates="drop")
decile_returns = df.groupby("bin")["target"].mean()
print("\nRetorno medio por decil:")
print(decile_returns)
# Interpretación:
# Ideal: decil 9 >> decil 0
# Relación creciente casi monotónica = señal robusta



# =========================================================
# 3️⃣ Distribución del Target
# Permite ver asimetría, colas y sesgo estructural
# Clave para interpretar R2 y MSE
# =========================================================
import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))
df["target"].hist(bins=50)
plt.title("Distribución del retorno futuro (target)")
plt.axvline(0, linestyle="--")
plt.savefig(BASE_DIR / "target_dist.png")
# Si está muy concentrado en 0 → modelo difícil
# Si hay colas gordas → cuidado con MSE (dominado por outliers)




# =========================================================
# 4️⃣ Estabilidad temporal de la señal
# Una señal buena debe funcionar en distintos regímenes
# =========================================================
df["decade"] = (df.index.year // 10) * 10
print("\nCorrelación por década:\n")
for col in ["sp500_horizon"]:   # puedes probar más features
    for decade, sub in df.groupby("decade"):
        corr = sub[col].corr(sub["target"])
        print(f"{col} - {decade}s: {corr:.3f}")
# Interpretación:
# Si cambia de signo frecuentemente → probablemente ruido
# Señal estable en el tiempo = mucho más valiosa



# =========================================================
# 5️⃣ Autocorrelación del Target
# Mide si el retorno a 10 meses tiene memoria propia
# =========================================================
auto_corr = df["target"].autocorr()
print("\nAutocorrelación del target:", round(auto_corr, 3))
# ~0  → mercado eficiente (normal)
# Alta → ya existe momentum estructural














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

    purge = HORIZON
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
ax.set_title(f"Walk-Forward: Retorno {HORIZON}m — Predicción vs Real", fontsize=13)
ax.set_ylabel("Retorno")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
fig.autofmt_xdate()
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(BASE_DIR / "walk_forward_predictions.png")
