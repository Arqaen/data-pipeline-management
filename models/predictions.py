from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil
import matplotlib.dates as mdates
import shap 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report,
)


def _sample_param_combo(param_dist: Dict[str, List], rng: np.random.RandomState) -> Dict:
    return {key: rng.choice(values) for key, values in param_dist.items()}


def tune_xgb_random_search_timeval(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_es: pd.DataFrame,
    y_es: pd.Series,
    X_score: pd.DataFrame,
    y_score: pd.Series,
    *,
    fixed_params: Dict,
    param_dist: Dict[str, List],
    n_iter: int = 30,
    random_state: int = 42,
    scale_pos_weight: float = 1.0,
) -> Tuple[Dict, int, float]:
    """Random search con validación temporal (X_tr -> early-stopping -> scoring).

    - Sin RandomizedSearchCV (que no encaja bien con early_stopping + embargo).
    - Score: maximiza ROC-AUC en el bloque de scoring.

    Devuelve: (best_params, best_n_estimators, best_roc_auc)
    """
    rng = np.random.RandomState(random_state)

    best_params: Dict = {}
    best_score = -np.inf
    best_n_estimators = int(fixed_params.get("n_estimators", 5000))

    # Si no hay suficientes clases, log_loss sigue siendo válido con labels=[0,1]
    # pero el tuning puede ser poco informativo. Aun así, lo dejamos correr.
    for _ in range(int(n_iter)):
        params = _sample_param_combo(param_dist, rng)
        model = XGBClassifier(
            **fixed_params,
            **params,
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_es, y_es)],
            verbose=False,
        )
        score_proba = model.predict_proba(X_score)[:, 1]

        if len(np.unique(y_score)) > 1:
            score = float(roc_auc_score(y_score, score_proba))
        else:
            # ROC-AUC no definido con una sola clase; fallback a -logloss para poder comparar.
            score = -float(log_loss(y_score, score_proba, labels=[0, 1]))

        # best_iteration está disponible cuando hay early_stopping
        bi = getattr(model, "best_iteration", None)
        if bi is not None:
            n_estimators = int(bi) + 1
        else:
            n_estimators = int(fixed_params.get("n_estimators", 5000))

        if score > best_score:
            best_score = score
            best_params = params

            # Guardar también el tamaño efectivo del modelo
            best_n_estimators = n_estimators

    print(
        f"[RandomSearch] best ROC-AUC(score)={best_score:.5f} "
        f"best_n_estimators={best_n_estimators} params={best_params}"
    )
    return best_params, best_n_estimators, float(best_score)



def simulate_monthly_dca_roi(
    prices: pd.Series,
    contributions: pd.Series,
) -> pd.DataFrame:
    prices = prices.astype(float)
    contributions = contributions.astype(float).reindex(prices.index).fillna(0.0)

    shares_bought = contributions.div(prices).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    shares = shares_bought.cumsum()
    invested = contributions.cumsum()
    value = shares.mul(prices)
    roi_pct = np.where(invested.values > 0, (value.values - invested.values) / invested.values * 100.0, np.nan)

    return pd.DataFrame(
        {
            "price": prices,
            "contribution": contributions,
            "invested": invested,
            "shares": shares,
            "value": value,
            "roi_pct": roi_pct,
        },
        index=prices.index,
    )

def correlation_report(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    corr = df[cols].corr()
    # pd.set_option('display.max_colwidth', 5)
    # pd.set_option('display.max_colwidth', None)
    print("\nCorrelation matrix:")
    print(corr.round(1))    
    return corr

def _has_cols(df: pd.DataFrame, cols: List[str]) -> bool:
    return all(c in df.columns for c in cols)


BASE = Path(__file__).resolve().parent
BASE_DIR = BASE / "data"
HORIZON = 12

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
    "core_cpi": load_series_csv("CORESTICKM159SFRBATL.csv", date_col="observation_date"),
    "dxy": load_series_csv("dxy.csv", date_col="Date",drop_columns=["High", "Low", "Open", "Volume"]).rename(columns={"Close": "DXY_Close"}),
    "TOTALSA": load_series_csv("TOTALSA.csv", date_col="observation_date"),
    "HOUST": load_series_csv("HOUST.csv", date_col="observation_date"),
    "TB3MS": load_series_csv("TB3MS.csv", date_col="observation_date"),
    "DGS3MO": load_series_csv("DGS3MO.csv", date_col="observation_date"),
    "T10Y2Y": load_series_csv("T10Y2Y.csv", date_col="observation_date"),
    "USSLIND": load_series_csv("USSLIND.csv", date_col="observation_date"),
    "BAA": load_series_csv("BAA.csv", date_col="observation_date"),
    "AAA": load_series_csv("AAA.csv", date_col="observation_date"),
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
    series_map["core_cpi"],
    series_map["dxy"],
    series_map["TOTALSA"],
    series_map["HOUST"],
    series_map["TB3MS"],
    series_map["DGS3MO"],
    series_map["T10Y2Y"],
    series_map["USSLIND"],
    series_map["BAA"],
    series_map["AAA"],
], how="left")



# DFII10 make dropna
# df = df.dropna(subset=["DFII10"]).copy()

# FEATURE ENGINEERING 

df = add_technical_indicators(df)

# release lag aproximado
df["GDPC1"] = df["GDPC1"].shift(3)
df["UNRATE"] = df["UNRATE"].shift(1)
df["PERMIT"] = df["PERMIT"].shift(1)
df["M2SL"] = df["M2SL"].shift(1)
df["TOTALSA"] = df["TOTALSA"].shift(1)
df["HOUST"] = df["HOUST"].shift(1)
df["CORESTICKM159SFRBATL"] = df["CORESTICKM159SFRBATL"].shift(1)
df["WALCL"] = df["WALCL"].shift(1)


df["balance_yoy"] = df["WALCL"].pct_change(12)
df["sp500_12m"] = df["Close"].pct_change(12)
df["sp500_horizon"] = df["Close"].pct_change(HORIZON)
df["gdp_yoy"] = df["GDPC1"].pct_change(12)
df["gdp_yoy_lag6"] = df["gdp_yoy"].shift(6)
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
df["liquidity_impulse"] = df["m2_yoy"] - df["gdp_yoy"]
df["curve_change_12m"] = df["T10Y3M"].diff(12)
df["value_momentum"] = df["sp500_earnings_yield"] * df[f"roc_{HORIZON}"]
df["high_inflation"] = (df["T10YIE"] > 2.5).astype(int)
df["equity_risk_premium"] = df["sp500_earnings_yield"] - df["DGS10"]
df["NFCI_3m_change"] = df["NFCI"].diff(3)
df["drawdown_12m"] = df["Close"] / df["Close"].rolling(12).max() - 1
df["momentum_12m"] = df["Close"].pct_change(12)
df["real_rate_change_6m"] = df["DFII10"].diff(6)
df["dxy_12m"] = df["DXY_Close"].pct_change(12)
df["dxy_3m_change"] = df["DXY_Close"].pct_change(3)
df["vix_z_score"] = (df["VIX_Close"] - df["VIX_Close"].rolling(12).mean()) / df["VIX_Close"].rolling(12).std()
df["earnings_growth_12m"] = df["sp500_pe_ratio"].diff(12) / df["sp500_pe_ratio"].shift(12)
df["hy_spread_change_3m"] = df["BAMLH0A0HYM2"].diff(3)
df["credit_impulse"] = -df["BAMLH0A0HYM2"].diff(12)
df["real_rate"] = df["DFII10"] - df["CORESTICKM159SFRBATL"]
df["gdp_yoy_ma6"] = df["gdp_yoy"].rolling(6).mean()
df["gdp_yoy_diff6"] = df["gdp_yoy"] - df["gdp_yoy"].shift(6)
df["ret_6m"] = df["Close"].pct_change(6)
df["ret_12m"] = df["Close"].pct_change(12)
df["recession"] = (df["UNRATE"] > df["UNRATE"].rolling(24).mean()).astype(int)
df["liquidity_trend"] = df["WALCL"].pct_change(6) - df["WALCL"].pct_change(12)
df["liquidity_impulse_lag6"] = df["liquidity_impulse"].shift(6)
df["curve_slope"] = df["DGS10"] - df["T10Y3M"]
df["credit_spread"] = df["BAA"] - df["AAA"]
df["momentum_change"] = df["momentum_12m"] - df["momentum_12m"].shift(6)
df["vol_regime"] = df["VIX_Close"] / df["VIX_Close"].rolling(12).mean()
df["credit_stress"] = df["BAMLH0A0HYM2"].diff(6)

h = HORIZON
short = max(3, h // 2)       
mid   = h                     
long  = h * 2                 

min_features = [
    "cape_earnings_yield",
    "equity_risk_premium",
    "curve_slope",
    "DFII10",
    "FEDFUNDS",
    "credit_spread",
    "BAMLH0A0HYM2",
    "CORESTICKM159SFRBATL",
    "T10YIE",
    "gdp_yoy",
    "unemp_change_12m",
    "permit_yoy",
    "balance_yoy",
    "m2_yoy",
    "NFCI"
]

features = [
    "balance_yoy",
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
    "liquidity_impulse",
    "curve_change_12m",
    "high_inflation",
    "CORESTICKM159SFRBATL",
    "equity_risk_premium",
    "NFCI_3m_change",
    "real_rate_change_6m",
    "dxy_12m",
    "vix_z_score",
    "earnings_growth_12m",
    "hy_spread_change_3m",
    "credit_impulse",
    "real_rate",
    "HOUST",
    "TOTALSA",
    "T10Y2Y",
    "curve_slope",
    "USSLIND",
    "credit_spread",
    "vol_regime",
    "credit_stress",    
    "recession",
    "liquidity_trend",
    "gdp_yoy_lag6",


    # "gdp_yoy_ma6",
    # "liquidity_impulse_lag6",
    # "value_momentum",
    # "ret_6m",
    # "momentum_change",
    # "momentum_12m",
    # "gdp_yoy",
    # "gdp_yoy_diff6",
    # "dxy_3m_change",
    # "drawdown_12m",
    # "sp500_12m",
    # "sp500_horizon",
    # "ret_12m",
    # f"ema_{short}_dist",
    # f"ema_{mid}_dist",
    # f"ema_{long}_dist",
    # "rsi_14",
    # f"roc_{HORIZON}",
]


# features = min_features
print(f"Number of features: {len(features)}")








# OBJETIVO
fecha_inicio = "1965-01-31"
fecha_inicio = "1900-01-31"
fecha_objetivo = "2001-05-31"
fecha_objetivo = "2035-01-31"
df = df.loc[fecha_inicio:fecha_objetivo].copy()







# usar solo variables con 60% historia
min_history = 0.6
valid_features = [
    f for f in features
    if df[f].notna().mean() > min_history
]
dropped_features = [f for f in features if f not in valid_features]
# print("Features eliminadas por tener poco historial:")
# for f in dropped_features:
#     print(f)
features = valid_features

   



# TARGET  
min_train_size = 180   
test_size = 12 



# Clasificación: 1 si retorno futuro > umbral, 0 si no
df["future_return"] = df["Close"].shift(-HORIZON) / df["Close"] - 1
df["close_fwd"] = df["Close"].shift(-HORIZON)
df["target"] = (df["Close"].shift(-HORIZON) > df["Close"]).astype(int)
df["target_reg"] = np.log(df["Close"].shift(-HORIZON) / df["Close"])

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["target"])
df = df.dropna(subset=features)


# eliminar overlap
# df = df.iloc[::HORIZON].copy()
# df[features] = df[features].fillna(0)
# df[features] = df[features].ffill()









# EDA
BASE_DIR = BASE / "metrics"
if BASE_DIR.exists():
    shutil.rmtree(BASE_DIR)
BASE_DIR.mkdir(exist_ok=True)



correlation_report(df, features + ["target"])
# La señal lineal es débil.
# Eso es normal en mercados financieros.
# Ninguna variable tiene correlación fuerte (> 0.3).
# Eso es buena señal:
# 👉 No hay leakage obvio.



print(df.head())




# =========================================================
# 1️⃣ Rank Correlation (Spearman) feature vs target
# Detecta relación monotónica (no necesariamente lineal)
# En mercados suele ser más informativo que Pearson
# =========================================================
# from scipy.stats import spearmanr

# print("Spearman rank correlation vs target\n")

# for col in features:
#     rc = spearmanr(df[col], df["target"]).correlation
#     print(f"{col:35s} {rc:.3f}")
# Interpretación:
# > 0.05 consistente ya es interesante en finanzas
# Signo estable > magnitud


# =========================================================
# 2️⃣ Decile / Binning Analysis
# Mide si extremos de la variable predicen retornos distintos
# Es mucho más útil que mirar solo correlación
# =========================================================
# feature_to_test = f"cape_earnings_yield"   # cambia si quieres probar otra
# df["bin"] = pd.qcut(df[feature_to_test], 10, labels=False, duplicates="drop")
# decile_returns = df.groupby("bin")["target"].mean()
# print("\nRetorno medio por decil:")
# print(decile_returns)
# Interpretación:
# Ideal: decil 9 >> decil 0
# Relación creciente casi monotónica = señal robusta



# =========================================================
# 3️⃣ Distribución del Target
# Permite ver asimetría, colas y sesgo estructural
# Clave para interpretar R2 y MSE
# =========================================================
plt.figure(figsize=(8,4))
df["target_reg"].hist(bins=50)
plt.title("Distribución del retorno futuro (target)")
plt.axvline(0, linestyle="--")
plt.savefig(BASE_DIR / "return_dist.png")
# Si está muy concentrado en 0 → modelo difícil
# Si hay colas gordas → cuidado con MSE (dominado por outliers)






















# Variables predictoras
X = df[features]
y = df["target"].astype(int)

base_rate = float(y.mean())
print(f"Target base-rate (P(y=1)) en dataset: {base_rate:.3f}")

# =============================
# Random Search (hiperparámetros)
# =============================
DO_RANDOM_SEARCH = True
TUNE_EACH_FOLD = False  # True = tunear en cada ventana; False = tunear 1 vez y reutilizar
RANDOM_SEARCH_N_ITER = 500
RANDOM_SEARCH_SEED = 42
SCORE_FRAC = 0.5  # fracción del bloque de validación reservada para scoring (ROC-AUC) + umbral

param_dist = {
    "learning_rate": [0.005, 0.01, 0.02, 0.05],
    "max_depth": [3, 4, 5, 6],
    "min_child_weight": [1, 3, 5, 8, 12],
    "gamma": [0, 0.5, 1, 2, 5],
    "subsample": [0.6, 0.7, 0.8, 0.9],
    # "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "reg_lambda": [3, 5, 10, 15, 20],
    # "reg_alpha": [0, 0.1, 0.5, 1],
}

best_params_global: Optional[Dict] = None
best_n_estimators_global: Optional[int] = None

# --- Walk-forward metrics ---
accs = []
aucs = []
loglosses = []
bal_accs = []
ap_scores = []
briers = []

baseline_accs = []
baseline_loglosses = []

fold_thresholds = []

all_proba = []
all_pred = []
all_actuals = []
all_dates = []
all_close = []
all_close_fwd = []

last_model = None

# Walk-forward validation con Purging + Embargo
start = min_train_size
while start < len(df) - test_size:

    purge = HORIZON
    embargo = HORIZON

    train_end = start - purge
    test_end = start + test_size

    train_df = df.iloc[:train_end]
    test_df = df.iloc[start:test_end]

    X_train = train_df[features]
    y_train = train_df["target"].astype(int)

    X_test = test_df[features]
    y_test = test_df["target"].astype(int)

    # balanceo opcional (solo si la clase positiva es rara)
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg / pos) if (pos > 0 and pos < neg) else 1.0

    # ===== Validation interna temporal =====
    val_size = int(len(X_train) * 0.2)
    gap = HORIZON
    tr_end = -(val_size + gap)

    X_tr = X_train.iloc[:tr_end]
    y_tr = y_train.iloc[:tr_end]

    X_val = X_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]

    # Separar bloque de validación: early-stopping (pasado) vs scoring (más reciente)
    if len(X_val) < 3:
        X_es = X_val
        y_es = y_val
        X_score = X_val
        y_score = y_val
    else:
        score_size = max(1, int(len(X_val) * SCORE_FRAC))
        es_size = len(X_val) - score_size
        if es_size < 1:
            es_size = 1
            score_size = len(X_val) - 1

        X_es = X_val.iloc[:es_size]
        y_es = y_val.iloc[:es_size]
        X_score = X_val.iloc[es_size:]
        y_score = y_val.iloc[es_size:]

    # ===== Random Search (opcional) =====
    fixed_params = dict(
        objective="binary:logistic",
        n_estimators=5000,
        random_state=42,
        tree_method="hist",
        eval_metric="auc",
        early_stopping_rounds=200,
    )

    if DO_RANDOM_SEARCH and (TUNE_EACH_FOLD or best_params_global is None):
        # Nota: el tuning usa solo (X_tr -> X_val). No toca el test.
        best_params, best_n_estimators, _best_val_score = tune_xgb_random_search_timeval(
            X_tr,
            y_tr,
            X_es,
            y_es,
            X_score,
            y_score,
            fixed_params=fixed_params,
            param_dist=param_dist,
            n_iter=RANDOM_SEARCH_N_ITER,
            random_state=RANDOM_SEARCH_SEED,
            scale_pos_weight=scale_pos_weight,
        )
        if not TUNE_EACH_FOLD:
            best_params_global = best_params
            best_n_estimators_global = best_n_estimators
    else:
        best_params = best_params_global or {}
        best_n_estimators = best_n_estimators_global

    # Aplicar best_n_estimators si lo tenemos; si no, usar el máximo (early stopping recorta)
    if best_n_estimators is None:
        best_n_estimators = int(fixed_params.get("n_estimators", 5000))

    print(
        f"[Fold] usando best_params={best_params} "
        f"best_n_estimators={int(best_n_estimators)} scale_pos_weight={scale_pos_weight:.3f}"
    )

    fold_fixed_params = dict(fixed_params)
    fold_fixed_params["n_estimators"] = int(best_n_estimators)

    model = XGBClassifier(
        **fold_fixed_params,
        **best_params,
        scale_pos_weight=scale_pos_weight,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_es, y_es)],
        verbose=False,
    )

    last_model = model

    # Probabilidades
    proba = model.predict_proba(X_test)[:, 1]

    # ===== Selección de umbral (en validación temporal) =====
    # Nota: 0.5 rara vez es óptimo si la base-rate != 0.5.
    val_proba = model.predict_proba(X_score)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    best_thr = 0.5
    best_score = -np.inf
    if len(np.unique(y_score)) > 1:
        for thr in thresholds:
            vpred = (val_proba >= thr).astype(int)
            score = balanced_accuracy_score(y_score, vpred)
            if score > best_score:
                best_score = score
                best_thr = float(thr)

    fold_thresholds.append(best_thr)
    pred = (proba >= best_thr).astype(int)

    all_proba.extend(proba.tolist())
    all_pred.extend(pred.tolist())
    all_actuals.extend(y_test.values.tolist())
    all_dates.extend(y_test.index.tolist())
    all_close.extend(test_df["Close"].values.tolist())
    all_close_fwd.extend(test_df["close_fwd"].values.tolist())

    # ===== Baselines (para no engañarse) =====
    # 1) Clasificador tonto: siempre predice la clase mayoritaria del TRAIN
    majority_class = int(y_train.mean() >= 0.5)
    baseline_pred = np.full_like(y_test.values, fill_value=majority_class)
    baseline_accs.append(accuracy_score(y_test, baseline_pred))

    # 2) Probabilidad constante: p = base-rate del TRAIN
    p0 = float(np.clip(y_train.mean(), 1e-6, 1 - 1e-6))
    baseline_loglosses.append(log_loss(y_test, np.full_like(proba, p0), labels=[0, 1]))

    # ===== Métricas del modelo =====
    accs.append(accuracy_score(y_test, pred))
    bal_accs.append(balanced_accuracy_score(y_test, pred))

    if len(np.unique(y_test)) > 1:
        aucs.append(roc_auc_score(y_test, proba))
        ap_scores.append(average_precision_score(y_test, proba))
    else:
        aucs.append(np.nan)  # AUC no definido si solo hay una clase en el tramo
        ap_scores.append(np.nan)

    loglosses.append(log_loss(y_test, proba, labels=[0, 1]))
    briers.append(brier_score_loss(y_test, proba))

    # ===== avanzar con embargo =====
    start = test_end + embargo

print("Walk-forward Accuracy promedio:", np.nanmean(accs))
print("Walk-forward ROC-AUC promedio:", np.nanmean(aucs))
print("Walk-forward PR-AUC (AvgPrecision) promedio:", np.nanmean(ap_scores))
print("Walk-forward Balanced Accuracy promedio:", np.nanmean(bal_accs))
print("Walk-forward LogLoss promedio:", np.nanmean(loglosses))
print("Walk-forward Brier score promedio:", np.nanmean(briers))

print("\nBaselines (comparación justa)")
print("Baseline Accuracy (mayoría en train):", float(np.nanmean(baseline_accs)))
print("Baseline LogLoss (p const = base-rate train):", float(np.nanmean(baseline_loglosses)))
print("Umbral elegido (mediana folds):", float(np.nanmedian(fold_thresholds)))

# Dataset WF para plots
wf_df = pd.DataFrame({
    "date": pd.to_datetime(all_dates),
    "proba_up": all_proba,   # prob. de clase positiva (sube)
    "pred": all_pred,
    "actual": all_actuals,
    "close_t": all_close,
    "close_t_plus_h": all_close_fwd,
})
wf_df = (
    wf_df.sort_values("date")
    .drop_duplicates(subset="date", keep="last")
    .reset_index(drop=True)
)

# Señal suavizada sobre probabilidad
wf_df["signal_raw"] = wf_df["proba_up"]
wf_df["signal"] = wf_df["proba_up"] - 0.5

# ── Gráfico Walk-Forward: probabilidad vs clase real ──
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    wf_df["date"],
    wf_df["actual"],
    label="Clase real (0/1)",
    color="black",
    alpha=0.6,
    drawstyle="steps-post",
)
# ax.plot(
#     wf_df["date"],
#     wf_df["signal_raw"],
#     label="P(sube) modelo",
#     color="purple",
#     alpha=0.25,
# )

# Precio en eje secundario para ver confirmación visual
ax2 = ax.twinx()
ax2.plot(
    wf_df["date"],
    wf_df["close_t"],
    label="Precio (Close t)",
    color="tab:blue",
    alpha=0.5,
)
ax2.plot(
    wf_df["date"],
    wf_df["close_t_plus_h"],
    label=f"Precio (Close t+{HORIZON}m)",
    color="tab:blue",
    alpha=0.35,
    linestyle="--",
)
ax2.set_ylabel("Precio (Close)")
ax2.set_yscale("log")

ax.set_title(
    f"Walk-Forward — Clasificación {HORIZON}m (P(sube)) + Precio (t y t+{HORIZON}m)"
)
ax.set_ylabel("Probabilidad / Clase")
ax.set_ylim(-0.05, 1.05)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
fig.autofmt_xdate()

# Leyenda combinada
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(BASE_DIR / "walk_forward_classification.png")


# ── Ranking power por deciles de probabilidad ──
dec_df = wf_df.copy()
dec_df["decile"] = pd.qcut(dec_df["proba_up"], 10, labels=False, duplicates="drop")
deciles = dec_df.groupby("decile")["actual"].mean()

fig, ax = plt.subplots(figsize=(8, 4))
deciles.plot(kind="bar", ax=ax)
ax.set_title("Tasa de acierto (clase=1) por decil de P(sube)")
ax.set_xlabel("Decil P(sube) (bajo → alto)")
ax.set_ylabel("P(real=1)")
plt.tight_layout()
plt.savefig(BASE_DIR / "decile_plot_classification.png")

# ── Confusion matrix global (sobre todo el WF) ──
cm = confusion_matrix(wf_df["actual"], wf_df["pred"], labels=[0, 1])
print("\nConfusion matrix global (WF):\n", cm)
print("\nClassification report global (WF):\n")
print(classification_report(wf_df["actual"], wf_df["pred"], digits=3))


# ==============================
# ROI: Buy&Hold DCA vs Señal (2x cuando pred=1)
# ==============================
monthly_amount = 1.0
signal_multiplier = 2.0

wf_signal = wf_df[["date", "pred"]].copy()
wf_signal["date"] = pd.to_datetime(wf_signal["date"])
wf_signal = wf_signal.set_index("date").sort_index()

eval_start = pd.to_datetime(wf_df["date"].min())
eval_end = pd.to_datetime(wf_df["date"].max())

prices_eval = df.loc[eval_start:eval_end, "Close"].copy()
pred_aligned = wf_signal["pred"].reindex(prices_eval.index)

# Comparación justa: usar SOLO los meses en los que hay predicción walk-forward.
has_pred = pred_aligned.notna()
prices_eval = prices_eval.loc[has_pred]
pred_aligned = pred_aligned.loc[has_pred]

contrib_bh = pd.Series(monthly_amount, index=prices_eval.index)
contrib_signal = pd.Series(0.0, index=prices_eval.index)
contrib_signal[pred_aligned.fillna(0).astype(int) == 1] = monthly_amount * signal_multiplier

bh_curve = simulate_monthly_dca_roi(prices_eval, contrib_bh)
sig_curve = simulate_monthly_dca_roi(prices_eval, contrib_signal)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(bh_curve.index, bh_curve["roi_pct"], label=f"Buy&Hold DCA (x={monthly_amount:g}/mes)", color="tab:blue")
ax.plot(
    sig_curve.index,
    sig_curve["roi_pct"],
    label=f"Señal (comprar {signal_multiplier:g}x si pred=1)",
    color="purple",
)

ax.axhline(0, linestyle="--", color="grey", alpha=0.6)
ax.set_title(f"ROI acumulado (%) — DCA mensual vs Señal (Walk-Forward, horizonte {HORIZON}m)")
ax.set_ylabel("ROI (%)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(BASE_DIR / "roi_strategies_walk_forward.png")

print("\nROI final Buy&Hold DCA (%):", float(bh_curve["roi_pct"].dropna().iloc[-1]))
print("ROI final Señal 2x si pred=1 (%):", float(sig_curve["roi_pct"].dropna().iloc[-1]))
print("Total invertido Buy&Hold:", float(bh_curve["invested"].dropna().iloc[-1]))
print("Total invertido Señal:", float(sig_curve["invested"].dropna().iloc[-1]))










exit(0)




# ===== Modelo final entrenado en todo el dataset =====
final_model = XGBClassifier(
    objective="binary:logistic",
    n_estimators=5000,
    learning_rate=0.01,
    max_depth=4,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=3,
    random_state=42,
    tree_method="hist",
    eval_metric="logloss",
)

final_model.fit(X, y)

last_X = X.iloc[[-1]]
final_proba = float(final_model.predict_proba(last_X)[:, 1][0])
final_pred = int(final_proba >= 0.5)

print("Última fecha:", X.index[-1])
print(f"P(sube) en {HORIZON} meses:", final_proba)
print("Predicción clase (>=0.5):", final_pred)

# ==============================
# SHAP (clasificación binaria)
# ==============================
explainer = shap.TreeExplainer(final_model)

shap_values = explainer.shap_values(X)
# En binario, a veces viene como lista [clase0, clase1]; nos quedamos con clase 1
if isinstance(shap_values, list) and len(shap_values) == 2:
    shap_values_pos = shap_values[1]
    expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
else:
    shap_values_pos = shap_values
    expected_value = explainer.expected_value

plt.figure()
shap.summary_plot(shap_values_pos, X, show=False)
plt.tight_layout()
plt.savefig(BASE_DIR / "shap_summary_cls.png")

plt.figure()
shap.summary_plot(shap_values_pos, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(BASE_DIR / "shap_importance_bar_cls.png")

# for feature in X.columns:
#     plt.figure()
#     shap.dependence_plot(feature, shap_values_pos, X, show=False)
#     fname = f"shap_dependence_cls_{feature}.png"
#     plt.tight_layout()
#     plt.savefig(BASE_DIR / fname)

# SHAP para la última predicción (clase positiva)
# shap_values_last = explainer.shap_values(last_X)
# if isinstance(shap_values_last, list) and len(shap_values_last) == 2:
#     shap_values_last_pos = shap_values_last[1][0]
# else:
#     shap_values_last_pos = shap_values_last[0]

# plt.figure()
# shap.plots.waterfall(
#     shap.Explanation(
#         values=shap_values_last_pos,
#         base_values=expected_value,
#         data=last_X.iloc[0],
#         feature_names=X.columns
#     ),
#     show=False
# )
# plt.tight_layout()
# plt.savefig(BASE_DIR / "shap_last_prediction_cls.png")