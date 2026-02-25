import pandas as pd
# import yfinance as yf

# sp500 = yf.download("^GSPC", progress=False, period="max")
# sp500.reset_index(inplace=True)
# sp500.to_csv("sp500.csv")
# print( pd.read_csv("sp500.csv"))

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

# df = df.dropna()
# print(df.head())

# FEATURE ENGINEERING 

# Retorno 12 meses SP500
# Variaciones macro
df["sp500_12m"] = df["Close"].pct_change(12)
df["gdp_yoy"] = df["GDPC1"].pct_change(12)
# df["balance_yoy"] = df["WALCL"].pct_change(12)
df["unemp_change_12m"] = df["UNRATE"].diff(12)


features = [
    "sp500_12m",
    "gdp_yoy",
    # "balance_yoy",
    "unemp_change_12m",
]


# Eliminar NaNs creados por los cambios
# df = df.dropna()
df = df.dropna(subset=features)

# TARGET 10 meses 
horizon = 10

min_train_size = 180   
test_size = 48        



df["future_return"] = df["Close"].shift(-horizon) / df["Close"] - 1
df["target"] = (df["future_return"] > 0).astype(int)
# df["target"] = (df["future_return"] > 1).astype(int)
df = df.dropna(subset=["future_return", "target"])

print(df.head())
print(df.tail())



from sklearn.metrics import roc_auc_score
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


fecha_objetivo = "2001-05-31"
# fecha_objetivo = "2035-01-31"
df = df.loc[:fecha_objetivo].copy()


# Variables predictoras
X = df.drop(["future_return", "target"], axis=1)
y = df["target"]

aucs = []
accuracies = []

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