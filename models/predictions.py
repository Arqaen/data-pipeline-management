from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil
import matplotlib.dates as mdates
import shap 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import precision_recall_curve, roc_curve
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


def top_k_signals(
    proba: np.ndarray,
    *,
    k_frac: float,
    min_k: int = 1,
) -> np.ndarray:
    """Devuelve señales binarias seleccionando exactamente las K probas más altas.

    - Controla nº de trades (K) por bloque.
    - Robustez: evita dependencia fuerte de calibración absoluta (umbral).

    Args:
        proba: array-like de probabilidades.
        k_frac: fracción objetivo a seleccionar (0..1).
        min_k: mínimo absoluto de selecciones (capado por n).

    Returns:
        np.ndarray de shape (n,) con 0/1.
    """

    proba = np.asarray(proba, dtype=float)
    if not np.all(np.isfinite(proba)):
        proba = np.nan_to_num(proba, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    n = int(proba.size)
    if n == 0:
        return np.asarray([], dtype=int)

    k_frac = float(np.clip(k_frac, 0.0, 1.0))
    min_k = int(max(0, min_k))

    k = int(n * k_frac)
    k = max(k, min_k)
    k = int(np.clip(k, 0, n))
    if k == 0:
        return np.zeros(n, dtype=int)

    # Orden descendente; mergesort es estable (útil si hay empates).
    idx_sorted = np.argsort(-proba, kind="mergesort")
    top_idx = idx_sorted[:k]

    signals = np.zeros(n, dtype=int)
    signals[top_idx] = 1
    return signals


def top_k_cutoff(proba: np.ndarray, *, k: int) -> float:
    """Umbral equivalente (k-ésimo mayor) para reportes/plots.

    Nota: seleccionar por top-k NO es lo mismo que aplicar un umbral fijo,
    pero este cutoff ayuda a visualizar el punto de corte del bloque.
    """

    proba = np.asarray(proba, dtype=float)
    if not np.all(np.isfinite(proba)):
        proba = np.nan_to_num(proba, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    n = int(proba.size)
    if n == 0:
        return 0.5

    k = int(np.clip(int(k), 1, n))
    # kth mayor => (n-k)th menor
    cutoff = float(np.partition(proba, n - k)[n - k])
    return cutoff


def choose_threshold(
    y_score: pd.Series,
    val_proba: np.ndarray,
    *,
    method: str,
    base_rate_train: float,
) -> Tuple[float, float]:
    """Elige un umbral de clasificación usando SOLO el bloque de scoring.

    Métodos:
    - "balanced_accuracy": maximiza balanced accuracy (tu comportamiento actual)
    - "fixed_0.5": umbral 0.5 (neutral si las probabilidades están calibradas)
    - "prevalence_match": fuerza %pred=1 ~ base_rate del train (neutral a exposición)
    - "youden_j": maximiza TPR - FPR (ROC, simétrico en errores)
    """

    val_proba = np.asarray(val_proba, dtype=float)
    val_proba = np.clip(val_proba, 1e-9, 1 - 1e-9)
    y_arr = np.asarray(y_score, dtype=int)

    if method == "fixed_0.5":
        return 0.5, float("nan")

    if method == "prevalence_match":
        br = float(np.clip(base_rate_train, 1e-6, 1 - 1e-6))
        # Queremos pred=1 en ~br de los casos -> umbral = cuantil (1-br)
        thr = float(np.quantile(val_proba, 1.0 - br))
        return thr, br

    # Si solo hay una clase en el bloque, no hay ROC/BA informativos.
    if len(np.unique(y_arr)) < 2:
        return 0.5, float("nan")

    if method == "youden_j":
        fpr, tpr, thr = roc_curve(y_arr, val_proba)
        thr = np.asarray(thr, dtype=float)
        mask = np.isfinite(thr)
        if not np.any(mask):
            return 0.5, float("nan")
        j = (tpr - fpr)[mask]
        i = int(np.nanargmax(j))
        thr_best = float(thr[mask][i])
        return thr_best, float(j[i])

    # default: balanced_accuracy
    thresholds = np.linspace(0.05, 0.95, 91)
    best_thr = 0.5
    best_score = -np.inf
    for thr in thresholds:
        vpred = (val_proba >= float(thr)).astype(int)
        score = balanced_accuracy_score(y_arr, vpred)
        if score > best_score:
            best_score = float(score)
            best_thr = float(thr)
    return best_thr, float(best_score)


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
) -> Tuple[Dict, float]:
    """Random search con validación temporal (X_tr -> early-stopping -> scoring).

    - Sin RandomizedSearchCV (que no encaja bien con early_stopping + embargo).
    - Score: minimiza LogLoss en el bloque de scoring.

    Devuelve: (best_params, best_logloss)

    Nota: `n_estimators` NO se optimiza aquí. Se asume fijo en `fixed_params`
    (early-stopping puede cortar antes, pero no devolvemos/ajustamos un
    `best_n_estimators` explícito).
    """
    rng = np.random.RandomState(random_state)

    best_params: Dict = {}
    best_logloss = np.inf
    fixed_n_estimators = int(fixed_params.get("n_estimators", 5000))

    # Si no hay suficientes clases, log_loss sigue siendo válido con labels=[0,1]
    # pero el tuning puede ser poco informativo. Aun así, lo dejamos correr.
    for _ in range(int(n_iter)):
        params = _sample_param_combo(param_dist, rng)
        # Importante: no podemos pasar claves duplicadas vía múltiples **kwargs.
        # Mezclamos para que params (random search) sobrescriba a fixed_params.
        model_params = dict(fixed_params)
        model_params.update(params)
        model_params["scale_pos_weight"] = scale_pos_weight
        model = XGBClassifier(**model_params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_es, y_es)],
            verbose=False,
        )
        score_proba = model.predict_proba(X_score)[:, 1]

        # Optimizamos LogLoss directamente (más estable que AUC con base-rate alta).
        # Clampeamos para evitar inf por probabilidades 0/1.
        score_proba = np.clip(score_proba, 1e-6, 1 - 1e-6)
        ll = float(log_loss(y_score, score_proba, labels=[0, 1]))

        if ll < best_logloss:
            best_logloss = ll
            best_params = params

    print(
        f"[RandomSearch] best logloss={best_logloss:.5f} "
        f"n_estimators(fijo)={fixed_n_estimators} params={best_params}"
    )
    return best_params, float(best_logloss)

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
    return corr


def _save_table_figure(
    table_df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    footer: Optional[str] = None,
    float_fmt: str = "{:.4f}",
) -> None:
    """Guarda un DataFrame como tabla (matplotlib) en un PNG."""
    if table_df is None or table_df.empty:
        return

    disp = table_df.copy()
    for col in disp.columns:
        if pd.api.types.is_numeric_dtype(disp[col]):
            disp[col] = disp[col].map(lambda x: "" if pd.isna(x) else float_fmt.format(float(x)))
        else:
            disp[col] = disp[col].astype(str)

    n_rows, n_cols = disp.shape
    fig_w = max(7.5, 1.25 * (n_cols + 1))
    fig_h = max(2.2, 0.45 * (n_rows + 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=disp.values,
        colLabels=[str(c) for c in disp.columns],
        rowLabels=[str(i) for i in disp.index],
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.3)

    fig.suptitle(title, y=0.97)
    if footer:
        fig.text(0.01, 0.02, footer, ha="left", va="bottom", fontsize=9)

    fig.tight_layout(rect=[0, 0.04 if footer else 0, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_correlation_heatmap(
    corr: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    target_col: str = "target",
    max_vars: int = 25,
    top_k: int = 20,
) -> None:
    """Heatmap de correlación; si hay muchas variables, usa top-k vs target."""
    if corr is None or corr.empty:
        return

    corr_plot = corr.copy()
    if corr_plot.shape[0] > max_vars:
        if target_col in corr_plot.columns:
            ranked = (
                corr_plot[target_col]
                .drop(labels=[target_col], errors="ignore")
                .abs()
                .sort_values(ascending=False)
            )
            keep = list(ranked.head(int(top_k)).index)
            keep = keep + [target_col]
            corr_plot = corr_plot.loc[keep, keep]
            title = f"{title} (top {top_k} vs {target_col})"
        else:
            corr_plot = corr_plot.iloc[:max_vars, :max_vars]
            title = f"{title} (primeras {max_vars})"

    labels = [str(c) for c in corr_plot.columns]
    fig_w = max(8.0, 0.35 * len(labels))
    fig_h = max(7.0, 0.35 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    data = corr_plot.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="coolwarm", vmin=-1.0, vmax=1.0)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlación")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def compute_spearman_rank_corr(
    df: pd.DataFrame,
    features: List[str],
    *,
    target_col: str = "target",
) -> pd.Series:
    """Spearman feature vs target con dropna por pares."""
    values: Dict[str, float] = {}
    for col in features:
        if col not in df.columns or target_col not in df.columns:
            continue
        pair = df[[col, target_col]].dropna()
        if len(pair) < 3:
            values[col] = np.nan
            continue
        values[col] = float(spearmanr(pair[col], pair[target_col]).correlation)
    return pd.Series(values, dtype=float)


def plot_spearman_rank_corr_bar(
    spearman_corr: pd.Series,
    *,
    out_path: Path,
    title: str,
    top_n: Optional[int] = None,
) -> None:
    if spearman_corr is None or spearman_corr.empty:
        return

    s = spearman_corr.dropna().copy()
    if s.empty:
        return

    # Orden por |corr| para lectura rápida.
    s = s.reindex(s.abs().sort_values(ascending=False).index)
    if top_n is not None:
        s = s.head(int(top_n))

    s = s.sort_values()  # para que el barh quede de menor a mayor

    fig_h = max(6.0, 0.22 * len(s) + 1.5)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    ax.barh(s.index.astype(str), s.values, color="tab:blue", alpha=0.85)
    ax.axvline(0, color="grey", lw=1, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Spearman ρ")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def _has_cols(df: pd.DataFrame, cols: List[str]) -> bool:
    return all(c in df.columns for c in cols)

def plot_classification_timeline(
    plot_df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    threshold: float = 0.5,
    year_locator: int = 2,
    date_col: str = "date",
    proba_col: str = "proba_up",
    pred_col: str = "pred",
    actual_col: str = "actual",
    price_col: str = "close_t",
    price_fwd_col: str = "close_t_plus_h",
) -> None:
    from matplotlib.patches import Patch

    dfp = plot_df.copy()
    dfp[date_col] = pd.to_datetime(dfp[date_col])
    dfp = dfp.sort_values(date_col).reset_index(drop=True)

    if dfp.empty:
        return

    ends = dfp[date_col].shift(-1)
    if len(dfp) > 1:
        step = dfp[date_col].diff().dropna().median()
        if pd.isna(step) or step <= pd.Timedelta(0):
            step = pd.Timedelta(days=31)
    else:
        step = pd.Timedelta(days=31)
    ends.iloc[-1] = dfp.loc[len(dfp) - 1, date_col] + step

    y_true = dfp[actual_col].astype(int).to_numpy()
    y_pred = dfp[pred_col].astype(int).to_numpy()

    # Fondo binario: verde si acierta (TP+TN), rojo si falla (FP+FN)
    is_correct = (y_true == y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(14, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.4]},
    )

    for i in range(len(dfp)):
        col = "tab:green" if bool(is_correct[i]) else "tab:red"
        ax0.axvspan(dfp.loc[i, date_col], ends.iloc[i], color=col, alpha=0.10, lw=0)

    # ax0.plot(
    #     dfp[date_col],
    #     dfp[actual_col],
    #     label="Clase real (0/1)",
    #     color="black",
    #     alpha=0.70,
    #     lw=1.6,
    #     drawstyle="steps-post",
    #     zorder=3,
    # )

    # Predicción como puntos (evita estética de función escalón)
    ax0.scatter(
        dfp[date_col],
        dfp[pred_col],
        label="Predicción de clase (0/1)",
        color="tab:orange",
        alpha=0.85,
        s=18,
        edgecolor="none",
        zorder=4,
    )
    ax0.set_ylim(-0.05, 1.05)
    ax0.set_ylabel("Clase")

    ax0b = ax0.twinx()
    if price_col in dfp.columns:
        ax0b.plot(
            dfp[date_col],
            dfp[price_col],
            label="Precio (Close t)",
            color="tab:blue",
            alpha=0.35,
            lw=1.2,
        )
    # if price_fwd_col in dfp.columns:
    #     ax0b.plot(
    #         dfp[date_col],
    #         dfp[price_fwd_col],
    #         label=f"Precio (Close t+{HORIZON}m)",
    #         color="tab:blue",
    #         alpha=0.20,
    #         lw=1.0,
    #         linestyle="--",
    #     )
    ax0b.set_ylabel("Precio (Close)")
    ax0b.set_yscale("log")

    # Leyenda: solo 2 categorías para el fondo (sin desglose TN/TP/FP/FN)
    outcome_handles = [
        Patch(facecolor="tab:green", edgecolor="none", alpha=0.25, label="Verde (TP+TN)"),
        Patch(facecolor="tab:red", edgecolor="none", alpha=0.25, label="Rojo (FP+FN)"),
    ]

    lines0, labels0 = ax0.get_legend_handles_labels()
    lines0b, labels0b = ax0b.get_legend_handles_labels()
    ax0.legend(
        lines0 + lines0b + outcome_handles,
        labels0 + labels0b + [h.get_label() for h in outcome_handles],
        loc="upper left",
        ncol=2,
        fontsize=9,
    )

    if proba_col in dfp.columns:
        ax1.plot(
            dfp[date_col],
            dfp[proba_col].astype(float),
            color="purple",
            alpha=0.50,
            lw=1.3,
            label="P(sube)",
        )

        ax1.axhline(
            float(threshold),
            color="grey",
            lw=1.0,
            ls="--",
            alpha=0.85,
            label=f"Umbral={float(threshold):.2f}",
        )
        ax1.set_ylim(0.0, 1.0)
        ax1.set_ylabel("Prob.")
        ax1.legend(loc="upper left", ncol=2, fontsize=9)

    ax1.set_xlabel("Fecha")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator(year_locator))

    ax0.grid(True, alpha=0.25)
    ax1.grid(True, alpha=0.25)

    fig.suptitle(f"{title}\nAcc={acc:.3f} | BalAcc={bal_acc:.3f}", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix_wf(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    out_path: Path,
    title: str,
    class_labels: Tuple[str, str] = ("0", "1"),
) -> None:
    cm = confusion_matrix(y_true.astype(int), y_pred.astype(int), labels=[0, 1])
    if cm.shape != (2, 2):
        raise ValueError(f"Confusion matrix inesperada: shape={cm.shape}")

    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    outcome = np.array([["TN", "FP"], ["FN", "TP"]], dtype=object)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)

    ax.set_title(title)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_xticks([0, 1], labels=list(class_labels))
    ax.set_yticks([0, 1], labels=list(class_labels))

    for i in range(2):
        for j in range(2):
            count = int(cm[i, j])
            pct = float(cm_norm[i, j]) * 100.0
            ax.text(
                j,
                i,
                f"{outcome[i, j]}\n{count:d}\n{pct:.1f}%",
                ha="center",
                va="center",
                fontsize=11,
                color="black" if cm_norm[i, j] < 0.65 else "white",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("% dentro de la clase real (normalizado por fila)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_calibration_curve_wf(
    wf_df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    n_bins: int = 10,
    actual_col: str = "actual",
    proba_col: str = "proba_up",
) -> None:
    dfp = wf_df[[actual_col, proba_col]].copy()
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna()
    if dfp.empty:
        return

    y_true = dfp[actual_col].astype(int).to_numpy()
    y_proba = dfp[proba_col].astype(float).to_numpy()
    y_proba = np.clip(y_proba, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))

    if len(np.unique(y_true)) < 2:
        ax.text(
            0.5,
            0.5,
            "Calibration curve no definida\n(solo 1 clase en WF)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        frac_pos, mean_pred = calibration_curve(
            y_true,
            y_proba,
            n_bins=int(n_bins),
            strategy="quantile",
        )
        ax.plot(mean_pred, frac_pos, marker="o", lw=1.8, label="Modelo")
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1.2, label="Ideal")

    ax.set_title(title)
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Frecuencia real de clase=1")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_proba_hist_by_class(
    wf_df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    actual_col: str = "actual",
    proba_col: str = "proba_up",
    bins: int = 25,
    kde: bool = True,
) -> None:
    from scipy.stats import gaussian_kde

    dfp = wf_df[[actual_col, proba_col]].copy()
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna()
    if dfp.empty:
        return

    y_true = dfp[actual_col].astype(int)
    y_proba = np.clip(dfp[proba_col].astype(float).to_numpy(), 0.0, 1.0)

    p0 = y_proba[(y_true == 0).to_numpy()]
    p1 = y_proba[(y_true == 1).to_numpy()]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(p0, bins=bins, density=True, alpha=0.45, color="tab:blue", label="Real=0")
    ax.hist(p1, bins=bins, density=True, alpha=0.45, color="tab:orange", label="Real=1")

    if kde:
        grid = np.linspace(0.0, 1.0, 300)
        if len(p0) > 3 and np.std(p0) > 1e-12:
            ax.plot(grid, gaussian_kde(p0)(grid), color="tab:blue", lw=1.6)
        if len(p1) > 3 and np.std(p1) > 1e-12:
            ax.plot(grid, gaussian_kde(p1)(grid), color="tab:orange", lw=1.6)

    ax.set_title(title)
    ax.set_xlabel("P(sube)")
    ax.set_ylabel("Densidad")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_roc_pr_wf(
    wf_df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    threshold: float,
    actual_col: str = "actual",
    proba_col: str = "proba_up",
) -> None:
    dfp = wf_df[[actual_col, proba_col]].copy()
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna()
    if dfp.empty:
        return

    y_true = dfp[actual_col].astype(int).to_numpy()
    y_proba = np.clip(dfp[proba_col].astype(float).to_numpy(), 0.0, 1.0)

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    fig.suptitle(title, y=0.98)

    if len(np.unique(y_true)) < 2:
        ax_roc.text(0.5, 0.5, "ROC no definida\n(solo 1 clase)", ha="center", va="center", transform=ax_roc.transAxes)
        ax_pr.text(0.5, 0.5, "PR no definida\n(solo 1 clase)", ha="center", va="center", transform=ax_pr.transAxes)
    else:
        fpr, tpr, _thr = roc_curve(y_true, y_proba)
        roc_auc = float(roc_auc_score(y_true, y_proba))
        ax_roc.plot(fpr, tpr, lw=1.8, label=f"ROC-AUC={roc_auc:.3f}")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1.0)

        # Punto operativo
        y_pred_thr = (y_proba >= float(threshold)).astype(int)
        cm = confusion_matrix(y_true, y_pred_thr, labels=[0, 1])
        tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
        fpr_thr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        tpr_thr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        ax_roc.scatter([fpr_thr], [tpr_thr], s=45, color="black", zorder=5, label=f"thr={float(threshold):.2f}")

        ax_roc.set_title("ROC")
        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        ax_roc.set_xlim(0.0, 1.0)
        ax_roc.set_ylim(0.0, 1.0)
        ax_roc.grid(True, alpha=0.25)
        ax_roc.legend(loc="lower right")

        precision, recall, _thr_pr = precision_recall_curve(y_true, y_proba)
        ap = float(average_precision_score(y_true, y_proba))
        base_rate = float(np.mean(y_true))
        ax_pr.plot(recall, precision, lw=1.8, label=f"AP={ap:.3f}")
        ax_pr.axhline(base_rate, linestyle="--", color="grey", lw=1.0, label=f"Base-rate={base_rate:.3f}")

        precision_thr = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall_thr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        ax_pr.scatter([recall_thr], [precision_thr], s=45, color="black", zorder=5, label=f"thr={float(threshold):.2f}")

        ax_pr.set_title("Precision-Recall")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_xlim(0.0, 1.0)
        ax_pr.set_ylim(0.0, 1.05)
        ax_pr.grid(True, alpha=0.25)
        ax_pr.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_metrics_by_proba_bin(
    wf_df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    threshold: float,
    n_bins: int = 10,
    actual_col: str = "actual",
    proba_col: str = "proba_up",
) -> None:
    dfp = wf_df[[actual_col, proba_col]].copy()
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna()
    if dfp.empty:
        return

    dfp["pred_thr"] = (dfp[proba_col].astype(float) >= float(threshold)).astype(int)

    # Deciles por cuantiles (si hay duplicados, puede haber < n_bins grupos)
    dfp["bin"] = pd.qcut(
        dfp[proba_col].astype(float),
        int(n_bins),
        labels=False,
        duplicates="drop",
    )

    rows = []
    for b, g in dfp.groupby("bin"):
        y_true = g[actual_col].astype(int).to_numpy()
        y_pred = g["pred_thr"].astype(int).to_numpy()
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
        n = int(len(g))

        acc = (tp + tn) / n if n > 0 else np.nan
        prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan

        rows.append({
            "bin": int(b),
            "n": n,
            "accuracy": float(acc),
            "precision": float(prec) if not pd.isna(prec) else np.nan,
            "recall": float(rec) if not pd.isna(rec) else np.nan,
        })

    mdf = pd.DataFrame(rows).sort_values("bin")

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(mdf["bin"], mdf["accuracy"], marker="o", lw=1.8, label="Accuracy")
    ax.plot(mdf["bin"], mdf["precision"], marker="o", lw=1.8, label="Precision")
    ax.plot(mdf["bin"], mdf["recall"], marker="o", lw=1.8, label="Recall")

    ax.set_title(title)
    ax.set_xlabel("Decil de P(sube) (bajo → alto)")
    ax.set_ylabel("Métrica")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_cumulative_gains_wf(
    wf_df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    actual_col: str = "actual",
    proba_col: str = "proba_up",
) -> None:
    dfp = wf_df[[actual_col, proba_col]].copy()
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna()
    if dfp.empty:
        return

    dfp = dfp.sort_values(proba_col, ascending=False).reset_index(drop=True)
    y_true = dfp[actual_col].astype(int).to_numpy()
    total_pos = int(np.sum(y_true))
    n = int(len(dfp))

    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    if total_pos == 0:
        ax.text(
            0.5,
            0.5,
            "Cumulative gains no definido\n(no hay positivos)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        cum_pos = np.cumsum(y_true)
        x = np.arange(1, n + 1) / n  # fracción de muestras seleccionadas
        gain = cum_pos / total_pos   # fracción de positivos capturados

        ax.plot(x, gain, lw=2.0, label="Modelo")
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1.2, label="Random")

        # Lift en el top 10% (si existe)
        k = max(1, int(round(0.10 * n)))
        top_pos = int(np.sum(y_true[:k]))
        lift10 = (top_pos / k) / (total_pos / n) if total_pos > 0 else np.nan
        ax.scatter([k / n], [top_pos / total_pos], color="black", s=40, zorder=5)
        ax.text(
            k / n,
            top_pos / total_pos,
            f"  top10% lift={lift10:.2f}",
            va="center",
            ha="left",
            fontsize=10,
        )

    ax.set_title(title)
    ax.set_xlabel("Fracción seleccionada (ordenado por P(sube) desc)")
    ax.set_ylabel("Fracción de positivos capturados")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_rolling_accuracy_wf(
    wf_df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    window: int = 36,
    date_col: str = "date",
    actual_col: str = "actual",
    pred_col: str = "pred",
) -> None:
    dfp = wf_df[[date_col, actual_col, pred_col]].copy()
    dfp[date_col] = pd.to_datetime(dfp[date_col])
    dfp = dfp.sort_values(date_col)
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna()
    if dfp.empty:
        return

    correct = (dfp[actual_col].astype(int) == dfp[pred_col].astype(int)).astype(float)
    roll = correct.rolling(int(window), min_periods=max(5, int(window // 3))).mean()

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(dfp[date_col], roll, color="tab:blue", lw=1.8, label=f"Rolling acc ({int(window)}m)")
    ax.axhline(0.5, linestyle="--", color="grey", lw=1.0, alpha=0.8, label="0.50")
    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_regime_performance_wf(
    wf_df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    regime_series: pd.Series,
    date_col: str = "date",
    actual_col: str = "actual",
    pred_col: str = "pred",
) -> pd.DataFrame:
    dfp = wf_df[[date_col, actual_col, pred_col]].copy()
    dfp[date_col] = pd.to_datetime(dfp[date_col])
    dfp = dfp.sort_values(date_col)

    reg = regime_series.copy()
    if not isinstance(reg.index, pd.DatetimeIndex):
        reg.index = pd.to_datetime(reg.index)
    reg = reg.sort_index()

    dfp = dfp.merge(reg.rename("regime"), left_on=date_col, right_index=True, how="left")
    dfp = dfp.dropna(subset=["regime"])
    if dfp.empty:
        return pd.DataFrame()

    rows = []
    for regime_name, g in dfp.groupby("regime"):
        y_true = g[actual_col].astype(int).to_numpy()
        y_pred = g[pred_col].astype(int).to_numpy()
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
        n = int(len(g))
        acc = (tp + tn) / n if n > 0 else np.nan
        prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        base_rate = float(np.mean(y_true)) if n > 0 else np.nan
        rows.append(
            {
                "regime": str(regime_name),
                "n": n,
                "base_rate": base_rate,
                "accuracy": float(acc),
                "precision": float(prec) if not pd.isna(prec) else np.nan,
                "recall": float(rec) if not pd.isna(rec) else np.nan,
            }
        )

    rdf = pd.DataFrame(rows).sort_values("regime")
    if rdf.empty:
        return rdf

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(rdf))
    w = 0.25
    ax.bar(x - w, rdf["accuracy"], width=w, label="Accuracy")
    ax.bar(x, rdf["precision"], width=w, label="Precision")
    ax.bar(x + w, rdf["recall"], width=w, label="Recall")
    ax.set_xticks(x, labels=rdf["regime"].tolist())
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.set_ylabel("Métrica")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="lower right", ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return rdf


def plot_equity_curve_directional_wf(
    wf_df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    date_col: str = "date",
    pred_col: str = "pred",
    close_col: str = "close_t",
    close_fwd_col: str = "close_t_plus_h",
) -> None:
    dfp = wf_df[[date_col, pred_col, close_col, close_fwd_col]].copy()
    dfp[date_col] = pd.to_datetime(dfp[date_col])
    dfp = dfp.sort_values(date_col)
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna()
    if dfp.empty:
        return

    fwd_ret = dfp[close_fwd_col].astype(float).to_numpy() / dfp[close_col].astype(float).to_numpy() - 1.0
    signal = dfp[pred_col].astype(int).to_numpy()
    strat_ret = np.where(signal == 1, fwd_ret, 0.0)

    equity_strat = np.cumprod(1.0 + np.nan_to_num(strat_ret, nan=0.0))
    equity_bh = np.cumprod(1.0 + np.nan_to_num(fwd_ret, nan=0.0))

    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    ax.plot(dfp[date_col], equity_bh, lw=1.8, color="tab:blue", alpha=0.75, label="Buy&Hold (horizon)")
    ax.plot(dfp[date_col], equity_strat, lw=2.0, color="purple", label="Estrategia (pred=1)")
    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Equity (multiplicador)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


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



corr = correlation_report(df, features + ["target"])
plot_correlation_heatmap(
    corr,
    out_path=BASE_DIR / "correlation_heatmap.png",
    title="Matriz de correlación (Pearson)",
    target_col="target",
    max_vars=25,
    top_k=20,
)
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
spearman_corr = compute_spearman_rank_corr(df, features, target_col="target")
plot_spearman_rank_corr_bar(
    spearman_corr,
    out_path=BASE_DIR / "spearman_rank_corr.png",
    title="Spearman rank correlation vs target",
)
# Interpretación:
# > 0.05 consistente ya es interesante en finanzas
# Signo estable > magnitud


# =========================================================
# 2️⃣ Decile / Binning Analysis
# Mide si extremos de la variable predicen retornos distintos
# Es mucho más útil que mirar solo correlación
# =========================================================
feature_to_test = f"cape_earnings_yield"   # cambia si quieres probar otra
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
DO_RANDOM_SEARCH = False
TUNE_EACH_FOLD = False  # True = tunear en cada ventana; False = tunear 1 vez y reutilizar
RANDOM_SEARCH_N_ITER = 300
RANDOM_SEARCH_SEED = 42
SCORE_FRAC = 0.5  # fracción del bloque de validación reservada para scoring (ROC-AUC) + umbral

# Umbral "neutral" por defecto: mantiene tasa de señales ~ base-rate del train.
# Alternativas: "fixed_0.5", "youden_j", "balanced_accuracy", "top_k".
THRESHOLD_METHOD = "top_k"

# TOP-K: en vez de umbral fijo, seleccionar las K mejores oportunidades.
# Para evitar overfitting, no se "tunea" k por fold.
# - Si TOP_K_FRAC=None, se usa la prevalencia del train (base-rate) como k_frac.
# - TOP_K_MIN_K garantiza un mínimo de trades por bloque.
TOP_K_FRAC: Optional[float] = None
TOP_K_MIN_K = 1

param_dist = {
    "learning_rate": [0.03, 0.05, 0.07],
    "max_depth": [4, 5],
    "min_child_weight": [5, 6],
    "gamma": [0.5, 1],
    "reg_lambda": [8, 10, 12],
    "reg_alpha": [0.05, 0.1, 0.2],
}

# Params base (reutilizables)
fixed_params_base = dict(
    objective="binary:logistic",
    # Dejar alto para que early-stopping tenga margen real
    n_estimators=5000,
    random_state=42,
    tree_method="hist",
    eval_metric="logloss",
    early_stopping_rounds=100,
    subsample=0.9,
    colsample_bytree=0.8,
)

manual_params_base = dict(
    learning_rate=0.07,
    max_depth=1,
    min_child_weight=8,
    gamma=0.0,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_lambda=8,
    reg_alpha=0.05,
    max_delta_step=1,
)

best_params_global: Optional[Dict] = None

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
    embargo = 0

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

    # Si desactivas el random search, usa configuración fija (sin re-balanceo)
    if not DO_RANDOM_SEARCH:
        scale_pos_weight = 1.0

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
    fixed_params = dict(fixed_params_base)
    manual_params = dict(manual_params_base)

    if not DO_RANDOM_SEARCH:
        best_params = manual_params
    elif DO_RANDOM_SEARCH and (TUNE_EACH_FOLD or best_params_global is None):
        # Nota: el tuning usa solo (X_tr -> X_val). No toca el test.
        best_params, _best_val_score = tune_xgb_random_search_timeval(
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
    else:
        best_params = best_params_global or {}

    # print(f"[Fold] usando best_params={best_params} ")
    fold_fixed_params = dict(fixed_params)







    # ===== Entrenar modelo =====
    # Importante: evitar claves duplicadas (p.ej. subsample/colsample) entre fixed y best_params.
    model_params = dict(fold_fixed_params)
    model_params.update(best_params)
    model_params["scale_pos_weight"] = scale_pos_weight
    model = XGBClassifier(**model_params)

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_es, y_es)],
        verbose=False,
    )

    last_model = model

    # Probabilidades
    proba = model.predict_proba(X_test)[:, 1]

    # ===== Regla de decisión: threshold vs TOP-K =====
    if THRESHOLD_METHOD == "top_k":
        k_frac = TOP_K_FRAC
        if k_frac is None:
            k_frac = float(np.clip(y_train.mean(), 0.0, 1.0))

        pred = top_k_signals(proba, k_frac=float(k_frac), min_k=int(TOP_K_MIN_K))
        cutoff = top_k_cutoff(proba, k=int(pred.sum()))
        fold_thresholds.append(float(cutoff))
    else:
        # ===== Selección de umbral (en validación temporal) =====
        # Nota: 0.5 rara vez es óptimo si la base-rate != 0.5.
        val_proba = model.predict_proba(X_score)[:, 1]
        best_thr, _thr_aux = choose_threshold(
            y_score,
            val_proba,
            method=THRESHOLD_METHOD,
            base_rate_train=float(y_train.mean()),
        )

        fold_thresholds.append(float(best_thr))
        pred = (proba >= float(best_thr)).astype(int)

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

# Scorecards (sin prints) para memoria
wf_metrics_scorecard = pd.DataFrame(
    {
        "Valor": [
            float(np.nanmean(accs)),
            float(np.nanmean(aucs)),
            float(np.nanmean(ap_scores)),
            float(np.nanmean(bal_accs)),
            float(np.nanmean(loglosses)),
            float(np.nanmean(briers)),
        ]
    },
    index=[
        "Accuracy (mean)",
        "ROC-AUC (mean)",
        "PR-AUC / AvgPrecision (mean)",
        "Balanced Accuracy (mean)",
        "LogLoss (mean)",
        "Brier score (mean)",
    ],
)
_save_table_figure(
    wf_metrics_scorecard,
    out_path=BASE_DIR / "walk_forward_metrics_scorecard.png",
    title=f"Walk-Forward — Métricas promedio (horizonte {HORIZON}m)",
)

thr_median = float(np.nanmedian(fold_thresholds)) if len(fold_thresholds) else 0.5
wf_baselines_table = pd.DataFrame(
    {
        "Accuracy": [float(np.nanmean(accs)), float(np.nanmean(baseline_accs))],
        "LogLoss": [float(np.nanmean(loglosses)), float(np.nanmean(baseline_loglosses))],
    },
    index=["Modelo (WF)", "Baseline"],
)
_save_table_figure(
    wf_baselines_table,
    out_path=BASE_DIR / "walk_forward_baselines_scorecard.png",
    title="Walk-Forward — Modelo vs Baselines",
    footer=f"Umbral elegido (mediana folds): {thr_median:.3f}",
)

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
thr_wf = float(np.nanmedian(fold_thresholds)) if len(fold_thresholds) else 0.5
plot_classification_timeline(
    wf_df,
    out_path=BASE_DIR / "walk_forward_classification.png",
    title=f"Walk-Forward — Clasificación {HORIZON}m (aciertos/errores por periodo)",
    threshold=thr_wf,
    year_locator=2,
)

# ── Calidad probabilística (WF) ──
plot_calibration_curve_wf(
    wf_df,
    out_path=BASE_DIR / "walk_forward_calibration.png",
    title=f"Walk-Forward — Calibration curve ({HORIZON}m)",
    n_bins=10,
)

plot_proba_hist_by_class(
    wf_df,
    out_path=BASE_DIR / "walk_forward_proba_hist_by_class.png",
    title=f"Walk-Forward — Distribución de P(sube) por clase ({HORIZON}m)",
    bins=25,
    kde=True,
)

plot_roc_pr_wf(
    wf_df,
    out_path=BASE_DIR / "walk_forward_roc_pr.png",
    title=f"Walk-Forward — ROC & PR ({HORIZON}m)",
    threshold=thr_wf,
)

plot_metrics_by_proba_bin(
    wf_df,
    out_path=BASE_DIR / "walk_forward_metrics_by_decile.png",
    title=f"Walk-Forward — Accuracy/Precision/Recall por decil (thr={thr_wf:.2f})",
    threshold=thr_wf,
    n_bins=10,
)

plot_cumulative_gains_wf(
    wf_df,
    out_path=BASE_DIR / "walk_forward_cumulative_gains.png",
    title=f"Walk-Forward — Cumulative gains / lift ({HORIZON}m)",
)

plot_rolling_accuracy_wf(
    wf_df,
    out_path=BASE_DIR / "walk_forward_rolling_accuracy_36m.png",
    title=f"Walk-Forward — Rolling Accuracy (36m, horizonte {HORIZON}m)",
    window=36,
)

# Performance por régimen macro (ej.: inflación alta/baja)
if "high_inflation" in df.columns:
    regime_series = df["high_inflation"].astype(float).map({1.0: "high_inflation", 0.0: "low_inflation"})
    _regime_df = plot_regime_performance_wf(
        wf_df,
        out_path=BASE_DIR / "walk_forward_regime_performance.png",
        title=f"Walk-Forward — Performance por régimen (high/low inflation)",
        regime_series=regime_series,
    )
    if not _regime_df.empty:
        print("\n[Regime performance]\n", _regime_df)

plot_equity_curve_directional_wf(
    wf_df,
    out_path=BASE_DIR / "walk_forward_equity_curve_directional.png",
    title=f"Walk-Forward — Equity curve direccional ({HORIZON}m)",
)


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

plot_confusion_matrix_wf(
    wf_df["actual"],
    wf_df["pred"],
    out_path=BASE_DIR / "confusion_matrix_walk_forward.png",
    title=f"Walk-Forward — Matriz de confusión {HORIZON}m",
    class_labels=("0 (no sube)", "1 (sube)"),
)


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

# ==============================
# Final roll-out en últimos 10 años
# ==============================
FINAL_ROLLOUT_MONTHS = 120
FINAL_GAP_MONTHS = HORIZON

rollout_end = pd.to_datetime(df.index.max())
rollout_start = rollout_end - pd.DateOffset(months=FINAL_ROLLOUT_MONTHS - 1)
rollout_df = df.loc[rollout_start:rollout_end].copy()

train_end_date = rollout_start - pd.DateOffset(months=FINAL_GAP_MONTHS)
train_df = df.loc[:train_end_date].copy()

if len(rollout_df) < 5 or len(train_df) < (min_train_size // 2):
    print(
        "[FinalRollout] No hay suficiente histórico para entrenar/validar "
        f"(train={len(train_df)} rollout={len(rollout_df)}). Saltando."
    )
else:
    X_train_full = train_df[features]
    y_train_full = train_df["target"].astype(int)

    X_roll = rollout_df[features]
    y_roll = rollout_df["target"].astype(int)

    # balanceo opcional (solo si la clase positiva es rara)
    pos = int((y_train_full == 1).sum())
    neg = int((y_train_full == 0).sum())
    scale_pos_weight = (neg / pos) if (pos > 0 and pos < neg) else 1.0
    if not DO_RANDOM_SEARCH:
        scale_pos_weight = 1.0

    # ===== Validación interna temporal (igual que walk-forward) =====
    val_size = int(len(X_train_full) * 0.2)
    gap = HORIZON
    tr_end = -(val_size + gap)

    X_tr = X_train_full.iloc[:tr_end]
    y_tr = y_train_full.iloc[:tr_end]
    X_val = X_train_full.iloc[-val_size:]
    y_val = y_train_full.iloc[-val_size:]

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

    # ===== Hiperparámetros =====
    fixed_params = dict(fixed_params_base)
    manual_params = dict(manual_params_base)

    if not DO_RANDOM_SEARCH:
        best_params = manual_params
    else:
        # Si hubo tuning walk-forward, reutiliza esos params; si no, cae al manual.
        best_params = best_params_global or manual_params

    model_roll = XGBClassifier(
        **fixed_params,
        **best_params,
        scale_pos_weight=scale_pos_weight,
    )

    model_roll.fit(
        X_tr,
        y_tr,
        eval_set=[(X_es, y_es)],
        verbose=False,
    )

    # ===== Predicción (rollout) =====
    roll_proba = model_roll.predict_proba(X_roll)[:, 1]

    if THRESHOLD_METHOD == "top_k":
        k_frac = TOP_K_FRAC
        if k_frac is None:
            k_frac = float(np.clip(y_train_full.mean(), 0.0, 1.0))

        roll_pred = top_k_signals(roll_proba, k_frac=float(k_frac), min_k=int(TOP_K_MIN_K))
        best_thr = top_k_cutoff(roll_proba, k=int(roll_pred.sum()))
    else:
        # ===== Umbral (validación temporal) =====
        val_proba = model_roll.predict_proba(X_score)[:, 1]
        best_thr, _thr_aux = choose_threshold(
            y_score,
            val_proba,
            method=THRESHOLD_METHOD,
            base_rate_train=float(y_train_full.mean()),
        )

        roll_pred = (roll_proba >= float(best_thr)).astype(int)

    # ===== Métricas rollout =====
    roll_acc = float(accuracy_score(y_roll, roll_pred))
    roll_bal_acc = float(balanced_accuracy_score(y_roll, roll_pred))
    roll_logloss = float(log_loss(y_roll, roll_proba, labels=[0, 1]))
    roll_brier = float(brier_score_loss(y_roll, roll_proba))
    if len(np.unique(y_roll)) > 1:
        roll_auc = float(roc_auc_score(y_roll, roll_proba))
        roll_ap = float(average_precision_score(y_roll, roll_proba))
    else:
        roll_auc = float("nan")
        roll_ap = float("nan")

    # Baselines (comparación justa, usando solo train)
    majority_class = int(y_train_full.mean() >= 0.5)
    baseline_pred = np.full_like(y_roll.values, fill_value=majority_class)
    baseline_acc = float(accuracy_score(y_roll, baseline_pred))
    p0 = float(np.clip(y_train_full.mean(), 1e-6, 1 - 1e-6))
    baseline_ll = float(log_loss(y_roll, np.full_like(roll_proba, p0), labels=[0, 1]))

    print("\n[FinalRollout] Ventana:", str(rollout_df.index.min().date()), "->", str(rollout_df.index.max().date()))
    print("[FinalRollout] Train hasta:", str(train_df.index.max().date()), f"(gap={FINAL_GAP_MONTHS}m)")
    print("[FinalRollout] Umbral elegido:", float(best_thr))
    print("[FinalRollout] Accuracy:", roll_acc)
    print("[FinalRollout] Balanced Accuracy:", roll_bal_acc)
    print("[FinalRollout] ROC-AUC:", roll_auc)
    print("[FinalRollout] PR-AUC:", roll_ap)
    print("[FinalRollout] LogLoss:", roll_logloss)
    print("[FinalRollout] Brier:", roll_brier)
    print("[FinalRollout] Baseline Accuracy (mayoría train):", baseline_acc)
    print("[FinalRollout] Baseline LogLoss (p const train):", baseline_ll)

    roll_plot_df = pd.DataFrame(
        {
            "date": pd.to_datetime(rollout_df.index),
            "proba_up": roll_proba,
            "pred": roll_pred,
            "actual": y_roll.values,
            "close_t": rollout_df["Close"].values,
            "close_t_plus_h": rollout_df["close_fwd"].values,
        }
    ).sort_values("date")

    plot_classification_timeline(
        roll_plot_df,
        out_path=BASE_DIR / "final_rollout_classification.png",
        title=f"Final Roll-out — Clasificación {HORIZON}m (aciertos/errores por periodo)",
        threshold=float(best_thr),
        year_locator=1,
    )

    # ==============================
    # ROI: Buy&Hold DCA vs Señal (2x cuando pred=1) — Final Roll-out
    # ==============================
    roll_signal = roll_plot_df[["date", "pred"]].copy()
    roll_signal["date"] = pd.to_datetime(roll_signal["date"])
    roll_signal = roll_signal.set_index("date").sort_index()

    prices_eval_roll = pd.Series(
        roll_plot_df["close_t"].values,
        index=pd.to_datetime(roll_plot_df["date"]),
        name="Close",
    ).sort_index()

    pred_aligned_roll = roll_signal["pred"].reindex(prices_eval_roll.index)

    # Comparación justa: usar SOLO los meses con predicción.
    has_pred_roll = pred_aligned_roll.notna()
    prices_eval_roll = prices_eval_roll.loc[has_pred_roll]
    pred_aligned_roll = pred_aligned_roll.loc[has_pred_roll]

    contrib_bh_roll = pd.Series(monthly_amount, index=prices_eval_roll.index)
    contrib_signal_roll = pd.Series(0.0, index=prices_eval_roll.index)
    contrib_signal_roll[pred_aligned_roll.fillna(0).astype(int) == 1] = monthly_amount * signal_multiplier

    bh_curve_roll = simulate_monthly_dca_roi(prices_eval_roll, contrib_bh_roll)
    sig_curve_roll = simulate_monthly_dca_roi(prices_eval_roll, contrib_signal_roll)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        bh_curve_roll.index,
        bh_curve_roll["roi_pct"],
        label=f"Buy&Hold DCA (x={monthly_amount:g}/mes)",
        color="tab:blue",
    )
    ax.plot(
        sig_curve_roll.index,
        sig_curve_roll["roi_pct"],
        label=f"Señal (comprar {signal_multiplier:g}x si pred=1)",
        color="purple",
    )

    ax.axhline(0, linestyle="--", color="grey", alpha=0.6)
    ax.set_title(f"ROI acumulado (%) — DCA mensual vs Señal (Final Roll-out, horizonte {HORIZON}m)")
    ax.set_ylabel("ROI (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "roi_strategies_final_rollout.png")

    print("\n[FinalRollout ROI] ROI final Buy&Hold DCA (%):", float(bh_curve_roll["roi_pct"].dropna().iloc[-1]))
    print("[FinalRollout ROI] ROI final Señal 2x si pred=1 (%):", float(sig_curve_roll["roi_pct"].dropna().iloc[-1]))
    print("[FinalRollout ROI] Total invertido Buy&Hold:", float(bh_curve_roll["invested"].dropna().iloc[-1]))
    print("[FinalRollout ROI] Total invertido Señal:", float(sig_curve_roll["invested"].dropna().iloc[-1]))














# ===== Modelo final entrenado en todo el dataset =====
final_fixed_params = dict(
    objective="binary:logistic",
    n_estimators=5000,
    random_state=42,
    tree_method="hist",
    eval_metric="auc",
    subsample=0.9,
    colsample_bytree=0.8,
)

final_best_params = manual_params_base if not DO_RANDOM_SEARCH else (best_params_global or manual_params_base)
final_scale_pos_weight = 1.0 if not DO_RANDOM_SEARCH else float((y == 0).sum() / max(1, (y == 1).sum()))

final_model = XGBClassifier(
    **final_fixed_params,
    **final_best_params,
    scale_pos_weight=final_scale_pos_weight,
)

final_model.fit(X, y)

last_X = X.iloc[[-1]]
final_proba = float(final_model.predict_proba(last_X)[:, 1][0])
final_pred = int(final_proba >= 0.5)

print("Última fecha:", X.index[-1])
print(f"P(sube) en {HORIZON} meses:", final_proba)
print("Predicción clase (>=0.5):", final_pred)

# ===== Métricas (in-sample, sobre todo el dataset) =====
final_proba_all = final_model.predict_proba(X)[:, 1]
final_pred_all = (final_proba_all >= 0.5).astype(int)

final_acc = float(accuracy_score(y, final_pred_all))
final_bal_acc = float(balanced_accuracy_score(y, final_pred_all))
final_logloss = float(log_loss(y, final_proba_all, labels=[0, 1]))
final_brier = float(brier_score_loss(y, final_proba_all))
if len(np.unique(y)) > 1:
    final_auc = float(roc_auc_score(y, final_proba_all))
    final_ap = float(average_precision_score(y, final_proba_all))
else:
    final_auc = float("nan")
    final_ap = float("nan")

# Baselines (comparación justa)
majority_class = int(y.mean() >= 0.5)
baseline_pred = np.full_like(y.values, fill_value=majority_class)
baseline_acc = float(accuracy_score(y, baseline_pred))
p0 = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
baseline_ll = float(log_loss(y, np.full_like(final_proba_all, p0), labels=[0, 1]))

print("\n[FinalModel] Accuracy:", final_acc)
print("[FinalModel] Balanced Accuracy:", final_bal_acc)
print("[FinalModel] ROC-AUC:", final_auc)
print("[FinalModel] PR-AUC:", final_ap)
print("[FinalModel] LogLoss:", final_logloss)
print("[FinalModel] Brier:", final_brier)
print("[FinalModel] Baseline Accuracy (mayoría train):", baseline_acc)
print("[FinalModel] Baseline LogLoss (p const train):", baseline_ll)

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

top_features = [
    "m2_yoy",
    "permit_yoy",
    "equity_risk_premium",
    "sp500_earnings_yield",
    "HOUST",
    "unemp_change_12m",
    "gdp_yoy_lag6"
]

for feature in top_features:
    plt.figure(figsize=(6, 4))
    
    shap.dependence_plot(
        feature,
        shap_values_pos,  
        X,
        interaction_index="auto",  # muestra interacción automáticamente
        show=False
    )
    
    fname = f"shap_dependence_cls_{feature}.png"
    plt.tight_layout()
    plt.savefig(BASE_DIR / fname, dpi=120)
    plt.close()

# SHAP para la última predicción (clase positiva)
shap_values_last = explainer.shap_values(last_X)
if isinstance(shap_values_last, list) and len(shap_values_last) == 2:
    shap_values_last_pos = shap_values_last[1][0]
else:
    shap_values_last_pos = shap_values_last[0]

plt.figure()
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values_last_pos,
        base_values=expected_value,
        data=last_X.iloc[0],
        feature_names=X.columns
    ),
    show=False
)
plt.tight_layout()
plt.savefig(BASE_DIR / "shap_last_prediction_cls.png")