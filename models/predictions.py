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
    confusion_matrix,
    f1_score,
    roc_auc_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def _binary_logloss(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    y_proba = np.clip(y_proba, 1e-9, 1.0 - 1e-9)
    return float(-np.mean(y_true * np.log(y_proba) + (1.0 - y_true) * np.log(1.0 - y_proba)))


def _best_threshold_by_f1(
    y_true: pd.Series,
    y_proba: np.ndarray,
    *,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    y_arr = np.asarray(y_true, dtype=int)
    if len(np.unique(y_arr)) < 2:
        return 0.5, float("nan")

    proba = np.asarray(y_proba, dtype=float)
    proba = np.clip(proba, 0.0, 1.0)

    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 50)

    best_t = 0.5
    best_f1 = -np.inf
    for t in thresholds:
        preds = (proba >= float(t)).astype(int)
        score = float(f1_score(y_arr, preds, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_t = float(t)

    return best_t, float(best_f1)


def _precision_at_k(y_true: pd.Series, y_proba: np.ndarray, *, top_frac: float = 0.2) -> float:
    y_arr = np.asarray(y_true, dtype=int)
    proba = np.asarray(y_proba, dtype=float)
    if len(y_arr) == 0:
        return float("nan")

    k = max(1, int(float(top_frac) * len(y_arr)))
    idx = np.argsort(proba)[-k:]
    return float(y_arr[idx].mean())


def _lift_at_k(y_true: pd.Series, y_proba: np.ndarray, *, top_frac: float = 0.2) -> float:
    y_arr = np.asarray(y_true, dtype=int)
    if len(y_arr) == 0:
        return float("nan")

    base_rate = float(y_arr.mean())
    if base_rate <= 0.0:
        return float("nan")

    return float(_precision_at_k(y_arr, y_proba, top_frac=top_frac) / base_rate)


def _confusion_matrix_by_thresholds(
    y_true: pd.Series,
    y_proba: np.ndarray,
    *,
    thresholds: Tuple[float, ...] = (0.5, 0.7, 0.8),
) -> pd.DataFrame:
    y_arr = np.asarray(y_true, dtype=int)
    proba = np.asarray(y_proba, dtype=float)
    rows = []

    for threshold in thresholds:
        y_pred = (proba >= float(threshold)).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_arr, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "precision": float(precision_score(y_arr, y_pred, zero_division=0)),
                "recall_1": float(recall_score(y_arr, y_pred, pos_label=1, zero_division=0)),
                "recall_0": float(recall_score(y_arr, y_pred, pos_label=0, zero_division=0)),
            }
        )

    return pd.DataFrame(rows).set_index("threshold")


def plot_confusion_matrix_heatmap(
    y_true: pd.Series,
    y_pred: np.ndarray,
    *,
    out_path: Path,
    title: str,
) -> None:
    y_arr = np.asarray(y_true, dtype=int)
    y_hat = np.asarray(y_pred, dtype=int)
    if y_arr.size == 0 or y_hat.size == 0:
        return
    if y_arr.size != y_hat.size:
        return

    cm = confusion_matrix(y_arr, y_hat, labels=[0, 1])
    if cm.shape != (2, 2):
        return
    tn, fp, fn, tp = cm.ravel()

    mat = np.array([[tn, fp], [fn, tp]], dtype=float)
    vmax = float(np.max(mat)) if np.isfinite(mat).any() else 1.0

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(mat, cmap="Blues", vmin=0.0, vmax=max(1.0, vmax))

    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["Real 0", "Real 1"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(title)

    labels = np.array([["TN", "FP"], ["FN", "TP"]], dtype=object)
    for i in range(2):
        for j in range(2):
            val = int(mat[i, j])
            txt = f"{labels[i, j]}\n{val}"  # solo etiqueta + count
            color = "white" if (vmax > 0 and mat[i, j] / vmax > 0.55) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=14, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Conteo")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


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
) -> Tuple[Dict, float]:
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
        model = XGBClassifier(**model_params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_es, y_es)],
            verbose=False,
            early_stopping_rounds=model_params.get("early_stopping_rounds", 100),
        )
        score_proba = model.predict_proba(X_score)[:, 1]

        # Optimizamos LogLoss directamente (más estable que AUC con base-rate alta).
        # Clampeamos para evitar inf por probabilidades 0/1.
        score_proba = np.clip(score_proba, 1e-6, 1 - 1e-6)

        score_ll = _binary_logloss(y_score.values, score_proba)

        if score_ll < best_logloss:
            best_logloss = float(score_ll)
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
) -> None:
    if corr is None or corr.empty:
        return

    corr_plot = corr.copy()
    if corr_plot.shape[0] > max_vars:
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

def plot_classification_timeline(
    plot_df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    year_locator: int = 2,
    date_col: str = "date",
    proba_col: str = "proba_up",
    actual_col: str = "actual",
    pred_col: str = "pred",
    price_col: str = "close_t",
    price_fwd_col: str = "close_t_plus_h",
) -> None:
    dfp = plot_df.copy()
    dfp[date_col] = pd.to_datetime(dfp[date_col])
    dfp = dfp.sort_values(date_col).reset_index(drop=True)

    if dfp.empty:
        return

    y_true = dfp[actual_col].astype(int).to_numpy()
    y_proba = np.clip(dfp[proba_col].astype(float).to_numpy(), 0.0, 1.0) if proba_col in dfp.columns else None
    if pred_col in dfp.columns:
        y_pred = dfp[pred_col].astype(int).to_numpy()
    elif y_proba is not None:
        y_pred = (y_proba >= 0.5).astype(int)
    else:
        y_pred = np.zeros_like(y_true)
    hit_mask = y_pred == y_true

    auc = float("nan")
    if y_proba is not None and len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_proba))
    ll = float("nan")
    br = float("nan")
    if y_proba is not None:
        ll = _binary_logloss(y_true, y_proba)
        br = float(brier_score_loss(y_true, y_proba))

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(14, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.4]},
    )

    # Clase real como escalón
    # ax0.plot(
    #     dfp[date_col],
    #     dfp[actual_col].astype(int),
    #     label="Clase real (0/1)",
    #     color="black",
    #     alpha=0.70,
    #     lw=1.6,
    #     drawstyle="steps-post",
    #     zorder=3,
    # )
    # Clase predicha + color por acierto/error
    ax0.plot(
        dfp[date_col],
        y_pred,
        label="Predicción de clase (0/1)",
        color="tab:gray",
        alpha=0.35,
        lw=1.1,
        drawstyle="steps-post",
        zorder=2,
    )
    ax0.scatter(
        dfp[date_col],
        y_pred,
        c=np.where(hit_mask, "green", "red"),
        s=18,
        alpha=0.85,
        # label="",
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

    lines0, labels0 = ax0.get_legend_handles_labels()
    lines0b, labels0b = ax0b.get_legend_handles_labels()
    ax0.legend(
        lines0 + lines0b,
        labels0 + labels0b,
        loc="upper left",
        ncol=2,
        fontsize=9,
    )

    if y_proba is not None:
        ax1.plot(
            dfp[date_col],
            y_proba,
            color="purple",
            alpha=0.50,
            lw=1.3,
            label="P(sube)",
        )
        ax1.set_ylim(0.0, 1.0)
        ax1.set_ylabel("Prob.")
        ax1.legend(loc="upper left", ncol=2, fontsize=9)

    ax1.set_xlabel("Fecha")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator(year_locator))

    ax0.grid(True, alpha=0.25)
    ax1.grid(True, alpha=0.25)

    fig.suptitle(f"{title}\nAUC={auc:.3f} | LogLoss={ll:.3f} | Brier={br:.3f}", y=0.98)
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
        fpr, tpr, _roc_cutpoints = roc_curve(y_true, y_proba)
        roc_auc = float(roc_auc_score(y_true, y_proba))
        ax_roc.plot(fpr, tpr, lw=1.8, label=f"ROC-AUC={roc_auc:.3f}")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1.0)

        ax_roc.set_title("ROC")
        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        ax_roc.set_xlim(0.0, 1.0)
        ax_roc.set_ylim(0.0, 1.0)
        ax_roc.grid(True, alpha=0.25)
        ax_roc.legend(loc="lower right")

        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = float(average_precision_score(y_true, y_proba))
        base_rate = float(np.mean(y_true))
        ax_pr.plot(recall, precision, lw=1.8, label=f"AP={ap:.3f}")
        ax_pr.axhline(base_rate, linestyle="--", color="grey", lw=1.0, label=f"Base-rate={base_rate:.3f}")

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
    n_bins: int = 10,
    actual_col: str = "actual",
    proba_col: str = "proba_up",
) -> None:
    dfp = wf_df[[actual_col, proba_col]].copy()
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna()
    if dfp.empty:
        return

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
        n = int(len(g))

        mean_proba = float(np.mean(g[proba_col].astype(float))) if n > 0 else np.nan
        emp_rate = float(np.mean(y_true)) if n > 0 else np.nan

        rows.append({
            "bin": int(b),
            "n": n,
            "mean_proba": mean_proba,
            "empirical_rate": emp_rate,
        })

    mdf = pd.DataFrame(rows).sort_values("bin")

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(mdf["bin"], mdf["empirical_rate"], marker="o", lw=1.8, label="P(real=1) por bin")
    ax.plot(mdf["bin"], mdf["mean_proba"], marker="o", lw=1.8, label="Mean P(sube) por bin")

    ax.set_title(title)
    ax.set_xlabel("Decil de P(sube) (bajo → alto)")
    ax.set_ylabel("Probabilidad")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", ncol=2)
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

        # Nota: evitamos marcar un "punto operativo" o selección fija; solo curva.

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


def plot_rolling_logloss_wf(
    wf_df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    window: int = 36,
    date_col: str = "date",
    actual_col: str = "actual",
    proba_col: str = "proba_up",
) -> None:
    dfp = wf_df[[date_col, actual_col, proba_col]].copy()
    dfp[date_col] = pd.to_datetime(dfp[date_col])
    dfp = dfp.sort_values(date_col)
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna()
    if dfp.empty:
        return

    y_true = dfp[actual_col].astype(int)
    y_proba = np.clip(dfp[proba_col].astype(float), 1e-9, 1.0 - 1e-9)
    point_ll = -(y_true * np.log(y_proba) + (1.0 - y_true) * np.log(1.0 - y_proba))
    roll = point_ll.rolling(int(window), min_periods=max(5, int(window // 3))).mean()

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(dfp[date_col], roll, color="tab:blue", lw=1.8, label=f"Rolling LogLoss ({int(window)}m)")
    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("LogLoss")
    ax.set_ylim(0.0, max(0.05, float(np.nanquantile(roll.dropna(), 0.98))) if roll.notna().any() else 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_rolling_brier_wf(
    wf_df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    window: int = 36,
    date_col: str = "date",
    actual_col: str = "actual",
    proba_col: str = "proba_up",
) -> None:
    dfp = wf_df[[date_col, actual_col, proba_col]].copy()
    dfp[date_col] = pd.to_datetime(dfp[date_col])
    dfp = dfp.sort_values(date_col)
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna()
    if dfp.empty:
        return

    y_true = dfp[actual_col].astype(float)
    y_proba = np.clip(dfp[proba_col].astype(float), 0.0, 1.0)
    point_bs = (y_true - y_proba) ** 2
    roll = point_bs.rolling(int(window), min_periods=max(5, int(window // 3))).mean()

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(dfp[date_col], roll, color="tab:orange", lw=1.8, label=f"Rolling Brier ({int(window)}m)")
    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Brier")
    ax.set_ylim(0.0, max(0.05, float(np.nanquantile(roll.dropna(), 0.98))) if roll.notna().any() else 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def compute_calibration_deciles_table(
    df: pd.DataFrame,
    *,
    n_bins: int = 10,
    actual_col: str = "actual",
    proba_col: str = "proba_up",
) -> pd.DataFrame:
    dfp = df[[actual_col, proba_col]].copy()
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna()
    if dfp.empty:
        return pd.DataFrame()

    y_true_all = dfp[actual_col].astype(int).to_numpy()
    y_proba_all = np.clip(dfp[proba_col].astype(float).to_numpy(), 1e-9, 1.0 - 1e-9)
    base_rate = float(np.mean(y_true_all)) if len(y_true_all) else np.nan

    # Deciles por cuantiles; con duplicados puede haber < n_bins.
    dfp["bin"] = pd.qcut(
        dfp[proba_col].astype(float),
        int(n_bins),
        labels=False,
        duplicates="drop",
    )

    rows = []
    for b, g in dfp.groupby("bin"):
        y_true = g[actual_col].astype(int).to_numpy()
        y_proba = np.clip(g[proba_col].astype(float).to_numpy(), 1e-9, 1.0 - 1e-9)
        n = int(len(g))

        mean_proba = float(np.mean(y_proba)) if n else np.nan
        emp_rate = float(np.mean(y_true)) if n else np.nan
        ll = float(_binary_logloss(y_true, y_proba)) if n else np.nan
        br = float(brier_score_loss(y_true, y_proba)) if n else np.nan
        lift = float(emp_rate / base_rate) if (n and base_rate and base_rate > 0) else np.nan

        rows.append(
            {
                "decil": int(b) + 1,
                "n": n,
                "p_min": float(np.min(y_proba)) if n else np.nan,
                "p_max": float(np.max(y_proba)) if n else np.nan,
                "mean_proba": mean_proba,
                "empirical_rate": emp_rate,
                "lift_vs_base": lift,
                "logloss": ll,
                "brier": br,
            }
        )

    out = pd.DataFrame(rows).sort_values("decil")
    if out.empty:
        return out

    out = out.set_index("decil")
    return out


def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> float:
    dfp = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "p": np.asarray(y_proba, dtype=float)})
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna()
    if dfp.empty:
        return float("nan")

    dfp["p"] = np.clip(dfp["p"].astype(float), 1e-9, 1.0 - 1e-9)
    dfp["y"] = dfp["y"].astype(int)

    if str(strategy).lower() == "quantile":
        dfp["bin"] = pd.qcut(dfp["p"], int(n_bins), labels=False, duplicates="drop")
    else:
        edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
        dfp["bin"] = pd.cut(dfp["p"], bins=edges, labels=False, include_lowest=True)

    n_total = float(len(dfp))
    if n_total <= 0:
        return float("nan")

    ece = 0.0
    for _, g in dfp.groupby("bin"):
        n_k = float(len(g))
        if n_k <= 0:
            continue
        p_k = float(g["p"].mean())
        o_k = float(g["y"].mean())
        ece += (n_k / n_total) * abs(o_k - p_k)

    return float(ece)


def brier_decomposition(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> Dict[str, float]:
    dfp = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "p": np.asarray(y_proba, dtype=float)})
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna()
    if dfp.empty:
        return {
            "brier": float("nan"),
            "reliability": float("nan"),
            "resolution": float("nan"),
            "uncertainty": float("nan"),
            "brier_decomp": float("nan"),
            "n_bins_eff": float("nan"),
        }

    dfp["p"] = np.clip(dfp["p"].astype(float), 1e-9, 1.0 - 1e-9)
    dfp["y"] = dfp["y"].astype(int)

    if str(strategy).lower() == "quantile":
        dfp["bin"] = pd.qcut(dfp["p"], int(n_bins), labels=False, duplicates="drop")
    else:
        edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
        dfp["bin"] = pd.cut(dfp["p"], bins=edges, labels=False, include_lowest=True)

    y_bar = float(dfp["y"].mean())
    uncertainty = float(y_bar * (1.0 - y_bar))

    n_total = float(len(dfp))
    reliability = 0.0
    resolution = 0.0
    n_bins_eff = 0
    for _, g in dfp.groupby("bin"):
        n_k = float(len(g))
        if n_k <= 0:
            continue
        n_bins_eff += 1
        w_k = n_k / n_total
        p_k = float(g["p"].mean())
        o_k = float(g["y"].mean())
        reliability += w_k * (p_k - o_k) ** 2
        resolution += w_k * (o_k - y_bar) ** 2

    brier = float(np.mean((dfp["y"].astype(float) - dfp["p"].astype(float)) ** 2))
    brier_decomp = float(reliability - resolution + uncertainty)
    return {
        "brier": brier,
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "brier_decomp": brier_decomp,
        "n_bins_eff": float(n_bins_eff),
    }


def _max_drawdown_from_equity(equity: np.ndarray) -> float:
    eq = np.asarray(equity, dtype=float)
    if eq.size < 2:
        return float("nan")
    eq = np.where(np.isfinite(eq), eq, np.nan)
    if np.all(np.isnan(eq)):
        return float("nan")
    eq = pd.Series(eq).ffill().bfill().to_numpy(dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = eq / np.where(peak == 0, np.nan, peak) - 1.0
    return float(np.nanmin(dd))


def compute_return_risk_metrics(
    returns: np.ndarray,
    *,
    periods_per_year: float = 12.0,
    risk_free_rate_annual: float = 0.0,
) -> Dict[str, float]:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n < 2:
        return {
            "n": float(n),
            "total_return": float("nan"),
            "cagr": float("nan"),
            "vol": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "mean_ret": float("nan"),
            "std_ret": float("nan"),
        }

    equity = np.cumprod(1.0 + np.nan_to_num(r, nan=0.0))
    total_return = float(equity[-1] - 1.0)

    years = float(n) / float(periods_per_year) if periods_per_year else float("nan")
    cagr = float(equity[-1] ** (1.0 / years) - 1.0) if years and years > 0 and equity[-1] > 0 else float("nan")

    vol = float(np.std(r, ddof=1))
    ann_vol = float(vol * np.sqrt(float(periods_per_year))) if periods_per_year else float("nan")

    # Sharpe sobre CAGR aproximado (rf anual) / vol anual.
    excess_ann = float(cagr - float(risk_free_rate_annual)) if np.isfinite(cagr) else float("nan")
    sharpe = float(excess_ann / ann_vol) if ann_vol and ann_vol > 0 and np.isfinite(excess_ann) else float("nan")

    mdd = _max_drawdown_from_equity(equity)
    return {
        "n": float(n),
        "total_return": total_return,
        "cagr": cagr,
        "vol": vol,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "mean_ret": float(np.mean(r)),
        "std_ret": float(np.std(r, ddof=1)),
    }


def compute_signal_stability_metrics(signal: pd.Series) -> Dict[str, float]:
    s = signal.copy()
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return {
            "n": float(0),
            "change_pct": float("nan"),
            "pct_long": float("nan"),
            "avg_hold_long": float("nan"),
            "avg_hold_flat": float("nan"),
        }

    # Normaliza a {0,1} si viene como bool/float.
    s_bin = (s.astype(float) > 0.5).astype(int)
    n = int(len(s_bin))
    if n < 2:
        return {
            "n": float(n),
            "change_pct": float("nan"),
            "pct_long": float(s_bin.mean()),
            "avg_hold_long": float("nan"),
            "avg_hold_flat": float("nan"),
        }

    change_pct = float((s_bin != s_bin.shift(1)).iloc[1:].mean())
    pct_long = float(s_bin.mean())

    run_id = (s_bin != s_bin.shift(1)).cumsum()
    run_len = s_bin.groupby(run_id).size().astype(float)
    run_val = s_bin.groupby(run_id).first().astype(int)

    avg_hold_long = float(run_len[run_val == 1].mean()) if (run_val == 1).any() else float("nan")
    avg_hold_flat = float(run_len[run_val == 0].mean()) if (run_val == 0).any() else float("nan")

    return {
        "n": float(n),
        "change_pct": change_pct,
        "pct_long": pct_long,
        "avg_hold_long": avg_hold_long,
        "avg_hold_flat": avg_hold_flat,
    }


def compute_exposure_turnover(exposure: pd.Series) -> Dict[str, float]:
    x = exposure.copy()
    x = x.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(x) < 2:
        return {"n": float(len(x)), "mean_abs_change": float("nan"), "median_abs_change": float("nan")}

    dx = x.diff().abs().iloc[1:]
    return {
        "n": float(len(x)),
        "mean_abs_change": float(dx.mean()),
        "median_abs_change": float(dx.median()),
    }


def plot_regime_performance_wf(
    wf_df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    regime_series: pd.Series,
    date_col: str = "date",
    actual_col: str = "actual",
    proba_col: str = "proba_up",
) -> pd.DataFrame:
    dfp = wf_df[[date_col, actual_col, proba_col]].copy()
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
        n = int(len(g))
        y_proba = np.clip(g[proba_col].astype(float).to_numpy(), 0.0, 1.0)
        base_rate = float(np.mean(y_true)) if n > 0 else np.nan

        ll = _binary_logloss(y_true, y_proba) if n > 0 else np.nan
        br = float(brier_score_loss(y_true, y_proba)) if n > 0 else np.nan

        auc = float("nan")
        ap = float("nan")
        if n > 2 and len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, y_proba))
            ap = float(average_precision_score(y_true, y_proba))
        rows.append(
            {
                "regime": str(regime_name),
                "n": n,
                "base_rate": base_rate,
                "mean_proba": float(np.mean(y_proba)) if n > 0 else np.nan,
                "logloss": float(ll),
                "brier": float(br),
                "auc": float(auc) if not pd.isna(auc) else np.nan,
                "ap": float(ap) if not pd.isna(ap) else np.nan,
            }
        )

    rdf = pd.DataFrame(rows).sort_values("regime")
    if rdf.empty:
        return rdf

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(rdf))
    w = 0.22
    ax.bar(x - w, rdf["auc"], width=w, label="AUC")
    ax.bar(x, rdf["brier"], width=w, label="Brier")
    ax.bar(x + w, rdf["logloss"], width=w, label="LogLoss")
    ax.set_xticks(x, labels=rdf["regime"].tolist())
    ax.set_ylim(0.0, max(1.05, float(np.nanmax(rdf[["auc", "brier", "logloss"]].to_numpy())) * 1.05))
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
    proba_col: str = "proba_up",
    close_col: str = "close_t",
    close_fwd_col: str = "close_t_plus_h",
) -> None:
    dfp = wf_df[[date_col, proba_col, close_col, close_fwd_col]].copy()
    dfp[date_col] = pd.to_datetime(dfp[date_col])
    dfp = dfp.sort_values(date_col)
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna()
    if dfp.empty:
        return

    fwd_ret = dfp[close_fwd_col].astype(float).to_numpy() / dfp[close_col].astype(float).to_numpy() - 1.0
    exposure = np.clip(dfp[proba_col].astype(float).to_numpy(), 0.0, 1.0)
    strat_ret = exposure * fwd_ret

    equity_strat = np.cumprod(1.0 + np.nan_to_num(strat_ret, nan=0.0))
    equity_bh = np.cumprod(1.0 + np.nan_to_num(fwd_ret, nan=0.0))

    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    ax.plot(dfp[date_col], equity_bh, lw=1.8, color="tab:blue", alpha=0.75, label="Buy&Hold (horizon)")
    ax.plot(dfp[date_col], equity_strat, lw=2.0, color="purple", label="Estrategia (exposure=P(sube))")
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
HORIZON = 36

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
    "liquidity_trend",


    # "high_inflation",
    # "gdp_yoy_lag6",
    # "recession",
    # "rsi_14",
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
    # f"roc_{HORIZON}",
]

# features = [
#     "equity_risk_premium",
#     "credit_spread",
#     "unemp_change_12m",
#     "m2_yoy",
#     "permit_yoy",
# ]
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

   

print( "features con suficiente historia:", features)

# TARGET  
min_train_size = 120
test_size = 12 
close_fwd = df["Close"].shift(-HORIZON)
df["close_fwd"] = close_fwd
df["future_return"] = close_fwd / df["Close"] - 1
df["target_reg"] = np.where(
    close_fwd.notna(),
    np.log(close_fwd / df["Close"]),
    np.nan,
)







# risk_free = (df["TB3MS"] / 100.0) * (HORIZON / 12)
# df["target"] = np.where(
#     close_fwd.notna() & df["TB3MS"].notna(),
#     df["future_return"] > risk_free,
#     np.nan,
# )




# ret = df["future_return"]
# df["target"] = np.where(
#     ret > 0.06, 1,
#     np.where(ret < -0.02, 0, np.nan)
# )




df["target"] = np.where(
    close_fwd.notna() & df["Close"].notna(),
    (close_fwd > df["Close"]),
    np.nan,
)








df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["target"])
df["target"] = df["target"].astype(int)
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
RANDOM_SEARCH_N_ITER = 180
RANDOM_SEARCH_SEED = 42
SCORE_FRAC = 0.5  # fracción del bloque de validación reservada para scoring/reporting

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
    n_estimators=5000,        
    random_state=42,
    tree_method="hist",
    eval_metric="logloss",
    early_stopping_rounds=100,
)

manual_params_base = dict(
    learning_rate=0.03,
    max_depth=5,
    min_child_weight=5,
    gamma=1.0,
    reg_lambda=9,
    reg_alpha=0.1,    
    subsample=0.9,
    colsample_bytree=0.8,
)

best_params_global: Optional[Dict] = None

# --- Walk-forward metrics ---
aucs = []
loglosses = []
ap_scores = []
briers = []
balanced_accs = []
mccs = []
precisions = []
recalls = []
recalls_0 = []
f1s = []
accuracies = []
precision_at_top20s = []
lift_top20s = []

baseline_loglosses = []
baseline_briers = []

all_proba = []
all_actuals = []
all_dates = []
all_close = []
all_close_fwd = []
all_pred = []
all_thresholds = []

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

    # ===== Validation interna temporal =====
    val_size = int(len(X_train) * 0.2)
    gap = HORIZON
    tr_end = -(val_size + gap)

    X_tr = X_train.iloc[:tr_end]
    y_tr = y_train.iloc[:tr_end]

    X_val = X_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]

    # if len(np.unique(y_tr)) < 2:
    #     start = test_end + embargo
    #     continue

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





    if len(np.unique(y_tr)) < 2:
        # XGBoost no puede entrenar clasificación binaria con una sola clase.
        # En esos folds usamos un predictor constante basado solo en el train disponible.
        best_t, _best_f1 = 0.5, float("nan")
        p_const = float(np.clip(y_tr.mean(), 1e-6, 1.0 - 1e-6))
        proba = np.full(len(X_test), p_const, dtype=float)
        y_pred = (proba >= best_t).astype(int)
    else:
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

        model = XGBClassifier(**model_params)
        # ===== CHECK: evitar validation inválida =====
        if len(np.unique(y_es)) < 2:
            model.set_params(early_stopping_rounds=None)
            model.fit(
                X_tr,
                y_tr,
                verbose=False,
            )
        else:
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_es, y_es)],
                verbose=False,
            )
        last_model = model

        # ===== Threshold óptimo en validation (para F1) =====
        if len(X_score) < 1:
            best_t, _best_f1 = 0.5, float("nan")
        else:
            score_proba = model.predict_proba(X_score)[:, 1]
            best_t, _best_f1 = _best_threshold_by_f1(y_score, score_proba)

        # Probabilidades
        proba = model.predict_proba(X_test)[:, 1]
        # proba = 1 - proba

        # Predicción de clase usando threshold optimizado (NO 0.5 fijo)
        y_pred = (proba >= best_t).astype(int)

    all_thresholds.append(float(best_t))

    all_proba.extend(proba.tolist())
    all_actuals.extend(y_test.values.tolist())
    all_dates.extend(y_test.index.tolist())
    all_close.extend(test_df["Close"].values.tolist())
    all_close_fwd.extend(test_df["close_fwd"].values.tolist())
    all_pred.extend(y_pred.tolist())

    # ===== Baselines (para no engañarse) =====
    # Probabilidad constante: p = base-rate del TRAIN
    p0 = float(np.clip(y_train.mean(), 1e-6, 1 - 1e-6))
    baseline_loglosses.append(_binary_logloss(y_test.values, np.full_like(proba, p0)))
    baseline_briers.append(brier_score_loss(y_test, np.full_like(proba, p0)))

    # ===== Métricas del modelo =====
    balanced_accs.append(float(balanced_accuracy_score(y_test, y_pred)))
    mccs.append(float(matthews_corrcoef(y_test, y_pred)))
    precisions.append(float(precision_score(y_test, y_pred, zero_division=0)))
    recalls.append(float(recall_score(y_test, y_pred, zero_division=0)))
    recalls_0.append(float(recall_score(y_test, y_pred, pos_label=0, zero_division=0)))
    f1s.append(float(f1_score(y_test, y_pred, zero_division=0)))
    accuracies.append(float(accuracy_score(y_test, y_pred)))
    precision_at_top20s.append(_precision_at_k(y_test, proba, top_frac=0.2))
    lift_top20s.append(_lift_at_k(y_test, proba, top_frac=0.2))

    if len(np.unique(y_test)) > 1:
        aucs.append(roc_auc_score(y_test, proba))
        ap_scores.append(average_precision_score(y_test, proba))
    else:
        aucs.append(np.nan)  # AUC no definido si solo hay una clase en el tramo
        ap_scores.append(np.nan)

    loglosses.append(_binary_logloss(y_test.values, proba))
    briers.append(brier_score_loss(y_test, proba))

    # ===== avanzar con embargo =====
    start = test_end + embargo

# Scorecards (sin prints) para memoria
wf_metrics_scorecard = pd.DataFrame(
    {
        "Valor": [
            float(np.nanmean(aucs)),
            float(np.nanmean(ap_scores)),
            float(np.nanmean(loglosses)),
            float(np.nanmean(briers)),
            float(np.nanmean(balanced_accs)),
            float(np.nanmean(mccs)),
            float(np.nanmean(precisions)),
            float(np.nanmean(recalls)),
            float(np.nanmean(recalls_0)),
            float(np.nanmean(f1s)),
            float(np.nanmean(accuracies)),
            float(np.nanmean(precision_at_top20s)),
            float(np.nanmean(lift_top20s)),
        ]
    },
    index=[
        "ROC-AUC (mean)",
        "PR-AUC / AvgPrecision (mean)",
        "LogLoss (mean)",
        "Brier score (mean)",
        "Balanced Accuracy (mean)",
        "MCC (mean)",
        "Precision (mean)",
        "Recall (mean)",
        "Recall clase 0 (mean)",
        "F1 (mean)",
        "Accuracy (mean)",
        "Precision@top20% (mean)",
        "Lift@top20% (mean)",
    ],
)
_save_table_figure(
    wf_metrics_scorecard,
    out_path=BASE_DIR / "walk_forward_metrics_scorecard.png",
    title=f"Walk-Forward — Métricas promedio (horizonte {HORIZON}m)",
)

wf_baselines_table = pd.DataFrame(
    {
        "LogLoss": [float(np.nanmean(loglosses)), float(np.nanmean(baseline_loglosses))],
        "Brier": [float(np.nanmean(briers)), float(np.nanmean(baseline_briers))],
    },
    index=["Modelo (WF)", "Baseline"],
)
_save_table_figure(
    wf_baselines_table,
    out_path=BASE_DIR / "walk_forward_baselines_scorecard.png",
    title="Walk-Forward — Modelo vs Baselines",
)

# Dataset WF para plots
wf_df = pd.DataFrame({
    "date": pd.to_datetime(all_dates),
    "proba_up": all_proba,   # prob. de clase positiva (sube)
    "actual": all_actuals,
    "pred": all_pred,
    "close_t": all_close,
    "close_t_plus_h": all_close_fwd,
})
wf_df = (
    wf_df.sort_values("date")
    .drop_duplicates(subset="date", keep="last")
    .reset_index(drop=True)
)

# ==============================
# WalkForward — Risk / Turnover / Calibration (más duro)
# ==============================
# Nota: estas métricas se calculan sobre retornos 1M realizados (close_t -> close_{t+1})
# para que Sharpe/vol/drawdown sean comparables y no sufran del solape del forward-return.
wf_ret_df = wf_df[["date", "close_t", "proba_up", "pred", "actual"]].copy()
wf_ret_df["date"] = pd.to_datetime(wf_ret_df["date"])
wf_ret_df = wf_ret_df.sort_values("date").replace([np.inf, -np.inf], np.nan).dropna(subset=["close_t", "proba_up"])

if len(wf_ret_df) >= 3:
    wf_ret_df["ret_1m"] = wf_ret_df["close_t"].astype(float).shift(-1) / wf_ret_df["close_t"].astype(float) - 1.0
    wf_ret_df = wf_ret_df.iloc[:-1].copy()  # última fila no tiene ret_1m

    exposure_proba = np.clip(wf_ret_df["proba_up"].astype(float).to_numpy(), 0.0, 1.0)
    exposure_pred = (wf_ret_df["pred"].astype(float).to_numpy() > 0.5).astype(float)
    ret_1m = wf_ret_df["ret_1m"].astype(float).to_numpy()

    wf_risk_bh = compute_return_risk_metrics(ret_1m, periods_per_year=12.0)
    wf_risk_proba = compute_return_risk_metrics(exposure_proba * ret_1m, periods_per_year=12.0)
    wf_risk_pred = compute_return_risk_metrics(exposure_pred * ret_1m, periods_per_year=12.0)

    wf_risk_table = pd.DataFrame(
        {
            "Buy&Hold": wf_risk_bh,
            "Estrategia (exposure=P)": wf_risk_proba,
            "Estrategia (pred 0/1)": wf_risk_pred,
        }
    )
    wf_risk_table = wf_risk_table.reindex(
        [
            "n",
            "total_return",
            "cagr",
            "ann_vol",
            "sharpe",
            "max_drawdown",
            "mean_ret",
            "std_ret",
        ]
    )
    _save_table_figure(
        wf_risk_table,
        out_path=BASE_DIR / "walk_forward_risk_metrics_1m.png",
        title=f"Walk-Forward — Riesgo / Sharpe (retornos 1M, horizonte target {HORIZON}m)",
    )

    # Turnover / estabilidad de señal
    wf_signal_series = wf_df.set_index(pd.to_datetime(wf_df["date"]))["pred"]
    wf_stab = compute_signal_stability_metrics(wf_signal_series)
    wf_turn = compute_exposure_turnover(wf_df.set_index(pd.to_datetime(wf_df["date"]))["proba_up"])

    wf_turnover_table = pd.DataFrame(
        {
            "Valor": [
                wf_stab.get("n", np.nan),
                wf_stab.get("pct_long", np.nan),
                wf_stab.get("change_pct", np.nan),
                wf_stab.get("avg_hold_long", np.nan),
                wf_stab.get("avg_hold_flat", np.nan),
                wf_turn.get("mean_abs_change", np.nan),
                wf_turn.get("median_abs_change", np.nan),
            ]
        },
        index=[
            "n (meses)",
            "% tiempo long (pred=1)",
            "% cambios de señal (pred)",
            "Duración media long (meses)",
            "Duración media flat (meses)",
            "Turnover exposure (mean |ΔP|)",
            "Turnover exposure (median |ΔP|)",
        ],
    )
    _save_table_figure(
        wf_turnover_table,
        out_path=BASE_DIR / "walk_forward_turnover_signal_stability.png",
        title=f"Walk-Forward — Turnover / Estabilidad de señal ({HORIZON}m)",
    )

    # Calibration más dura: ECE + descomposición de Brier
    wf_y_true = wf_df["actual"].astype(int).to_numpy()
    wf_y_proba = np.clip(wf_df["proba_up"].astype(float).to_numpy(), 1e-9, 1.0 - 1e-9)
    wf_base = float(np.mean(wf_y_true)) if len(wf_y_true) else float("nan")

    wf_ece = expected_calibration_error(wf_y_true, wf_y_proba, n_bins=10, strategy="quantile")
    wf_bd = brier_decomposition(wf_y_true, wf_y_proba, n_bins=10, strategy="quantile")

    wf_y_proba_base = np.full_like(wf_y_proba, float(np.clip(wf_base, 1e-9, 1.0 - 1e-9)))
    wf_ece_base = expected_calibration_error(wf_y_true, wf_y_proba_base, n_bins=10, strategy="quantile")
    wf_bd_base = brier_decomposition(wf_y_true, wf_y_proba_base, n_bins=10, strategy="quantile")

    wf_calib_hard_table = pd.DataFrame(
        {
            "Modelo (WF)": [
                wf_ece,
                float(brier_score_loss(wf_y_true, wf_y_proba)),
                wf_bd.get("reliability", np.nan),
                wf_bd.get("resolution", np.nan),
                wf_bd.get("uncertainty", np.nan),
                wf_bd.get("brier_decomp", np.nan),
                wf_bd.get("n_bins_eff", np.nan),
            ],
            "Baseline (p const)": [
                wf_ece_base,
                float(brier_score_loss(wf_y_true, wf_y_proba_base)),
                wf_bd_base.get("reliability", np.nan),
                wf_bd_base.get("resolution", np.nan),
                wf_bd_base.get("uncertainty", np.nan),
                wf_bd_base.get("brier_decomp", np.nan),
                wf_bd_base.get("n_bins_eff", np.nan),
            ],
        },
        index=[
            "ECE (quantile bins)",
            "Brier",
            "Brier: Reliability",
            "Brier: Resolution",
            "Brier: Uncertainty",
            "Brier: Decomposition total",
            "Bins efectivos",
        ],
    )
    _save_table_figure(
        wf_calib_hard_table,
        out_path=BASE_DIR / "walk_forward_calibration_hard_metrics.png",
        title=f"Walk-Forward — Calibration dura (ECE + Brier decomposition, {HORIZON}m)",
    )

# Señal suavizada sobre probabilidad
wf_df["signal_raw"] = wf_df["proba_up"]
wf_df["signal"] = wf_df["proba_up"] - 0.5

wf_top20_precision = _precision_at_k(wf_df["actual"], wf_df["proba_up"].to_numpy(), top_frac=0.2)
wf_top20_lift = _lift_at_k(wf_df["actual"], wf_df["proba_up"].to_numpy(), top_frac=0.2)
wf_recall_0 = float(recall_score(wf_df["actual"], wf_df["pred"], pos_label=0, zero_division=0))
wf_base_rate = float(wf_df["actual"].mean())
wf_threshold_cm = _confusion_matrix_by_thresholds(wf_df["actual"], wf_df["proba_up"].to_numpy())

wf_ranking_metrics = pd.DataFrame(
    {
        "Valor": [
            wf_base_rate,
            wf_top20_precision,
            wf_top20_lift,
            wf_recall_0,
        ]
    },
    index=[
        "Base rate clase 1",
        "Precision@top20%",
        "Lift@top20%",
        "Recall clase 0",
    ],
)
_save_table_figure(
    wf_ranking_metrics,
    out_path=BASE_DIR / "walk_forward_ranking_metrics.png",
    title=f"Walk-Forward — Métricas para señal top 20% ({HORIZON}m)",
)
_save_table_figure(
    wf_threshold_cm,
    out_path=BASE_DIR / "walk_forward_confusion_matrix_thresholds.png",
    title=f"Walk-Forward — Confusion matrix por threshold ({HORIZON}m)",
)

plot_confusion_matrix_heatmap(
    wf_df["actual"],
    wf_df["pred"].to_numpy(),
    out_path=BASE_DIR / "walk_forward_confusion_matrix_heatmap.png",
    title=f"Walk-Forward — Matriz de confusión (pred optimizado, {HORIZON}m)",
)

print("\n[WalkForward ranking metrics]\n", wf_ranking_metrics)
print("\n[WalkForward confusion matrix por threshold]\n", wf_threshold_cm)

# ── Gráfico Walk-Forward: probabilidad vs clase real ──
plot_classification_timeline(
    wf_df,
    out_path=BASE_DIR / "walk_forward_classification.png",
    title=f"Walk-Forward — Probabilidades vs clase real ({HORIZON}m)",
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
)

plot_metrics_by_proba_bin(
    wf_df,
    out_path=BASE_DIR / "walk_forward_metrics_by_decile.png",
    title=f"Walk-Forward — Calibration por decil (P(sube) vs P(real=1), {HORIZON}m)",
    n_bins=10,
)

plot_cumulative_gains_wf(
    wf_df,
    out_path=BASE_DIR / "walk_forward_cumulative_gains.png",
    title=f"Walk-Forward — Cumulative gains / lift ({HORIZON}m)",
)

plot_rolling_logloss_wf(
    wf_df,
    out_path=BASE_DIR / "walk_forward_rolling_accuracy_36m.png",
    title=f"Walk-Forward — Rolling LogLoss (36m, horizonte {HORIZON}m)",
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


# ── Evaluación por deciles (all_proba/all_actuals) ──
df_eval = pd.DataFrame({
    "proba": all_proba,
    "y": all_actuals,
})
df_eval["bin"] = pd.qcut(df_eval["proba"], 10, duplicates="drop")
print("\n[Deciles eval] mean(y) por bin de proba:")
print(df_eval.groupby("bin")["y"].mean())


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

# ==============================
# ROI: Buy&Hold DCA vs Señal (2x cuando pred=1, 0x cuando pred=0)
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
contrib_signal = monthly_amount * float(signal_multiplier) * (pred_aligned.astype(int) == 1).astype(float)



bh_curve = simulate_monthly_dca_roi(prices_eval, contrib_bh)
sig_curve = simulate_monthly_dca_roi(prices_eval, contrib_signal)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(bh_curve.index, bh_curve["roi_pct"], label=f"Buy&Hold DCA (x={monthly_amount:g}/mes)", color="tab:blue")
ax.plot(
    sig_curve.index,
    sig_curve["roi_pct"],
    label=f"Señal (clase 1: {signal_multiplier:g}x, clase 0: 0x)",
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
print("ROI final Señal (clase) (%):", float(sig_curve["roi_pct"].dropna().iloc[-1]))
# print("Total invertido Buy&Hold:", float(bh_curve["invested"].dropna().iloc[-1]))
# print("Total invertido Señal:", float(sig_curve["invested"].dropna().iloc[-1]))






















# exit(0)

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

    if len(np.unique(y_tr)) < 2:
        # Mismo fallback que en walk-forward: sin ambas clases, no hay modelo XGB válido.
        best_t_roll, _best_f1_roll = 0.5, float("nan")
        p_const = float(np.clip(y_tr.mean(), 1e-6, 1.0 - 1e-6))
        roll_proba = np.full(len(X_roll), p_const, dtype=float)
        roll_pred = (roll_proba >= best_t_roll).astype(int)
    else:
        # ===== Hiperparámetros =====
        fixed_params = dict(fixed_params_base)
        manual_params = dict(manual_params_base)

        if not DO_RANDOM_SEARCH:
            best_params = manual_params
        else:
            # Si hubo tuning walk-forward, reutiliza esos params; si no, cae al manual.
            best_params = best_params_global or manual_params

        model_params = dict(fixed_params)
        model_params.update(best_params)

        model_roll = XGBClassifier(**model_params)

        if len(np.unique(y_es)) < 2:
            model_roll.set_params(early_stopping_rounds=None)
            model_roll.fit(
                X_tr,
                y_tr,
                verbose=False,
            )
        else:
            model_roll.fit(
                X_tr,
                y_tr,
                eval_set=[(X_es, y_es)],
                verbose=False,
            )

        # ===== Threshold óptimo (rollout) =====
        if len(X_score) < 1:
            best_t_roll, _best_f1_roll = 0.5, float("nan")
        else:
            score_proba = model_roll.predict_proba(X_score)[:, 1]
            best_t_roll, _best_f1_roll = _best_threshold_by_f1(y_score, score_proba)

        # ===== Predicción (rollout) =====
        roll_proba = model_roll.predict_proba(X_roll)[:, 1]
        roll_pred = (roll_proba >= best_t_roll).astype(int)

    # ===== Métricas rollout =====
    roll_logloss = float(_binary_logloss(y_roll.values, roll_proba))
    roll_brier = float(brier_score_loss(y_roll, roll_proba))
    roll_base_rate = float(y_roll.mean())
    roll_precision_top20 = _precision_at_k(y_roll, roll_proba, top_frac=0.2)
    roll_lift_top20 = _lift_at_k(y_roll, roll_proba, top_frac=0.2)
    roll_recall_0 = float(recall_score(y_roll, roll_pred, pos_label=0, zero_division=0))
    roll_threshold_cm = _confusion_matrix_by_thresholds(y_roll, roll_proba)
    if len(np.unique(y_roll)) > 1:
        roll_auc = float(roc_auc_score(y_roll, roll_proba))
        roll_ap = float(average_precision_score(y_roll, roll_proba))
    else:
        roll_auc = float("nan")
        roll_ap = float("nan")

    # Baselines (comparación justa, usando solo train): prob. constante
    p0 = float(np.clip(y_train_full.mean(), 1e-6, 1 - 1e-6))
    baseline_ll = float(_binary_logloss(y_roll.values, np.full_like(roll_proba, p0)))
    baseline_br = float(brier_score_loss(y_roll, np.full_like(roll_proba, p0)))

    print("\n[FinalRollout] Ventana:", str(rollout_df.index.min().date()), "->", str(rollout_df.index.max().date()))
    print("[FinalRollout] Train hasta:", str(train_df.index.max().date()), f"(gap={FINAL_GAP_MONTHS}m)")
    print("[FinalRollout] ROC-AUC:", roll_auc)
    print("[FinalRollout] PR-AUC:", roll_ap)
    print("[FinalRollout] LogLoss:", roll_logloss)
    print("[FinalRollout] Brier:", roll_brier)
    print("[FinalRollout] Base rate clase 1:", roll_base_rate)
    print("[FinalRollout] Precision@top20%:", roll_precision_top20)
    print("[FinalRollout] Lift@top20%:", roll_lift_top20)
    print("[FinalRollout] Recall clase 0:", roll_recall_0)
    print("[FinalRollout] Baseline LogLoss (p const train):", baseline_ll)
    print("[FinalRollout] Baseline Brier (p const train):", baseline_br)
    print("\n[FinalRollout confusion matrix por threshold]\n", roll_threshold_cm)

    roll_ranking_metrics = pd.DataFrame(
        {
            "Valor": [
                roll_base_rate,
                roll_precision_top20,
                roll_lift_top20,
                roll_recall_0,
            ]
        },
        index=[
            "Base rate clase 1",
            "Precision@top20%",
            "Lift@top20%",
            "Recall clase 0",
        ],
    )
    _save_table_figure(
        roll_ranking_metrics,
        out_path=BASE_DIR / "final_rollout_ranking_metrics.png",
        title=f"Final Roll-out — Métricas para señal top 20% ({HORIZON}m)",
    )
    _save_table_figure(
        roll_threshold_cm,
        out_path=BASE_DIR / "final_rollout_confusion_matrix_thresholds.png",
        title=f"Final Roll-out — Confusion matrix por threshold ({HORIZON}m)",
    )

    roll_plot_df = pd.DataFrame(
        {
            "date": pd.to_datetime(rollout_df.index),
            "proba_up": roll_proba,
            "actual": y_roll.values,
            "pred": roll_pred,
            "close_t": rollout_df["Close"].values,
            "close_t_plus_h": rollout_df["close_fwd"].values,
        }
    ).sort_values("date")

    # ==============================
    # Final Roll-out — Risk / Turnover / Calibration dura
    # ==============================
    roll_ret_df = roll_plot_df[["date", "close_t", "proba_up", "pred", "actual"]].copy()
    roll_ret_df["date"] = pd.to_datetime(roll_ret_df["date"])
    roll_ret_df = roll_ret_df.sort_values("date").replace([np.inf, -np.inf], np.nan).dropna(subset=["close_t", "proba_up"])

    if len(roll_ret_df) >= 3:
        roll_ret_df["ret_1m"] = roll_ret_df["close_t"].astype(float).shift(-1) / roll_ret_df["close_t"].astype(float) - 1.0
        roll_ret_df = roll_ret_df.iloc[:-1].copy()

        roll_exposure_proba = np.clip(roll_ret_df["proba_up"].astype(float).to_numpy(), 0.0, 1.0)
        roll_exposure_pred = (roll_ret_df["pred"].astype(float).to_numpy() > 0.5).astype(float)
        roll_ret_1m = roll_ret_df["ret_1m"].astype(float).to_numpy()

        roll_risk_bh = compute_return_risk_metrics(roll_ret_1m, periods_per_year=12.0)
        roll_risk_proba = compute_return_risk_metrics(roll_exposure_proba * roll_ret_1m, periods_per_year=12.0)
        roll_risk_pred = compute_return_risk_metrics(roll_exposure_pred * roll_ret_1m, periods_per_year=12.0)

        roll_risk_table = pd.DataFrame(
            {
                "Buy&Hold": roll_risk_bh,
                "Estrategia (exposure=P)": roll_risk_proba,
                "Estrategia (pred 0/1)": roll_risk_pred,
            }
        ).reindex(
            [
                "n",
                "total_return",
                "cagr",
                "ann_vol",
                "sharpe",
                "max_drawdown",
                "mean_ret",
                "std_ret",
            ]
        )
        _save_table_figure(
            roll_risk_table,
            out_path=BASE_DIR / "final_rollout_risk_metrics_1m.png",
            title=f"Final Roll-out — Riesgo / Sharpe (retornos 1M, target {HORIZON}m)",
        )

        roll_stab = compute_signal_stability_metrics(roll_plot_df.set_index(pd.to_datetime(roll_plot_df["date"]))["pred"])
        roll_turn = compute_exposure_turnover(roll_plot_df.set_index(pd.to_datetime(roll_plot_df["date"]))["proba_up"])
        roll_turnover_table = pd.DataFrame(
            {
                "Valor": [
                    roll_stab.get("n", np.nan),
                    roll_stab.get("pct_long", np.nan),
                    roll_stab.get("change_pct", np.nan),
                    roll_stab.get("avg_hold_long", np.nan),
                    roll_stab.get("avg_hold_flat", np.nan),
                    roll_turn.get("mean_abs_change", np.nan),
                    roll_turn.get("median_abs_change", np.nan),
                ]
            },
            index=[
                "n (meses)",
                "% tiempo long (pred=1)",
                "% cambios de señal (pred)",
                "Duración media long (meses)",
                "Duración media flat (meses)",
                "Turnover exposure (mean |ΔP|)",
                "Turnover exposure (median |ΔP|)",
            ],
        )
        _save_table_figure(
            roll_turnover_table,
            out_path=BASE_DIR / "final_rollout_turnover_signal_stability.png",
            title=f"Final Roll-out — Turnover / Estabilidad de señal ({HORIZON}m)",
        )

        roll_y_true = roll_plot_df["actual"].astype(int).to_numpy()
        roll_y_proba = np.clip(roll_plot_df["proba_up"].astype(float).to_numpy(), 1e-9, 1.0 - 1e-9)
        roll_base = float(np.mean(roll_y_true)) if len(roll_y_true) else float("nan")
        roll_ece = expected_calibration_error(roll_y_true, roll_y_proba, n_bins=10, strategy="quantile")
        roll_bd = brier_decomposition(roll_y_true, roll_y_proba, n_bins=10, strategy="quantile")

        roll_y_proba_base = np.full_like(roll_y_proba, float(np.clip(roll_base, 1e-9, 1.0 - 1e-9)))
        roll_ece_base = expected_calibration_error(roll_y_true, roll_y_proba_base, n_bins=10, strategy="quantile")
        roll_bd_base = brier_decomposition(roll_y_true, roll_y_proba_base, n_bins=10, strategy="quantile")

        roll_calib_hard_table = pd.DataFrame(
            {
                "Modelo (rollout)": [
                    roll_ece,
                    float(brier_score_loss(roll_y_true, roll_y_proba)),
                    roll_bd.get("reliability", np.nan),
                    roll_bd.get("resolution", np.nan),
                    roll_bd.get("uncertainty", np.nan),
                    roll_bd.get("brier_decomp", np.nan),
                    roll_bd.get("n_bins_eff", np.nan),
                ],
                "Baseline (p const)": [
                    roll_ece_base,
                    float(brier_score_loss(roll_y_true, roll_y_proba_base)),
                    roll_bd_base.get("reliability", np.nan),
                    roll_bd_base.get("resolution", np.nan),
                    roll_bd_base.get("uncertainty", np.nan),
                    roll_bd_base.get("brier_decomp", np.nan),
                    roll_bd_base.get("n_bins_eff", np.nan),
                ],
            },
            index=[
                "ECE (quantile bins)",
                "Brier",
                "Brier: Reliability",
                "Brier: Resolution",
                "Brier: Uncertainty",
                "Brier: Decomposition total",
                "Bins efectivos",
            ],
        )
        _save_table_figure(
            roll_calib_hard_table,
            out_path=BASE_DIR / "final_rollout_calibration_hard_metrics.png",
            title=f"Final Roll-out — Calibration dura (ECE + Brier decomposition, {HORIZON}m)",
        )

    plot_classification_timeline(
        roll_plot_df,
        out_path=BASE_DIR / "final_rollout_classification.png",
        title=f"Final Roll-out — Probabilidades vs clase real ({HORIZON}m)",
        year_locator=1,
    )

    # ==============================
    # (1) Análisis temporal interno (roll) + split early/late
    # ==============================
    plot_rolling_logloss_wf(
        roll_plot_df,
        out_path=BASE_DIR / "final_rollout_rolling_logloss_36m.png",
        title=f"Final Roll-out — Rolling LogLoss (36m, horizonte {HORIZON}m)",
        window=36,
    )
    plot_rolling_brier_wf(
        roll_plot_df,
        out_path=BASE_DIR / "final_rollout_rolling_brier_36m.png",
        title=f"Final Roll-out — Rolling Brier (36m, horizonte {HORIZON}m)",
        window=36,
    )

    roll_split_metrics = None
    if len(roll_plot_df) >= 10:
        split_idx = int(len(roll_plot_df) // 2)
        split_date = pd.to_datetime(roll_plot_df["date"].iloc[split_idx])

        def _block_metrics(name: str, g: pd.DataFrame) -> Dict[str, float]:
            y_true = g["actual"].astype(int).to_numpy()
            y_proba = np.clip(g["proba_up"].astype(float).to_numpy(), 1e-9, 1.0 - 1e-9)
            y_pred = (y_proba >= float(best_t_roll)).astype(int)
            n = int(len(g))

            out_m: Dict[str, float] = {
                "n": float(n),
                "base_rate": float(np.mean(y_true)) if n else np.nan,
                "logloss": float(_binary_logloss(y_true, y_proba)) if n else np.nan,
                "brier": float(brier_score_loss(y_true, y_proba)) if n else np.nan,
                "precision_top20": float(_precision_at_k(pd.Series(y_true), y_proba, top_frac=0.2)) if n else np.nan,
                "lift_top20": float(_lift_at_k(pd.Series(y_true), y_proba, top_frac=0.2)) if n else np.nan,
                "recall_0": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)) if n else np.nan,
                "accuracy": float(accuracy_score(y_true, y_pred)) if n else np.nan,
                "f1": float(f1_score(y_true, y_pred, zero_division=0)) if n else np.nan,
            }
            if n > 2 and len(np.unique(y_true)) > 1:
                out_m["auc"] = float(roc_auc_score(y_true, y_proba))
                out_m["ap"] = float(average_precision_score(y_true, y_proba))
            else:
                out_m["auc"] = np.nan
                out_m["ap"] = np.nan
            return out_m

        g_early = roll_plot_df.loc[roll_plot_df["date"] <= split_date]
        g_late = roll_plot_df.loc[roll_plot_df["date"] > split_date]
        m_all = _block_metrics("all", roll_plot_df)
        m_early = _block_metrics("early", g_early)
        m_late = _block_metrics("late", g_late)

        roll_split_metrics = pd.DataFrame(
            {
                "All": m_all,
                f"Early (<= {split_date.date()})": m_early,
                f"Late (> {split_date.date()})": m_late,
            }
        )
        # Orden de lectura: tamaño/base-rate y luego probabilísticas.
        row_order = [
            "n",
            "base_rate",
            "auc",
            "ap",
            "logloss",
            "brier",
            "precision_top20",
            "lift_top20",
            "recall_0",
            "accuracy",
            "f1",
        ]
        roll_split_metrics = roll_split_metrics.reindex(row_order)

        _save_table_figure(
            roll_split_metrics,
            out_path=BASE_DIR / "final_rollout_subperiod_metrics.png",
            title=f"Final Roll-out — Métricas por subperiodo (threshold={best_t_roll:.3f})",
        )
        print("\n[FinalRollout temporal split metrics]\n", roll_split_metrics)

    # ==============================
    # (2) Calibración (curve + deciles)
    # ==============================
    plot_calibration_curve_wf(
        roll_plot_df,
        out_path=BASE_DIR / "final_rollout_calibration.png",
        title=f"Final Roll-out — Calibration curve ({HORIZON}m)",
        n_bins=10,
    )
    plot_metrics_by_proba_bin(
        roll_plot_df,
        out_path=BASE_DIR / "final_rollout_metrics_by_decile.png",
        title=f"Final Roll-out — Calibration por decil (P(sube) vs P(real=1), {HORIZON}m)",
        n_bins=10,
    )
    roll_calib_deciles = compute_calibration_deciles_table(roll_plot_df, n_bins=10)
    if roll_calib_deciles is not None and not roll_calib_deciles.empty:
        _save_table_figure(
            roll_calib_deciles,
            out_path=BASE_DIR / "final_rollout_calibration_deciles_table.png",
            title=f"Final Roll-out — Tabla de calibración por decil ({HORIZON}m)",
        )
        print("\n[FinalRollout calibration deciles table]\n", roll_calib_deciles)

    # ==============================
    # (3) Análisis de régimen (si existe high_inflation)
    # ==============================
    if "high_inflation" in df.columns:
        regime_series_roll = df["high_inflation"].astype(float).map({1.0: "high_inflation", 0.0: "low_inflation"})
        roll_regime_df = plot_regime_performance_wf(
            roll_plot_df,
            out_path=BASE_DIR / "final_rollout_regime_performance.png",
            title=f"Final Roll-out — Performance por régimen (high/low inflation)",
            regime_series=regime_series_roll,
        )
        if roll_regime_df is not None and not roll_regime_df.empty:
            _save_table_figure(
                roll_regime_df.set_index("regime"),
                out_path=BASE_DIR / "final_rollout_regime_performance_table.png",
                title="Final Roll-out — Tabla performance por régimen",
            )
            print("\n[FinalRollout regime performance]\n", roll_regime_df)
    else:
        print("[FinalRollout] Nota: no existe columna 'high_inflation' → sin análisis de régimen.")

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
    contrib_signal_roll = monthly_amount * float(signal_multiplier) * (pred_aligned_roll.astype(int) == 1).astype(float)

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
        label=f"Señal (clase 1: {signal_multiplier:g}x, clase 0: 0x)",
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
    print("[FinalRollout ROI] ROI final Señal (clase) (%):", float(sig_curve_roll["roi_pct"].dropna().iloc[-1]))
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

final_model_params = dict(final_fixed_params)
final_model_params.update(final_best_params)

final_model = XGBClassifier(**final_model_params)

final_model.fit(X, y)

last_X = X.iloc[[-1]]
final_proba = float(final_model.predict_proba(last_X)[:, 1][0])

print("Última fecha:", X.index[-1])
print(f"P(sube) en {HORIZON} meses:", final_proba)

# ===== Métricas (in-sample, sobre todo el dataset) =====
final_proba_all = final_model.predict_proba(X)[:, 1]
final_logloss = float(_binary_logloss(y.values, final_proba_all))
final_brier = float(brier_score_loss(y, final_proba_all))
final_precision_top20 = _precision_at_k(y, final_proba_all, top_frac=0.2)
if len(np.unique(y)) > 1:
    final_auc = float(roc_auc_score(y, final_proba_all))
else:
    final_auc = float("nan")

# ===== Incertidumbre (histórico de probabilidades in-sample) =====
# Nota: esto no es un intervalo de confianza estadístico; es una medida
# de rareza relativa vs el histórico de P(sube) generado por el propio modelo.
hist_mean = float(np.nanmean(final_proba_all))
hist_std = float(np.nanstd(final_proba_all, ddof=0))
final_proba_percentile = float(np.mean(final_proba_all <= final_proba) * 100.0)
final_proba_z = float((final_proba - hist_mean) / hist_std) if hist_std > 0 else float("nan")

print("\n[FinalModel] NOTE: métricas in-sample (solo diagnóstico, no performance real)")
print("[FinalModel] ROC-AUC:", final_auc)
print("[FinalModel] LogLoss:", final_logloss)
print("[FinalModel] Brier:", final_brier)
print("[FinalModel] Precision@top20%:", final_precision_top20)
print("[FinalModel] P(sube) percentil histórico (%):", final_proba_percentile)
print("[FinalModel] P(sube) z-score vs histórico:", final_proba_z)
print("[FinalModel vs WF] LogLoss diff:", final_logloss - np.nanmean(loglosses))
print("[FinalModel vs WF] AUC diff:", final_auc - np.nanmean(aucs))

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
