import datetime as dt
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "backtest"
ROI_HISTORY_DIR = OUT_DIR / "roi_history"

START_DATE = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
END_DATE = dt.datetime.now(dt.timezone.utc)
INTERVAL = "1wk"

CONTRIBUTION_AMOUNT = 100.0
DCA_TOTAL_CASH = 1000.0
RSI_WINDOW = 14
MOVING_AVERAGES = (30, 50, 200)
STRATEGY_ORDER = tuple(
    ["RSI & MA MODE (MA: {0})".format(value) for value in MOVING_AVERAGES]
    + ["MA MODE (MA: {0})".format(value) for value in MOVING_AVERAGES]
    + ["RSI MODE", "DCA MODE", "Value Averaging Modified"]
)

# Lista completa del backtest original. Se ignora el override final de prueba: tickets = ["^IBEX"].
TICKERS = [
    "^GSPC",
    "^IXIC",
    "^FTSE",
    "^N225",
    "^HSI",
    "^BSESN",
    "^GDAXI",
    "^GSPTSE",
    "^IBEX",
    "FTSEMIB.MI",
    "^KS11",
    "^RUT",
    "CL=F",
    "GC=F",
    "SI=F",
    "PFE",
    "IDR.MC",
    "NHH.MC",
    "ACX.MC",
    "REPYY",
    "SAN",
    "BKT.MC",
    "GEST.MC",
    "ITX.MC",
    "GE",
    "X",
    "BB",
    "GM",
    "EGRNF",
]


def unix_seconds(value: dt.datetime) -> int:
    return int(value.timestamp())


def sanitize_ticker(ticker: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_")
    return cleaned or "ticker"


def fetch_weekly_close(ticker: str) -> Tuple[pd.Series, str]:
    encoded_ticker = urllib.parse.quote(ticker, safe="")
    params = urllib.parse.urlencode(
        {
            "period1": unix_seconds(START_DATE),
            "period2": unix_seconds(END_DATE),
            "interval": INTERVAL,
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    url = "https://query1.finance.yahoo.com/v8/finance/chart/{0}?{1}".format(
        encoded_ticker,
        params,
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(chart["error"])

    results = chart.get("result") or []
    if not results:
        raise RuntimeError("Yahoo no devolvio datos")

    result = results[0]
    meta = result.get("meta", {})
    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    close_values = quote.get("close") or []

    if not timestamps or not close_values:
        raise RuntimeError("Yahoo devolvio una serie vacia")

    dates = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None)
    prices = pd.Series(close_values, index=dates, name="Close")
    prices = pd.to_numeric(prices, errors="coerce").dropna()
    prices = prices[prices > 0].sort_index()

    if prices.empty:
        raise RuntimeError("No hay cierres validos")

    return prices, meta.get("longName") or meta.get("shortName") or ticker


def nz(values: pd.Series) -> pd.Series:
    return values.fillna(0.0)


def calculate_rsi(prices: pd.Series, window: int = RSI_WINDOW, adjust: bool = False) -> pd.Series:
    delta = prices.diff(1).dropna()
    gains = delta.copy()
    losses = delta.copy()

    gains[gains < 0] = 0.0
    losses[losses > 0] = 0.0

    avg_gain = gains.ewm(com=window - 1, adjust=adjust).mean()
    avg_loss = losses.abs().ewm(com=window - 1, adjust=adjust).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)

    return rsi.reindex(prices.index)


def simulate_contributions(prices: pd.Series, contributions: pd.Series) -> pd.DataFrame:
    prices = prices.astype(float)
    contribution = contributions.astype(float).reindex(prices.index).fillna(0.0)
    shares_bought = contribution.div(prices).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    shares = shares_bought.cumsum()
    invested = contribution.cumsum()
    value = shares * prices
    roi = pd.Series(np.nan, index=prices.index, dtype=float)
    has_investment = invested > 0
    roi.loc[has_investment] = (
        (value.loc[has_investment] - invested.loc[has_investment])
        / invested.loc[has_investment]
        * 100.0
    )

    return pd.DataFrame(
        {
            "price": prices,
            "contribution": contribution,
            "invested": invested,
            "shares": shares,
            "value": value,
            "roi_pct": roi,
        },
        index=prices.index,
    )


def simulate_dca_backtest(prices: pd.Series) -> pd.DataFrame:
    weeks = max(1, (END_DATE - START_DATE).days // 7)
    weekly_amount = DCA_TOTAL_CASH / float(weeks)
    invested = 0.0
    contributions = []

    for _date in prices.index:
        if invested <= DCA_TOTAL_CASH:
            contribution = weekly_amount
            invested += weekly_amount
        else:
            contribution = 0.0
        contributions.append(contribution)

    return simulate_contributions(prices, pd.Series(contributions, index=prices.index))


def simulate_signal_strategy(
    prices: pd.Series,
    signal: pd.Series,
    contribution_amount: float = CONTRIBUTION_AMOUNT,
) -> pd.DataFrame:
    aligned_signal = signal.reindex(prices.index).fillna(False)
    contributions = pd.Series(
        np.where(aligned_signal, float(contribution_amount), 0.0),
        index=prices.index,
    )
    return simulate_contributions(prices, contributions)


def simulate_value_averaging_modified(prices: pd.Series) -> pd.DataFrame:
    x = float(CONTRIBUTION_AMOUNT)
    rows = []
    shares = 0.0
    invested = 0.0

    for date, price in prices.items():
        portfolio_value_before = shares * float(price)
        target_value = x * float(len(rows) + 1)
        contribution = min(max(target_value - portfolio_value_before, x), 3.0 * x)

        shares += contribution / float(price)
        invested += contribution
        value = shares * float(price)
        roi_pct = (value - invested) / invested * 100.0 if invested > 0 else np.nan

        rows.append(
            {
                "date": date,
                "price": float(price),
                "target_value": target_value,
                "portfolio_value_before": portfolio_value_before,
                "contribution": contribution,
                "invested": invested,
                "shares": shares,
                "value": value,
                "roi_pct": roi_pct,
            }
        )

    return pd.DataFrame(rows).set_index("date")


def build_strategy_curves(prices: pd.Series) -> Dict[str, pd.DataFrame]:
    rsi = calculate_rsi(prices)
    rsi_signal = nz(rsi) < 30.0

    curves = {}

    for ma_window in MOVING_AVERAGES:
        moving_average = prices.rolling(int(ma_window)).mean()
        ma_signal = nz(moving_average) > prices
        curves["RSI & MA MODE (MA: {0})".format(ma_window)] = simulate_signal_strategy(
            prices,
            rsi_signal & ma_signal,
        )

    for ma_window in MOVING_AVERAGES:
        moving_average = prices.rolling(int(ma_window)).mean()
        ma_signal = nz(moving_average) > prices
        curves["MA MODE (MA: {0})".format(ma_window)] = simulate_signal_strategy(
            prices,
            ma_signal,
        )

    curves["RSI MODE"] = simulate_signal_strategy(prices, rsi_signal)
    curves["DCA MODE"] = simulate_dca_backtest(prices)
    curves["Value Averaging Modified"] = simulate_value_averaging_modified(prices)

    return curves


def summarize(ticker: str, company_name: str, strategy: str, curve: pd.DataFrame) -> dict:
    final = curve.iloc[-1]
    final_invested = float(final["invested"])
    final_value = float(final["value"])
    final_profit = final_value - final_invested
    final_roi_pct = float(final["roi_pct"]) if final_invested > 0 else 0.0

    return {
        "ticker": ticker,
        "company_name": company_name,
        "strategy": strategy,
        "start_date": curve.index.min().date().isoformat(),
        "end_date": curve.index.max().date().isoformat(),
        "final_invested": final_invested,
        "final_value": final_value,
        "final_profit": final_profit,
        "final_roi_pct": final_roi_pct,
        "gross_return_pct": (final_value / final_invested * 100.0) if final_invested > 0 else 0.0,
        "total_trades": int((curve["contribution"] > 0).sum()),
        "avg_contribution": float(curve["contribution"].mean()),
        "max_contribution": float(curve["contribution"].max()),
    }


def plot_ticker_roi_history(
    ticker: str,
    company_name: str,
    curves: Dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 7))
    for strategy in STRATEGY_ORDER:
        curve = curves.get(strategy)
        if curve is None:
            continue

        roi = curve["roi_pct"].dropna()
        if roi.empty:
            continue

        linewidth = 2.0 if strategy in {"DCA MODE", "Value Averaging Modified"} else 1.2
        linestyle = "--" if strategy == "DCA MODE" else "-"
        label = "{0} ({1:.1f}%)".format(strategy, float(roi.iloc[-1]))
        ax.plot(roi.index, roi, label=label, linewidth=linewidth, linestyle=linestyle)

    ax.axhline(0, color="grey", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title("{0} - {1}\nROI historico acumulado por estrategia".format(ticker, company_name))
    ax.set_xlabel("Fecha")
    ax.set_ylabel("ROI acumulado (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out_dir / "{0}_roi_history.png".format(sanitize_ticker(ticker)), dpi=140)
    plt.close(fig)


def save_matrices(summary: pd.DataFrame, out_dir: Path) -> None:
    roi_matrix = summary.pivot(index="strategy", columns="ticker", values="final_roi_pct")
    roi_matrix = roi_matrix.reindex(STRATEGY_ORDER)
    roi_matrix["Average"] = roi_matrix.mean(axis=1, skipna=True)
    roi_matrix.to_csv(out_dir / "strategy_roi_matrix.csv")

    trades_matrix = summary.pivot(index="strategy", columns="ticker", values="total_trades")
    trades_matrix = trades_matrix.reindex(STRATEGY_ORDER)
    trades_matrix["Average"] = trades_matrix.mean(axis=1, skipna=True)
    trades_matrix.to_csv(out_dir / "strategy_trades_matrix.csv")


def plot_average_roi(summary: pd.DataFrame, out_path: Path) -> None:
    average_roi = summary.groupby("strategy")["final_roi_pct"].mean().sort_values()

    fig_height = max(5, 0.45 * len(average_roi))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    average_roi.plot(kind="barh", ax=ax, color="tab:blue")
    ax.axvline(0, color="grey", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title("ROI promedio por estrategia")
    ax.set_xlabel("ROI final promedio (%)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def run_backtests() -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[dict] = []
    skipped_rows: List[dict] = []

    for ticker in TICKERS:
        try:
            prices, company_name = fetch_weekly_close(ticker)
            curves = build_strategy_curves(prices)
            plot_ticker_roi_history(ticker, company_name, curves, ROI_HISTORY_DIR)

            for strategy, curve in curves.items():
                summary_rows.append(summarize(ticker, company_name, strategy, curve))

            print("OK {0}: {1} registros".format(ticker, len(prices)))
        except Exception as exc:
            skipped_rows.append({"ticker": ticker, "error": str(exc)})
            print("SKIP {0}: {1}".format(ticker, exc))

    return pd.DataFrame(summary_rows), pd.DataFrame(skipped_rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ROI_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    summary, skipped = run_backtests()
    if summary.empty:
        raise RuntimeError("No se pudo calcular ningun ticker")

    summary_path = OUT_DIR / "strategy_summary_all_tickers.csv"
    skipped_path = OUT_DIR / "skipped_tickers.csv"
    plot_path = OUT_DIR / "average_roi_by_strategy.png"
    settings_path = OUT_DIR / "settings.csv"

    summary.to_csv(summary_path, index=False)
    skipped.to_csv(skipped_path, index=False)
    save_matrices(summary, OUT_DIR)
    plot_average_roi(summary, plot_path)

    pd.DataFrame(
        [
            {
                "start_date": START_DATE.date().isoformat(),
                "end_date": END_DATE.date().isoformat(),
                "interval": INTERVAL,
                "signal_contribution_amount": CONTRIBUTION_AMOUNT,
                "dca_total_cash": DCA_TOTAL_CASH,
                "rsi_window": RSI_WINDOW,
                "moving_averages": ",".join(str(value) for value in MOVING_AVERAGES),
                "tickers_requested": len(TICKERS),
                "tickers_ok": summary["ticker"].nunique(),
                "tickers_skipped": len(skipped),
            }
        ]
    ).to_csv(settings_path, index=False)

    print("\nResumen guardado en: {0}".format(summary_path))
    print("ROI por ticker guardado en: {0}".format(OUT_DIR / "strategy_roi_matrix.csv"))
    print("Grafico guardado en: {0}".format(plot_path))
    print("Graficos historicos por ticker guardados en: {0}".format(ROI_HISTORY_DIR))
    if not skipped.empty:
        print("Tickers omitidos guardados en: {0}".format(skipped_path))


if __name__ == "__main__":
    main()
