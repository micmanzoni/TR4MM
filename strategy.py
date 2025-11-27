import yfinance as yf
import pandas as pd
import numpy as np

MIN_SCORE_DEFAULT = 0.01
STOP_PCT = 0.03   # 3% sotto
TP_PCT   = 0.06   # 6% sopra

TICKERS_INDICI = ["^GSPC", "^NDX", "^DJI", "^RUT"]
TICKERS_MATERIE = ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F"]
TICKERS_USA = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "BRK-B", "JPM", "JNJ",
    "XOM", "UNH", "V", "PG"
]
TICKERS_ETF = [
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV",
    "XLE", "XLF", "XLK", "XLV", "ARKK"
]

def build_universe():
    ticker_categoria = {}
    for t in TICKERS_INDICI:
        ticker_categoria[t] = "Indice"
    for t in TICKERS_MATERIE:
        ticker_categoria[t] = "Materia Prima"
    for t in TICKERS_USA:
        ticker_categoria[t] = "Titolo USA"
    for t in TICKERS_ETF:
        ticker_categoria[t] = "ETF"
    return ticker_categoria

def _download_single(ticker, period="6mo", interval="1d", min_bars=60):
    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False
    )
    if data is None or data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if "Adj Close" in data.columns:
        price_col = "Adj Close"
    elif "Close" in data.columns:
        price_col = "Close"
    else:
        return None

    data[price_col] = data[price_col].astype(float)
    if data.shape[0] < min_bars:
        return None

    data["Return_20d"] = data[price_col].pct_change(20)
    data["Return_60d"] = data[price_col].pct_change(60)
    data["SMA_50"] = data[price_col].rolling(window=50).mean()
    data["SMA_200"] = data[price_col].rolling(window=200).mean()

    last_idx = data["Return_20d"].last_valid_index()
    if last_idx is None:
        return None

    last = data.loc[last_idx]

    return {
        "price": float(last[price_col]),
        "ret20": float(last["Return_20d"]),
        "ret60": float(last["Return_60d"]),
        "sma50": float(last["SMA_50"]) if not pd.isna(last["SMA_50"]) else None,
        "sma200": float(last["SMA_200"]) if not pd.isna(last["SMA_200"]) else None,
    }

def analyze_market_regime():
    info = {}
    for idx in TICKERS_INDICI:
        res = _download_single(idx)
        if res is None:
            continue
        info[idx] = res

    if not info:
        return {"phase": "Dati indici non disponibili", "details": {}}

    keys = [k for k in ["^GSPC", "^NDX"] if k in info]
    bull_count = 0
    strong_mom_count = 0

    for k in keys:
        d = info[k]
        price = d["price"]
        sma50 = d["sma50"]
        sma200 = d["sma200"]
        ret60 = d["ret60"]

        if sma50 and sma200 and price > sma50 and price > sma200:
            bull_count += 1
        if ret60 and ret60 > 0.10:
            strong_mom_count += 1

    if keys and bull_count == len(keys) and strong_mom_count == len(keys):
        phase = "Possibile fase di rally (indici in forte rialzo)"
    elif keys and bull_count == len(keys):
        phase = "Mercato in rialzo (indici sopra medie principali)"
    elif bull_count > 0:
        phase = "Mercato misto (solo alcuni indici forti)"
    else:
        phase = "Mercato debole / in correzione"

    details = {}
    for name, d in info.items():
        details[name] = {
            "price": round(d["price"], 3),
            "return_20d_pct": round(d["ret20"] * 100, 3) if d["ret20"] is not None else None,
            "return_60d_pct": round(d["ret60"] * 100, 3) if d["ret60"] is not None else None,
        }

    return {"phase": phase, "details": details}

def label_trend(trend_pct):
    if trend_pct > 5:
        return "Forte rialzo"
    elif trend_pct > 1:
        return "Rialzo moderato"
    elif trend_pct > -1:
        return "Laterale"
    elif trend_pct > -5:
        return "Ribasso moderato"
    else:
        return "Forte ribasso"

def label_momentum(ret60_pct):
    if ret60_pct > 20:
        return "Momentum molto forte"
    elif ret60_pct > 5:
        return "Momentum positivo"
    elif ret60_pct > -5:
        return "Momentum neutro"
    else:
        return "Momentum negativo"

def label_signal(score):
    if score > 0.5:
        return "Molto positivo"
    elif score > 0.15:
        return "Positivo"
    elif score > 0:
        return "Leggermente positivo"
    elif score > -0.15:
        return "Neutro / leggermente negativo"
    else:
        return "Negativo"

def scan_market(
    min_score=MIN_SCORE_DEFAULT,
    category_filter=("Titolo USA", "ETF"),
    period="1y",
    interval="1d",
    min_bars=60
):
    ticker_categoria = build_universe()
    tickers = list(ticker_categoria.keys())
    results = []

    for ticker in tickers:
        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False
        )
        if data is None or data.empty:
            continue

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if "Adj Close" in data.columns:
            price_col = "Adj Close"
        elif "Close" in data.columns:
            price_col = "Close"
        else:
            continue

        data[price_col] = data[price_col].astype(float)
        if data.shape[0] < min_bars:
            continue

        data["Return_20d"] = data[price_col].pct_change(20)
        data["Return_60d"] = data[price_col].pct_change(60)
        data["SMA_20"] = data[price_col].rolling(window=20).mean()

        if "Volume" in data.columns and not data["Volume"].isna().all():
            data["Vol_20d_avg"] = data["Volume"].rolling(window=20).mean()
        else:
            data["Vol_20d_avg"] = np.nan

        last_idx = data["Return_60d"].last_valid_index()
        if last_idx is None:
            continue

        last = data.loc[last_idx]
        if pd.isna(last["SMA_20"]) or last["SMA_20"] == 0:
            continue

        price_now = float(last[price_col])
        ret20 = float(last["Return_20d"])
        ret60 = float(last["Return_60d"])
        sma20 = float(last["SMA_20"])
        trend_vs_sma20 = price_now / sma20 - 1.0

        volume_ratio = 1.0
        if "Volume" in data.columns and "Vol_20d_avg" in data.columns:
            vol_20 = last.get("Vol_20d_avg", np.nan)
            if not pd.isna(vol_20) and vol_20 != 0:
                volume_ratio = float(last["Volume"] / vol_20)

        score = (
            0.4 * ret60 +
            0.3 * ret20 +
            0.2 * trend_vs_sma20 +
            0.1 * (volume_ratio - 1.0)
        )

        if score <= min_score:
            continue

        cat = ticker_categoria[ticker]
        if cat not in category_filter:
            continue

        ret20_pct = ret20 * 100.0
        ret60_pct = ret60 * 100.0
        trend_pct = trend_vs_sma20 * 100.0

        trend_lbl = label_trend(trend_pct)
        mom_lbl = label_momentum(ret60_pct)
        signal_lbl = label_signal(score)

        entry = price_now
        sl = entry * (1.0 - STOP_PCT)
        tp = entry * (1.0 + TP_PCT)

        results.append({
            "ticker": ticker,
            "categoria": cat,
            "price": round(price_now, 3),
            "return_20d_pct": round(ret20_pct, 3),
            "return_60d_pct": round(ret60_pct, 3),
            "trend_vs_sma20_pct": round(trend_pct, 3),
            "volume_ratio": round(volume_ratio, 3),
            "score": round(score, 3),
            "trend_label": trend_lbl,
            "momentum_label": mom_lbl,
            "signal_label": signal_lbl,
            "entry_price": round(entry, 3),
            "stop_loss": round(sl, 3),
            "take_profit": round(tp, 3),
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    market = analyze_market_regime()
    return {"market": market, "assets": results}
