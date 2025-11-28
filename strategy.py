import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone

MIN_SCORE_DEFAULT = 0.01
STOP_PCT = 0.03   # 3% sotto
TP_PCT   = 0.06   # 6% sopra

CUSTOM_TICKERS_FILE = "tickers_custom.txt"

# Universe base
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

# Nomi leggibili
TICKER_NAME = {
    "^GSPC": "S&P 500",
    "^NDX": "Nasdaq 100",
    "^DJI": "Dow Jones",
    "^RUT": "Russell 2000",
    "GC=F": "Oro",
    "SI=F": "Argento",
    "CL=F": "Petrolio WTI",
    "NG=F": "Gas Naturale",
    "HG=F": "Rame",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet (Google)",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "BRK-B": "Berkshire Hathaway B",
    "JPM": "JPMorgan Chase",
    "JNJ": "Johnson & Johnson",
    "XOM": "Exxon Mobil",
    "UNH": "UnitedHealth",
    "V": "Visa",
    "PG": "Procter & Gamble",
    "SPY": "SPDR S&P 500 ETF",
    "QQQ": "Invesco QQQ Trust",
    "IWM": "iShares Russell 2000",
    "DIA": "SPDR Dow Jones ETF",
    "GLD": "SPDR Gold Shares",
    "SLV": "iShares Silver Trust",
    "XLE": "Energy Select Sector",
    "XLF": "Financial Select Sector",
    "XLK": "Technology Select Sector",
    "XLV": "Health Care Select Sector",
    "ARKK": "ARK Innovation ETF",
}

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

def load_custom_tickers():
    extra = {}
    try:
        with open(CUSTOM_TICKERS_FILE, "r") as f:
            for line in f:
                t = line.strip().upper()
                if not t or t.startswith("#"):
                    continue
                extra[t] = "Titolo USA"
    except FileNotFoundError:
        pass
    return extra

# ---------- helper per avere SEMPRE una Serie prezzo ----------

def _extract_price_series(data: pd.DataFrame) -> pd.Series | None:
    """
    Da un DataFrame yfinance estrae una SINGOLA serie di prezzo.
    Se trova 'Adj Close' o 'Close', usa quella.
    Se la colonna è DataFrame (più colonne uguali), prende la prima.
    """
    if data is None or data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    base_col = None
    if "Adj Close" in data.columns:
        base_col = "Adj Close"
    elif "Close" in data.columns:
        base_col = "Close"
    else:
        return None

    price = data[base_col]
    if isinstance(price, pd.DataFrame):
        # se per qualche motivo è un DataFrame (colonne duplicate),
        # prendo la prima colonna
        price = price.iloc[:, 0]

    price = price.astype(float)
    return price

# ---------- Analisi indici ----------

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

    price = _extract_price_series(data)
    if price is None or price.shape[0] < min_bars:
        return None

    df = pd.DataFrame(index=price.index)
    df["Price"] = price
    df["Return_20d"] = df["Price"].pct_change(20)
    df["Return_60d"] = df["Price"].pct_change(60)
    df["SMA_50"] = df["Price"].rolling(window=50).mean()
    df["SMA_200"] = df["Price"].rolling(window=200).mean()

    last_idx = df["Return_20d"].last_valid_index()
    if last_idx is None:
        return None

    last = df.loc[last_idx]

    return {
        "price": float(last["Price"]),
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
        return {"phase": "Dati indici non disponibili", "details": {}, "highlights": []}

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
    highlights = []
    for name, d in info.items():
        d20 = d["ret20"]
        d60 = d["ret60"]
        d20p = round(d20 * 100, 3) if d20 is not None else None
        d60p = round(d60 * 100, 3) if d60 is not None else None
        details[name] = {
            "price": round(d["price"], 3),
            "return_20d_pct": d20p,
            "return_60d_pct": d60p,
        }
        if d60p is not None:
            if d60p > 10:
                highlights.append(f"{name} in forte rialzo negli ultimi 60 giorni (+{d60p:.1f}%).")
            elif d60p < -5:
                highlights.append(f"{name} in calo negli ultimi 60 giorni ({d60p:.1f}%).")

    return {"phase": phase, "details": details, "highlights": highlights}

# ---------- Etichette & rating ----------

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
    elif score > 0.35:
        return "Positivo"
    elif score > 0.15:
        return "Leggermente positivo"
    elif score > 0:
        return "Quasi neutro"
    else:
        return "Negativo"

def rating_from_score(score):
    if score > 0.6:
        return 5
    elif score > 0.35:
        return 4
    elif score > 0.15:
        return 3
    elif score > 0.05:
        return 2
    else:
        return 1

def hunter_score_from_features(ret20, ret60, volume_ratio, distance_high_pct):
    m20 = max(ret20, 0.0)
    m60 = max(ret60, 0.0)
    vol_boost = max(volume_ratio - 1.0, 0.0)
    space = 0.0
    if distance_high_pct is not None and distance_high_pct < 0:
        space = min(abs(distance_high_pct) / 100.0, 1.0)
    return (
        0.5 * m20 +
        0.3 * m60 +
        0.1 * vol_boost +
        0.1 * space
    )

# ---------- Scansione mercato ----------

def scan_market(
    min_score=MIN_SCORE_DEFAULT,
    category_filter=("Titolo USA", "ETF"),
    period="1y",
    interval="1d",
    min_bars=60
):
    ticker_categoria = build_universe()
    extra = load_custom_tickers()
    for t, cat in extra.items():
        if t not in ticker_categoria:
            ticker_categoria[t] = cat

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

        price = _extract_price_series(data)
        if price is None or price.shape[0] < min_bars:
            continue

        df = pd.DataFrame(index=price.index)
        df["Price"] = price
        df["Return_20d"] = df["Price"].pct_change(20)
        df["Return_60d"] = df["Price"].pct_change(60)
        df["SMA_20"] = df["Price"].rolling(window=20).mean()
        df["Daily_ret"] = df["Price"].pct_change()

        if "Volume" in data.columns and not data["Volume"].isna().all():
            df["Volume"] = data["Volume"]
            df["Vol_20d_avg"] = df["Volume"].rolling(window=20).mean()
        else:
            df["Volume"] = np.nan
            df["Vol_20d_avg"] = np.nan

        last_idx = df["Return_60d"].last_valid_index()
        if last_idx is None:
            continue

        last = df.loc[last_idx]
        if pd.isna(last["SMA_20"]) or last["SMA_20"] == 0:
            continue

        price_now = float(last["Price"])
        ret20 = float(last["Return_20d"])
        ret60 = float(last["Return_60d"])
        sma20 = float(last["SMA_20"])
        trend_vs_sma20 = price_now / sma20 - 1.0

        volume_ratio = 1.0
        vol_20 = last.get("Vol_20d_avg", np.nan)
        if not pd.isna(vol_20) and vol_20 != 0:
            volume_ratio = float(last["Volume"] / vol_20)

        vol20 = df["Daily_ret"].rolling(20).std().iloc[-1]
        vol20_pct = float(vol20 * np.sqrt(252) * 100) if not pd.isna(vol20) else None

        high_1y = df["Price"].max()
        distance_1y_high_pct = float(price_now / high_1y * 100 - 100.0) if high_1y > 0 else None

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
        rating = rating_from_score(score)
        hs = hunter_score_from_features(ret20, ret60, volume_ratio, distance_1y_high_pct)

        entry = price_now
        sl = entry * (1.0 - STOP_PCT)
        tp = entry * (1.0 + TP_PCT)

        name = TICKER_NAME.get(ticker, ticker)

        results.append({
            "ticker": ticker,
            "name": name,
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
            "rating_1_5": rating,
            "volatility_20d_pct": round(vol20_pct, 3) if vol20_pct is not None else None,
            "distance_1y_high_pct": round(distance_1y_high_pct, 3) if distance_1y_high_pct is not None else None,
            "hunter_score": round(hs, 3),
            "entry_price": round(entry, 3),
            "stop_loss": round(sl, 3),
            "take_profit": round(tp, 3),
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    market = analyze_market_regime()
    run_time_utc = datetime.now(timezone.utc).isoformat()
    top_hunters = sorted(results, key=lambda x: x["hunter_score"], reverse=True)

    return {
        "run_time_utc": run_time_utc,
        "market": market,
        "assets": results,
        "hunters": top_hunters[:50]
    }
