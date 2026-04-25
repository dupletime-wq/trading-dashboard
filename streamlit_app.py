from __future__ import annotations

# Required packages:
# pip install streamlit pandas numpy scipy yfinance plotly pyarrow beautifulsoup4

import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


CACHE_DIR = Path.home() / ".trading_dashboard_cache"
PRICE_CACHE_DIR = CACHE_DIR / "prices"
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DISCLAIMER = (
    "Research dashboard only. This is not investment advice, an order execution "
    "system, or a real-time trading feed."
)
MARKET_CONFIG = {
    "S&P 500": {"benchmark": "SPY", "currency": "USD", "default_limit": 503},
    "KOSPI 200": {"benchmark": "069500.KS", "currency": "KRW", "default_limit": 200},
}
KOSPI_SEED = [
    ("005930", "Samsung Electronics"),
    ("000660", "SK Hynix"),
    ("373220", "LG Energy Solution"),
    ("207940", "Samsung Biologics"),
    ("005380", "Hyundai Motor"),
    ("000270", "Kia"),
    ("068270", "Celltrion"),
    ("105560", "KB Financial"),
    ("055550", "Shinhan Financial"),
    ("035420", "NAVER"),
    ("051910", "LG Chem"),
    ("006400", "Samsung SDI"),
    ("005490", "POSCO Holdings"),
    ("028260", "Samsung C&T"),
    ("012330", "Hyundai Mobis"),
    ("096770", "SK Innovation"),
    ("066570", "LG Electronics"),
    ("032830", "Samsung Life"),
    ("003550", "LG Corp"),
    ("086790", "Hana Financial"),
    ("033780", "KT&G"),
    ("017670", "SK Telecom"),
    ("015760", "KEPCO"),
    ("009150", "Samsung Electro-Mechanics"),
    ("034730", "SK Inc"),
    ("018260", "Samsung SDS"),
    ("010130", "Korea Zinc"),
    ("316140", "Woori Financial"),
    ("011200", "HMM"),
    ("024110", "IBK"),
    ("030200", "KT"),
    ("000810", "Samsung Fire & Marine"),
    ("003670", "Posco Future M"),
    ("090430", "Amorepacific"),
    ("010950", "S-Oil"),
    ("086280", "Hyundai Glovis"),
    ("251270", "Netmarble"),
    ("352820", "HYBE"),
    ("011170", "Lotte Chemical"),
    ("034020", "Doosan Enerbility"),
    ("009540", "HD Korea Shipbuilding"),
    ("010140", "Samsung Heavy Industries"),
    ("267250", "HD Hyundai"),
    ("047050", "Posco International"),
    ("010620", "Hyundai Mipo Dockyard"),
    ("042660", "Hanwha Ocean"),
    ("000720", "Hyundai Engineering & Construction"),
    ("028050", "Samsung E&A"),
    ("161390", "Hankook Tire"),
    ("004020", "Hyundai Steel"),
]
SP500_FALLBACK = [
    ("AAPL", "Apple Inc.", "Information Technology"),
    ("MSFT", "Microsoft", "Information Technology"),
    ("NVDA", "NVIDIA", "Information Technology"),
    ("AMZN", "Amazon", "Consumer Discretionary"),
    ("META", "Meta Platforms", "Communication Services"),
    ("GOOGL", "Alphabet Class A", "Communication Services"),
    ("BRK-B", "Berkshire Hathaway", "Financials"),
    ("LLY", "Eli Lilly", "Health Care"),
    ("AVGO", "Broadcom", "Information Technology"),
    ("JPM", "JPMorgan Chase", "Financials"),
]

st.set_page_config(page_title="Trading Dashboard MVP", layout="wide")


ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class UniverseMember:
    symbol: str
    name: str
    sector: str = ""


@dataclass
class PriceDownloadResult:
    prices: dict[str, pd.DataFrame]
    failures: list[str]
    from_cache: bool
    cache_path: Path


@dataclass
class TrendTemplateResult:
    score: float
    passed: int
    total: int
    conditions: dict[str, bool]


@dataclass
class VCPResult:
    score: float
    is_vcp: bool
    contraction_depths: list[float]
    pivot: float | None
    pivot_distance_pct: float | None
    volume_dry_up: bool


@dataclass
class MarketState:
    status: str
    color: str
    distribution_days: int
    follow_through_day: bool
    message: str


def normalize_us_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


def normalize_kr_symbol(symbol: str) -> str:
    raw = str(symbol).strip()
    if raw.endswith(".KS") or raw.endswith(".KQ"):
        return raw
    return f"{raw.zfill(6)}.KS"


def load_sp500_universe(timeout: int = 15) -> list[UniverseMember]:
    try:
        request = Request(SP500_WIKI_URL, headers={"User-Agent": "trading-dashboard-single-file/0.1"})
        with urlopen(request, timeout=timeout) as response:
            html = response.read()
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", id="constituents")
        if table is None:
            raise ValueError("S&P 500 table not found")

        members = []
        for row in table.find_all("tr")[1:]:
            cells = [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]
            if len(cells) < 2:
                continue
            members.append(
                UniverseMember(
                    symbol=normalize_us_symbol(cells[0]),
                    name=cells[1],
                    sector=cells[2] if len(cells) > 2 else "",
                )
            )
        return members or [UniverseMember(*row) for row in SP500_FALLBACK]
    except Exception:
        return [UniverseMember(*row) for row in SP500_FALLBACK]


def load_kospi_universe() -> list[UniverseMember]:
    return [UniverseMember(symbol=normalize_kr_symbol(symbol), name=name) for symbol, name in KOSPI_SEED]


def members_to_frame(members: list[UniverseMember]) -> pd.DataFrame:
    return pd.DataFrame([member.__dict__ for member in members])


def cache_key(market: str, tickers: list[str], period: str, interval: str) -> str:
    raw = "|".join([market, period, interval, *sorted(tickers)])
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    safe_market = market.lower().replace(" ", "_").replace("&", "and")
    return f"{safe_market}_{period}_{interval}_{digest}.parquet"


def is_fresh(path: Path, ttl_seconds: int) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < ttl_seconds


def normalize_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    out = df.copy()
    out.columns = [str(col).title().replace("Adj Close", "Adj Close") for col in out.columns]
    if "Adj Close" in out.columns and "Close" not in out.columns:
        out["Close"] = out["Adj Close"]
    keep = [col for col in ["Open", "High", "Low", "Close", "Volume"] if col in out.columns]
    out = out[keep].dropna(subset=["Close"])
    if "Volume" not in out.columns:
        out["Volume"] = 0
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.sort_index()


def split_yfinance_frame(raw: pd.DataFrame, requested_tickers: list[str]) -> dict[str, pd.DataFrame]:
    if raw is None or raw.empty:
        return {}

    frames: dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(map(str, raw.columns.get_level_values(0)))
        for ticker in requested_tickers:
            if ticker in level0:
                frames[ticker] = normalize_ohlcv_frame(raw[ticker])
        if not frames:
            level1 = set(map(str, raw.columns.get_level_values(1)))
            for ticker in requested_tickers:
                if ticker in level1:
                    frames[ticker] = normalize_ohlcv_frame(raw.xs(ticker, axis=1, level=1))
    elif len(requested_tickers) == 1:
        frames[requested_tickers[0]] = normalize_ohlcv_frame(raw)
    return {symbol: frame for symbol, frame in frames.items() if not frame.empty}


def prices_to_long(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for ticker, df in prices.items():
        if df.empty:
            continue
        part = df.copy()
        part["Ticker"] = ticker
        part["Date"] = pd.to_datetime(part.index)
        rows.append(part.reset_index(drop=True))
    if not rows:
        return pd.DataFrame(columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"])
    return pd.concat(rows, ignore_index=True)[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]


def long_to_prices(long_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    prices: dict[str, pd.DataFrame] = {}
    if long_df is None or long_df.empty:
        return prices
    for ticker, part in long_df.groupby("Ticker", sort=False):
        frame = part.sort_values("Date").set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
        prices[str(ticker)] = normalize_ohlcv_frame(frame)
    return prices


def load_price_data(
    market: str,
    tickers: list[str],
    period: str = "2y",
    interval: str = "1d",
    chunk_size: int = 80,
    threads: int = 12,
    ttl_seconds: int = 6 * 60 * 60,
    force_refresh: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> PriceDownloadResult:
    PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    unique_tickers = list(dict.fromkeys(tickers))
    cache_path = PRICE_CACHE_DIR / cache_key(market, unique_tickers, period, interval)

    if not force_refresh and is_fresh(cache_path, ttl_seconds):
        prices = long_to_prices(pd.read_parquet(cache_path))
        if progress_callback:
            progress_callback(1, 1, "Loaded prices from local cache.")
        return PriceDownloadResult(prices=prices, failures=[], from_cache=True, cache_path=cache_path)

    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance first: pip install yfinance") from exc

    prices: dict[str, pd.DataFrame] = {}
    failures: list[str] = []
    chunks = [unique_tickers[i : i + chunk_size] for i in range(0, len(unique_tickers), chunk_size)]
    total = max(len(chunks), 1)

    for idx, chunk in enumerate(chunks, start=1):
        if progress_callback:
            progress_callback(idx - 1, total, f"Downloading chunk {idx}/{total}...")
        try:
            raw = yf.download(
                chunk,
                period=period,
                interval=interval,
                auto_adjust=True,
                group_by="ticker",
                threads=threads,
                progress=False,
                actions=False,
                timeout=20,
            )
            split = split_yfinance_frame(raw, chunk)
            prices.update(split)
            failures.extend([ticker for ticker in chunk if ticker not in split])
        except Exception:
            failures.extend(chunk)
        if progress_callback:
            progress_callback(idx, total, f"Downloaded {idx}/{total} chunks.")

    long_df = prices_to_long(prices)
    if not long_df.empty:
        long_df.to_parquet(cache_path, index=False)
    return PriceDownloadResult(prices=prices, failures=sorted(set(failures)), from_cache=False, cache_path=cache_path)


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)
    ranges = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1)
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    return true_range(df).rolling(window, min_periods=window).mean()


def weighted_quarter_return(close: pd.Series) -> float:
    clean = close.dropna()
    if len(clean) < 253:
        return np.nan
    q1 = clean.iloc[-1] / clean.iloc[-64] - 1
    q2 = clean.iloc[-64] / clean.iloc[-127] - 1
    q3 = clean.iloc[-127] / clean.iloc[-190] - 1
    q4 = clean.iloc[-190] / clean.iloc[-253] - 1
    return float(q1 * 0.4 + q2 * 0.2 + q3 * 0.2 + q4 * 0.2)


def calculate_rs_ratings(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for ticker, df in prices.items():
        if "Close" in df:
            rows.append({"Ticker": ticker, "weighted_return": weighted_quarter_return(df["Close"])})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["Ticker", "weighted_return", "rs_rating"])
    valid = out["weighted_return"].notna()
    out["rs_rating"] = 1
    if valid.any():
        ranks = out.loc[valid, "weighted_return"].rank(pct=True, method="min")
        out.loc[valid, "rs_rating"] = np.clip(np.ceil(ranks * 99), 1, 99).astype(int)
    return out


def trend_template(df: pd.DataFrame, rs_rating: float | int | None) -> TrendTemplateResult:
    close = df["Close"].dropna()
    if len(close) < 252:
        return TrendTemplateResult(score=0.0, passed=0, total=8, conditions={})

    sma50 = sma(close, 50)
    sma150 = sma(close, 150)
    sma200 = sma(close, 200)
    current = close.iloc[-1]
    high_52w = close.tail(252).max()
    low_52w = close.tail(252).min()
    rs = 0 if rs_rating is None or pd.isna(rs_rating) else float(rs_rating)
    conditions = {
        "price_above_150_200": bool(current > sma150.iloc[-1] and current > sma200.iloc[-1]),
        "sma150_above_sma200": bool(sma150.iloc[-1] > sma200.iloc[-1]),
        "sma200_rising": bool(sma200.iloc[-1] > sma200.iloc[-22]),
        "sma50_above_150_200": bool(sma50.iloc[-1] > sma150.iloc[-1] and sma50.iloc[-1] > sma200.iloc[-1]),
        "price_above_sma50": bool(current > sma50.iloc[-1]),
        "price_30pct_above_low": bool(current >= low_52w * 1.30),
        "price_within_25pct_high": bool(current >= high_52w * 0.75),
        "rs_at_least_70": bool(rs >= 70),
    }
    passed = sum(conditions.values())
    return TrendTemplateResult(score=passed / len(conditions) * 20.0, passed=passed, total=8, conditions=conditions)


def fallback_extrema(values: np.ndarray, order: int, comparator) -> np.ndarray:
    indices: list[int] = []
    for idx in range(order, len(values) - order):
        window = values[idx - order : idx + order + 1]
        if comparator(values[idx], np.delete(window, order)).all():
            indices.append(idx)
    return np.array(indices, dtype=int)


def local_extrema(series: pd.Series, order: int = 3) -> tuple[np.ndarray, np.ndarray]:
    values = series.dropna().to_numpy(dtype=float)
    if len(values) < order * 2 + 1:
        return np.array([], dtype=int), np.array([], dtype=int)
    try:
        from scipy.signal import argrelextrema

        peaks = argrelextrema(values, np.greater_equal, order=order)[0]
        troughs = argrelextrema(values, np.less_equal, order=order)[0]
    except Exception:
        peaks = fallback_extrema(values, order, np.greater_equal)
        troughs = fallback_extrema(values, order, np.less_equal)
    return peaks, troughs


def detect_vcp(df: pd.DataFrame, lookback: int = 80) -> VCPResult:
    if len(df) < 40:
        return VCPResult(0.0, False, [], None, None, False)

    recent = df.tail(lookback).copy()
    close = recent["Close"].rolling(3, min_periods=1).mean()
    peaks, troughs = local_extrema(close, order=3)
    pairs: list[tuple[int, int, float]] = []
    for peak_idx in peaks:
        later_troughs = troughs[troughs > peak_idx]
        if len(later_troughs) == 0:
            continue
        trough_idx = int(later_troughs[0])
        peak_price = close.iloc[peak_idx]
        trough_price = close.iloc[trough_idx]
        if peak_price > 0:
            pairs.append((int(peak_idx), trough_idx, float((peak_price - trough_price) / peak_price)))

    pairs = pairs[-4:]
    depths = [depth for _, _, depth in pairs if depth > 0.02]
    contracting = len(depths) >= 2 and all(left > right for left, right in zip(depths, depths[1:]))
    volume50 = recent["Volume"].rolling(50, min_periods=20).mean().iloc[-1]
    last5_volume = recent["Volume"].tail(5).mean()
    volume_dry_up = bool(pd.notna(volume50) and volume50 > 0 and last5_volume < volume50 * 0.65)

    pivot = None
    pivot_distance_pct = None
    if pairs:
        last_trough_idx = pairs[-1][1]
        pivot = float(recent["High"].iloc[last_trough_idx:].max())
        current = float(recent["Close"].iloc[-1])
        if pivot > 0:
            pivot_distance_pct = float((pivot - current) / pivot * 100)

    pivot_near = pivot_distance_pct is not None and -3.0 <= pivot_distance_pct <= 5.0
    score = 0.0
    if contracting:
        score += 8.0
    if volume_dry_up:
        score += 6.0
    if pivot_near:
        score += 6.0
    return VCPResult(
        score=min(score, 20.0),
        is_vcp=bool(contracting and (volume_dry_up or pivot_near)),
        contraction_depths=depths,
        pivot=pivot,
        pivot_distance_pct=pivot_distance_pct,
        volume_dry_up=volume_dry_up,
    )


def pocket_pivot_series(df: pd.DataFrame) -> pd.Series:
    if len(df) < 55:
        return pd.Series(False, index=df.index)
    close = df["Close"]
    volume = df["Volume"]
    sma10 = sma(close, 10)
    sma50 = sma(close, 50)
    atr20 = atr(df, 20)
    signals = []
    for idx in range(len(df)):
        if idx < 51:
            signals.append(False)
            continue
        window = df.iloc[idx - 10 : idx]
        down_volume_max = window.loc[window["Close"] < window["Close"].shift(1), "Volume"].max()
        if pd.isna(down_volume_max):
            down_volume_max = 0
        current_close = close.iloc[idx]
        current_low = df["Low"].iloc[idx]
        near_sma = (
            abs(current_low - sma10.iloc[idx]) / sma10.iloc[idx] <= 0.025
            or abs(current_low - sma50.iloc[idx]) / sma50.iloc[idx] <= 0.025
        )
        spread_ok = (df["High"].iloc[idx] - df["Low"].iloc[idx]) <= atr20.iloc[idx] * 2.2
        signals.append(bool(current_close > close.iloc[idx - 1] and volume.iloc[idx] > down_volume_max and near_sma and spread_ok))
    return pd.Series(signals, index=df.index)


def is_pocket_pivot(df: pd.DataFrame) -> bool:
    series = pocket_pivot_series(df)
    return bool(len(series) and series.iloc[-1])


def ants_indicator(df: pd.DataFrame, lookback: int = 15, min_up_days: int = 12) -> bool:
    if len(df) < lookback + 1:
        return False
    recent = df.tail(lookback).copy()
    changes = df["Close"].diff().tail(lookback)
    up_mask = changes > 0
    down_mask = changes < 0
    up_days = int(up_mask.sum())
    up_volume = recent.loc[up_mask, "Volume"].mean()
    down_volume = recent.loc[down_mask, "Volume"].mean()
    volume_ok = pd.isna(down_volume) or up_volume > down_volume
    return bool(up_days >= min_up_days and volume_ok)


def htf_lite(df: pd.DataFrame) -> bool:
    if len(df) < 65:
        return False
    recent = df.tail(65)
    pole_window = recent.iloc[:45]
    flag = recent.tail(20)
    pole_low = pole_window["Close"].min()
    pole_high = pole_window["Close"].max()
    pole_return = pole_high / pole_low - 1 if pole_low > 0 else 0
    flag_drawdown = 1 - flag["Low"].min() / flag["High"].max() if flag["High"].max() > 0 else 1
    flag_at_high = flag["Close"].iloc[-1] >= pole_high * 0.75
    return bool(pole_return >= 0.90 and flag_drawdown <= 0.25 and flag_at_high)


def bearish_reversal_series(df: pd.DataFrame) -> pd.Series:
    if len(df) < 2:
        return pd.Series(False, index=df.index)
    signal = (
        (df["High"] > df["High"].shift(1))
        & (df["Close"] < df["Close"].shift(1))
        & (df["Close"] < df["Open"])
        & (df["Volume"] > df["Volume"].shift(1))
    )
    return signal.fillna(False)


def bearish_one_day_reversal(df: pd.DataFrame) -> bool:
    series = bearish_reversal_series(df)
    return bool(len(series) and series.iloc[-1])


def breakout_volume(df: pd.DataFrame, lookback: int = 20) -> bool:
    if len(df) < 55:
        return False
    recent_high = df["High"].iloc[-lookback - 1 : -1].max()
    volume50 = df["Volume"].rolling(50, min_periods=30).mean().iloc[-1]
    return bool(df["Close"].iloc[-1] >= recent_high and df["Volume"].iloc[-1] >= volume50 * 1.4)


def analyze_market(df: pd.DataFrame) -> MarketState:
    if df.empty or len(df) < 220:
        return MarketState("Unknown", "gray", 0, False, "Not enough benchmark data.")

    close = df["Close"]
    volume = df["Volume"]
    sma50 = sma(close, 50)
    sma200 = sma(close, 200)
    distribution = int(((close < close.shift(1)) & (volume > volume.shift(1))).tail(25).sum())
    recent = df.tail(20)
    low_pos = int(np.argmin(recent["Close"].to_numpy()))
    ftd = False
    for offset in range(low_pos + 4, min(low_pos + 8, len(recent))):
        pct = recent["Close"].iloc[offset] / recent["Close"].iloc[offset - 1] - 1
        vol_up = recent["Volume"].iloc[offset] > recent["Volume"].iloc[offset - 1]
        if pct >= 0.015 and vol_up:
            ftd = True
            break

    uptrend = close.iloc[-1] > sma50.iloc[-1] > sma200.iloc[-1]
    if distribution >= 5:
        return MarketState("Red", "red", distribution, ftd, "Distribution pressure is elevated.")
    if uptrend and (ftd or distribution <= 3):
        return MarketState("Green", "green", distribution, ftd, "Benchmark trend is constructive.")
    return MarketState("Yellow", "orange", distribution, ftd, "Mixed trend conditions; reduce signal aggressiveness.")


def scan_universe(prices: dict[str, pd.DataFrame], benchmark_symbol: str) -> tuple[pd.DataFrame, MarketState]:
    benchmark = prices.get(benchmark_symbol, pd.DataFrame())
    market_state = analyze_market(benchmark)
    scan_prices = {ticker: df for ticker, df in prices.items() if ticker != benchmark_symbol and len(df) >= 60}
    rs = calculate_rs_ratings(scan_prices)
    rs_map = dict(zip(rs["Ticker"], rs["rs_rating"], strict=False))

    rows = []
    for ticker, df in scan_prices.items():
        latest = df.iloc[-1]
        rs_rating = float(rs_map.get(ticker, 1))
        trend = trend_template(df, rs_rating)
        vcp = detect_vcp(df)
        pocket = is_pocket_pivot(df)
        ants = ants_indicator(df)
        htf = htf_lite(df)
        reversal = bearish_one_day_reversal(df)
        breakout = breakout_volume(df)

        rs_score = rs_rating / 99 * 35
        volume_score = min((5 if pocket else 0) + (5 if ants else 0) + (5 if breakout else 0), 15)
        pivot_score = 0.0
        if vcp.pivot_distance_pct is not None:
            if -3 <= vcp.pivot_distance_pct <= 2:
                pivot_score = 10.0
            elif 2 < vcp.pivot_distance_pct <= 5:
                pivot_score = 7.0
            elif 5 < vcp.pivot_distance_pct <= 10:
                pivot_score = 3.0
        composite = min(100.0, rs_score + trend.score + vcp.score + volume_score + pivot_score)
        patterns = []
        if vcp.is_vcp:
            patterns.append("VCP")
        if pocket:
            patterns.append("Pocket Pivot")
        if ants:
            patterns.append("Ants")
        if htf:
            patterns.append("HTF")
        if breakout:
            patterns.append("Breakout Vol")
        if reversal:
            patterns.append("Bearish Reversal")

        rows.append(
            {
                "Ticker": ticker,
                "Last Close": float(latest["Close"]),
                "Volume": int(latest["Volume"]) if pd.notna(latest["Volume"]) else 0,
                "RS": int(rs_rating),
                "Trend": round(trend.score, 1),
                "Trend Pass": f"{trend.passed}/{trend.total}",
                "VCP": round(vcp.score, 1),
                "Pivot": vcp.pivot,
                "Pivot Dist %": None if vcp.pivot_distance_pct is None else round(vcp.pivot_distance_pct, 2),
                "Volume Score": volume_score,
                "Composite": round(composite, 1),
                "Pocket Pivot": pocket,
                "Ants": ants,
                "HTF": htf,
                "Bearish Reversal": reversal,
                "Patterns": ", ".join(patterns) if patterns else "-",
            }
        )

    leaderboard = pd.DataFrame(rows)
    if leaderboard.empty:
        return leaderboard, market_state
    return leaderboard.sort_values(["Composite", "RS"], ascending=False).reset_index(drop=True), market_state


def make_price_chart(ticker: str, df: pd.DataFrame, lookback: int = 180) -> go.Figure:
    chart_df = df.tail(lookback).copy()
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
    )
    fig.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df["Open"],
            high=chart_df["High"],
            low=chart_df["Low"],
            close=chart_df["Close"],
            name="OHLC",
            increasing_line_color="#0f9d58",
            decreasing_line_color="#d93025",
        ),
        row=1,
        col=1,
    )
    for window, color in [(50, "#1a73e8"), (150, "#f9ab00"), (200, "#5f6368")]:
        fig.add_trace(
            go.Scatter(x=chart_df.index, y=sma(df["Close"], window).tail(lookback), mode="lines", name=f"SMA {window}", line=dict(color=color, width=1.4)),
            row=1,
            col=1,
        )

    volume_colors = ["#0f9d58" if close >= open_ else "#d93025" for open_, close in zip(chart_df["Open"], chart_df["Close"], strict=False)]
    fig.add_trace(go.Bar(x=chart_df.index, y=chart_df["Volume"], name="Volume", marker_color=volume_colors), row=2, col=1)

    pocket_dates = pocket_pivot_series(df).tail(lookback)
    pocket_dates = pocket_dates[pocket_dates].index
    if len(pocket_dates):
        fig.add_trace(
            go.Scatter(x=pocket_dates, y=chart_df.loc[pocket_dates, "Low"] * 0.98, mode="markers", name="Pocket Pivot", marker=dict(color="#1a73e8", size=9)),
            row=1,
            col=1,
        )

    reversal_dates = bearish_reversal_series(df).tail(lookback)
    reversal_dates = reversal_dates[reversal_dates].index
    if len(reversal_dates):
        fig.add_trace(
            go.Scatter(
                x=reversal_dates,
                y=chart_df.loc[reversal_dates, "High"] * 1.02,
                mode="markers",
                name="Bearish Reversal",
                marker=dict(color="#b3261e", size=10, symbol="triangle-down"),
            ),
            row=1,
            col=1,
        )

    vcp = detect_vcp(df)
    if vcp.pivot is not None:
        fig.add_hline(y=vcp.pivot, line_width=1.3, line_dash="dash", line_color="#673ab7", row=1, col=1)
        fig.add_annotation(
            x=chart_df.index[-1],
            y=vcp.pivot,
            text=f"Pivot {vcp.pivot:.2f}",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=11, color="#673ab7"),
            row=1,
            col=1,
        )
    if vcp.contraction_depths:
        depths = " / ".join(f"{depth * 100:.1f}%" for depth in vcp.contraction_depths[-3:])
        fig.add_annotation(
            x=chart_df.index[0],
            y=chart_df["High"].max(),
            text=f"VCP depths: {depths}",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(size=11, color="#202124"),
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="#dadce0",
            row=1,
            col=1,
        )

    fig.update_layout(
        title=f"{ticker} Daily Chart",
        height=680,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def cached_universe(market: str) -> pd.DataFrame:
    members = load_sp500_universe() if market == "S&P 500" else load_kospi_universe()
    return members_to_frame(members)


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def cached_scan(long_prices: pd.DataFrame, benchmark_symbol: str):
    prices = long_to_prices(long_prices)
    leaderboard, market_state = scan_universe(prices, benchmark_symbol)
    return leaderboard, {
        "status": market_state.status,
        "color": market_state.color,
        "distribution_days": int(market_state.distribution_days),
        "follow_through_day": bool(market_state.follow_through_day),
        "message": market_state.message,
    }


def render_market_badge(status: str, color: str, message: str) -> None:
    palette = {
        "green": ("#e6f4ea", "#137333"),
        "orange": ("#fef7e0", "#b06000"),
        "red": ("#fce8e6", "#b3261e"),
        "gray": ("#f1f3f4", "#5f6368"),
    }
    bg, fg = palette.get(color, palette["gray"])
    st.sidebar.markdown(
        f"""
        <div style="background:{bg}; color:{fg}; padding:10px 12px; border-radius:8px; border:1px solid rgba(0,0,0,.06);">
            <div style="font-weight:700; font-size:15px;">Market: {status}</div>
            <div style="font-size:12px; line-height:1.35; margin-top:4px;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.title("Trading Dashboard MVP")
    st.caption(DISCLAIMER)

    st.sidebar.header("Scanner")
    market = st.sidebar.radio("Market", list(MARKET_CONFIG.keys()))
    period = st.sidebar.selectbox("History", ["1y", "2y", "5y", "10y"], index=1)
    config = MARKET_CONFIG[market]
    benchmark = config["benchmark"]
    universe_df = cached_universe(market)
    if universe_df.empty:
        st.error("No universe symbols found.")
        return

    default_limit = min(int(config["default_limit"]), len(universe_df))
    max_symbols = st.sidebar.number_input("Max symbols to scan", min_value=5, max_value=max(len(universe_df), 5), value=default_limit, step=10)
    rs_cutoff = st.sidebar.slider("Minimum RS", min_value=1, max_value=99, value=70)
    required_patterns = st.sidebar.multiselect(
        "Required pattern",
        ["VCP", "Pocket Pivot", "Ants", "HTF", "Breakout Vol", "Bearish Reversal"],
        default=[],
    )
    force_refresh = st.sidebar.button("Refresh market data", use_container_width=True)

    selected_universe = universe_df.head(int(max_symbols)).copy()
    tickers = selected_universe["symbol"].tolist()
    tickers_with_benchmark = list(dict.fromkeys([*tickers, benchmark]))
    st.sidebar.caption(f"Universe symbols: {len(tickers)} | Benchmark: {benchmark}")

    progress_bar = st.progress(0, text="Preparing data load...")

    def update_progress(done: int, total: int, message: str) -> None:
        ratio = 1.0 if total <= 0 else min(max(done / total, 0), 1)
        progress_bar.progress(ratio, text=message)

    with st.status("Loading market data", expanded=False) as status:
        result = load_price_data(
            market=market,
            tickers=tickers_with_benchmark,
            period=period,
            force_refresh=force_refresh,
            progress_callback=update_progress,
        )
        source = "cache" if result.from_cache else "yfinance"
        status.update(label=f"Loaded {len(result.prices)} symbols from {source}", state="complete")
    progress_bar.empty()

    if not result.prices:
        st.error("No price data was loaded. Try a smaller universe, different period, or refresh later.")
        if result.failures:
            st.warning("Failed symbols: " + ", ".join(result.failures[:30]))
        return

    long_prices = prices_to_long(result.prices)
    leaderboard, market_state_data = cached_scan(long_prices, benchmark)
    market_state = MarketState(**market_state_data)
    render_market_badge(market_state.status, market_state.color, market_state.message)
    st.sidebar.metric("Distribution Days", market_state.distribution_days)
    st.sidebar.metric("Follow-through Day", "Yes" if market_state.follow_through_day else "No")

    if leaderboard.empty:
        st.warning("Not enough price history to scan this universe.")
        return

    meta = selected_universe.rename(columns={"symbol": "Ticker", "name": "Name", "sector": "Sector"})
    display_df = leaderboard.merge(meta, on="Ticker", how="left")
    display_df["Name"] = display_df["Name"].fillna("")
    display_df["Sector"] = display_df["Sector"].fillna("")
    display_df = display_df[display_df["RS"] >= rs_cutoff]
    for pattern in required_patterns:
        display_df = display_df[display_df["Patterns"].str.contains(pattern, regex=False)]

    st.subheader("Leaderboard")
    if result.failures:
        st.warning(f"{len(result.failures)} symbols failed or returned empty data. First failures: {', '.join(result.failures[:15])}")
    st.caption(f"Local cache: `{result.cache_path}`")

    if display_df.empty:
        st.info("No symbols match the current filters.")
        return

    columns = [
        "Ticker",
        "Name",
        "Sector",
        "Composite",
        "RS",
        "Trend",
        "Trend Pass",
        "VCP",
        "Pivot Dist %",
        "Volume Score",
        "Patterns",
        "Last Close",
        "Volume",
    ]
    event = st.dataframe(
        display_df[columns],
        use_container_width=True,
        hide_index=True,
        height=430,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Composite": st.column_config.ProgressColumn("Composite", min_value=0, max_value=100, format="%.1f"),
            "RS": st.column_config.ProgressColumn("RS", min_value=1, max_value=99, format="%d"),
            "Last Close": st.column_config.NumberColumn("Last Close", format="%.2f"),
            "Volume": st.column_config.NumberColumn("Volume", format="%d"),
            "Pivot Dist %": st.column_config.NumberColumn("Pivot Dist %", format="%.2f"),
        },
    )

    selected_rows = event.selection.rows
    row_idx = selected_rows[0] if selected_rows else 0
    selected = display_df.iloc[row_idx]
    ticker = selected["Ticker"]
    prices = long_to_prices(long_prices)
    selected_df = prices.get(ticker)
    if selected_df is None or selected_df.empty:
        st.warning(f"No chart data for {ticker}.")
        return

    st.subheader(f"{ticker} Detail")
    metric_cols = st.columns(5)
    metric_cols[0].metric("Composite", f"{selected['Composite']:.1f}")
    metric_cols[1].metric("RS", f"{int(selected['RS'])}")
    metric_cols[2].metric("Trend", selected["Trend Pass"])
    metric_cols[3].metric("Pivot Dist", "-" if pd.isna(selected["Pivot Dist %"]) else f"{selected['Pivot Dist %']:.2f}%")
    metric_cols[4].metric("Patterns", selected["Patterns"])
    st.plotly_chart(make_price_chart(ticker, selected_df), use_container_width=True)


if __name__ == "__main__":
    main()
