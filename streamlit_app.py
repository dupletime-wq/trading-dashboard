from __future__ import annotations

# Required packages:
# pip install streamlit pandas numpy scipy yfinance plotly pyarrow beautifulsoup4

import hashlib
import re
import sys
import time
from dataclasses import dataclass, field
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
QUICK_SCAN_LIMIT = 60
DOWNLOAD_TIME_BUDGET_SECONDS = 90
RS_MIN_BARS = 120
DISCLAIMER = (
    "Research dashboard only. This is not investment advice, an order execution "
    "system, or a real-time trading feed."
)
MARKET_CONFIG = {
    "S&P 500": {"benchmark": "SPY", "currency": "USD", "default_limit": QUICK_SCAN_LIMIT},
    "KOSPI 200": {"benchmark": "069500.KS", "currency": "KRW", "default_limit": QUICK_SCAN_LIMIT},
}
KOSPI_SEED = [
    ("090430", "Amorepacific", "Consumer Staples"),
    ("002790", "Amorepacific Holdings", "Consumer Staples"),
    ("278470", "APR", "Consumer Staples"),
    ("002030", "Asia Holdings", "Steels & Materials"),
    ("282330", "BGF Retail", "Consumer Staples"),
    ("138930", "BNK Financial", "Financials"),
    ("068270", "Celltrion", "Health Care"),
    ("030000", "Cheil Worldwide", "Communication Services"),
    ("185750", "Chong Kun Dang", "Health Care"),
    ("001040", "CJ", "Consumer Staples"),
    ("097950", "CJ CheilJedang", "Consumer Staples"),
    ("000120", "CJ Logistics", "Industrials"),
    ("192820", "Cosmax", "Consumer Staples"),
    ("005420", "Cosmo Chemical", "Energy & Chemicals"),
    ("021240", "Coway", "Consumer Discretionary"),
    ("112610", "CS Wind", "Heavy Industries"),
    ("001680", "Daesang", "Consumer Staples"),
    ("047040", "Daewoo E&C", "Constructions"),
    ("003090", "Daewoong", "Health Care"),
    ("069620", "Daewoong Pharmaceutical", "Health Care"),
    ("005830", "DB Insurance", "Financials"),
    ("000210", "DL", "Constructions"),
    ("375500", "DL E&C", "Constructions"),
    ("007340", "DN Automotive", "Consumer Discretionary"),
    ("026960", "Dongsuh", "Consumer Staples"),
    ("006040", "Dongwon Industries", "Consumer Staples"),
    ("014820", "Dongwon Systems", "Steels & Materials"),
    ("000150", "Doosan", "Heavy Industries"),
    ("241560", "Doosan Bobcat", "Heavy Industries"),
    ("034020", "Doosan Enerbility", "Heavy Industries"),
    ("454910", "Doosan Robotics", "Heavy Industries"),
    ("192080", "DoubleU Games", "Consumer Discretionary"),
    ("450080", "Ecopro Materials", "Industrials"),
    ("139480", "Emart", "Consumer Staples"),
    ("383220", "F&F", "Consumer Discretionary"),
    ("093370", "Foosung", "Energy & Chemicals"),
    ("114090", "Grand Korea Leisure", "Consumer Discretionary"),
    ("006280", "Green Cross", "Health Care"),
    ("005250", "Green Cross Holdings", "Health Care"),
    ("078930", "GS", "Energy & Chemicals"),
    ("006360", "GS E&C", "Constructions"),
    ("007070", "GS Retail", "Consumer Staples"),
    ("012630", "HDC", "Constructions"),
    ("086790", "Hana Financial", "Financials"),
    ("009420", "Hanall Biopharma", "Health Care"),
    ("300720", "Hanil Cement", "Constructions"),
    ("180640", "Hanjin KAL", "Consumer Discretionary"),
    ("161390", "Hankook", "Consumer Discretionary"),
    ("000240", "Hankook & Company", "Consumer Discretionary"),
    ("128940", "Hanmi Pharm", "Health Care"),
    ("008930", "Hanmi Science", "Health Care"),
    ("042700", "Hanmi Semiconductor", "IT"),
    ("018880", "Hanon Systems", "Consumer Discretionary"),
    ("014680", "Hansol Chemical", "Energy & Chemicals"),
    ("009240", "Hanssem", "Consumer Discretionary"),
    ("000880", "Hanwha", "Energy & Chemicals"),
    ("012450", "Hanwha Aerospace", "Industrials"),
    ("082740", "Hanwha Engine", "Heavy Industries"),
    ("088350", "Hanwha Life", "Financials"),
    ("042660", "Hanwha Ocean", "Heavy Industries"),
    ("009830", "Hanwha Solutions", "Energy & Chemicals"),
    ("272210", "Hanwha Systems", "Industrials"),
    ("267250", "HD Hyundai", "Energy & Chemicals"),
    ("267260", "HD Hyundai Electric", "Heavy Industries"),
    ("329180", "HD Hyundai Heavy Industries", "Heavy Industries"),
    ("071970", "HD Hyundai Marine Engine", "Heavy Industries"),
    ("443060", "HD Hyundai Marine Solution", "Heavy Industries"),
    ("009540", "HD KSOE", "Heavy Industries"),
    ("017960", "Hankuk Carbon", "Energy & Chemicals"),
    ("000080", "HiteJinro", "Consumer Staples"),
    ("204320", "HL Mando", "Consumer Discretionary"),
    ("011200", "HMM", "Industrials"),
    ("008770", "Hotel Shilla", "Consumer Discretionary"),
    ("298050", "HS Hyosung Advanced Materials", "Energy & Chemicals"),
    ("352820", "Hybe", "Communication Services"),
    ("298040", "Hyosung Heavy Industries", "Heavy Industries"),
    ("298020", "Hyosung TNC", "Energy & Chemicals"),
    ("307950", "Hyundai AutoEver", "IT"),
    ("069960", "Hyundai Department Store", "Consumer Discretionary"),
    ("000720", "Hyundai E&C", "Constructions"),
    ("017800", "Hyundai Elevator", "Heavy Industries"),
    ("086280", "Hyundai Glovis", "Industrials"),
    ("001450", "Hyundai Marine & Fire", "Financials"),
    ("012330", "Hyundai Mobis", "Consumer Discretionary"),
    ("005380", "Hyundai Motor", "Consumer Discretionary"),
    ("064350", "Hyundai Rotem", "Heavy Industries"),
    ("004020", "Hyundai Steel", "Steels & Materials"),
    ("011210", "Hyundai WIA", "Consumer Discretionary"),
    ("139130", "IM Financial", "Financials"),
    ("024110", "Industrial Bank of Korea", "Financials"),
    ("007660", "Isu Petasys", "IT"),
    ("457190", "Isu Specialty Chemical", "Energy & Chemicals"),
    ("175330", "JB Financial", "Financials"),
    ("035720", "Kakao", "Communication Services"),
    ("323410", "KakaoBank", "Financials"),
    ("377300", "KakaoPay", "Financials"),
    ("035250", "Kangwon Land", "Consumer Discretionary"),
    ("105560", "KB Financial", "Financials"),
    ("002380", "KCC", "Constructions"),
    ("015760", "KEPCO", "Consumer Staples"),
    ("052690", "KEPCO E&C", "Constructions"),
    ("051600", "KEPCO KPS", "Industrials"),
    ("000270", "Kia", "Consumer Discretionary"),
    ("039490", "Kiwoom Securities", "Financials"),
    ("161890", "Kolmar Korea", "Consumer Staples"),
    ("120110", "Kolon Industries", "Energy & Chemicals"),
    ("047810", "Korea Aerospace", "Industrials"),
    ("071320", "Korea District Heating", "Consumer Staples"),
    ("036460", "Korea Gas", "Consumer Staples"),
    ("071050", "Korea Investment", "Financials"),
    ("006650", "Korea Petrochemical", "Energy & Chemicals"),
    ("010130", "Korea Zinc", "Steels & Materials"),
    ("003490", "Korean Air", "Industrials"),
    ("259960", "Krafton", "Communication Services"),
    ("030200", "KT", "Communication Services"),
    ("033780", "KT&G", "Consumer Staples"),
    ("011780", "Kumho Petrochemical", "Energy & Chemicals"),
    ("073240", "Kumho Tire", "Consumer Discretionary"),
    ("066970", "L&F", "Industrials"),
    ("003550", "LG", "IT"),
    ("051910", "LG Chem", "Energy & Chemicals"),
    ("064400", "LG CNS", "IT"),
    ("034220", "LG Display", "IT"),
    ("066570", "LG Electronics", "IT"),
    ("373220", "LG Energy Solution", "Industrials"),
    ("051900", "LG H&H", "Consumer Staples"),
    ("011070", "LG Innotek", "IT"),
    ("032640", "LG Uplus", "Communication Services"),
    ("079550", "LIG Nex1", "Industrials"),
    ("004990", "Lotte", "Consumer Staples"),
    ("011170", "Lotte Chemical", "Energy & Chemicals"),
    ("005300", "Lotte Chilsung", "Consumer Staples"),
    ("004000", "Lotte Fine Chemical", "Energy & Chemicals"),
    ("023530", "Lotte Shopping", "Consumer Discretionary"),
    ("280360", "Lotte Wellfood", "Consumer Staples"),
    ("006260", "LS", "Industrials"),
    ("010120", "LS Electric", "Industrials"),
    ("138040", "Meritz Financial", "Financials"),
    ("006800", "Mirae Asset Securities", "Financials"),
    ("081660", "Misto", "Consumer Discretionary"),
    ("002840", "Miwon Commercial", "Energy & Chemicals"),
    ("268280", "Miwon Specialty Chemical", "Energy & Chemicals"),
    ("035420", "Naver", "Communication Services"),
    ("036570", "NCSoft", "Communication Services"),
    ("251270", "Netmarble", "Communication Services"),
    ("005940", "NH Investment & Securities", "Financials"),
    ("004370", "Nongshim", "Consumer Staples"),
    ("010060", "OCI Holdings", "Energy & Chemicals"),
    ("271560", "Orion", "Consumer Staples"),
    ("001800", "Orion Holdings", "Consumer Staples"),
    ("007310", "Ottogi", "Consumer Staples"),
    ("028670", "Pan Ocean", "Industrials"),
    ("034230", "Paradise", "Consumer Discretionary"),
    ("103140", "Poongsan", "Steels & Materials"),
    ("005490", "POSCO", "Steels & Materials"),
    ("022100", "POSCO DX", "IT"),
    ("003670", "POSCO Future M", "Industrials"),
    ("047050", "POSCO International", "Industrials"),
    ("012750", "S-1", "Industrials"),
    ("062040", "Sanil Electric", "Heavy Industries"),
    ("207940", "Samsung Biologics", "Health Care"),
    ("028260", "Samsung C&T", "Constructions"),
    ("029780", "Samsung Card", "Financials"),
    ("028050", "Samsung E&A", "Constructions"),
    ("009150", "Samsung Electro-Mechanics", "IT"),
    ("005930", "Samsung Electronics", "IT"),
    ("000810", "Samsung Fire & Marine", "Financials"),
    ("010140", "Samsung Heavy Industries", "Heavy Industries"),
    ("032830", "Samsung Life", "Financials"),
    ("006400", "Samsung SDI", "IT"),
    ("018260", "Samsung SDS", "IT"),
    ("016360", "Samsung Securities", "Financials"),
    ("003230", "Samyang Foods", "Consumer Staples"),
    ("137310", "SD Biosensor", "Health Care"),
    ("001430", "Seah Besteel", "Steels & Materials"),
    ("003030", "Seah Steel Holdings", "Steels & Materials"),
    ("004490", "Sebang Global Battery", "Consumer Discretionary"),
    ("055550", "Shinhan Financial", "Financials"),
    ("004170", "Shinsegae", "Consumer Discretionary"),
    ("034730", "SK", "Energy & Chemicals"),
    ("326030", "SK Biopharm", "Health Care"),
    ("302440", "SK Bioscience", "Health Care"),
    ("285130", "SK Chemicals", "Energy & Chemicals"),
    ("000660", "SK Hynix", "IT"),
    ("361610", "SK IE Technology", "Industrials"),
    ("096770", "SK Innovation", "Energy & Chemicals"),
    ("402340", "SK Square", "IT"),
    ("017670", "SK Telecom", "Communication Services"),
    ("011790", "SKC", "Energy & Chemicals"),
    ("005850", "SL", "Consumer Discretionary"),
    ("010950", "S-Oil", "Energy & Chemicals"),
    ("003240", "Taekwang Industrial", "Energy & Chemicals"),
    ("001440", "Taihan Cable & Solution", "Industrials"),
    ("069260", "TKG Huchems", "Energy & Chemicals"),
    ("316140", "Woori Financial", "Financials"),
    ("008730", "Youlchon Chemical", "Steels & Materials"),
    ("000670", "Young Poong", "Steels & Materials"),
    ("111770", "Youngone", "Consumer Discretionary"),
    ("009970", "Youngone Holdings", "Consumer Discretionary"),
    ("000100", "Yuhan", "Health Care"),
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
    stale_cache: bool = False
    warnings: list[str] = field(default_factory=list)
    invalid_tickers: list[str] = field(default_factory=list)


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
    label: str = "No VCP"
    max_depth: float | None = None


@dataclass
class BreakoutStatus:
    state: str
    is_breakout: bool
    is_active: bool
    is_extended: bool
    is_risk: bool
    breakout_age: int | None
    extension_pct: float | None
    close_position: float | None
    risk_flags: list[str] = field(default_factory=list)


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
    members = []
    for row in KOSPI_SEED:
        symbol, name, *rest = row
        sector = rest[0] if rest else ""
        members.append(UniverseMember(symbol=normalize_kr_symbol(symbol), name=name, sector=sector))
    return members


def members_to_frame(members: list[UniverseMember]) -> pd.DataFrame:
    return pd.DataFrame([member.__dict__ for member in members])


def is_valid_ticker(market: str, ticker: str) -> bool:
    if market == "KOSPI 200":
        return bool(re.fullmatch(r"\d{6}\.KS", ticker))
    return bool(re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,14}", ticker))


def split_valid_tickers(market: str, tickers: list[str]) -> tuple[list[str], list[str]]:
    valid: list[str] = []
    invalid: list[str] = []
    for ticker in dict.fromkeys(tickers):
        symbol = str(ticker).strip()
        if is_valid_ticker(market, symbol):
            valid.append(symbol)
        else:
            invalid.append(symbol)
    return valid, invalid


def cache_key(market: str, tickers: list[str], period: str, interval: str) -> str:
    raw = "|".join([market, period, interval, *sorted(tickers)])
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    safe_market = market.lower().replace(" ", "_").replace("&", "and")
    return f"{safe_market}_{period}_{interval}_{digest}.parquet"


def is_fresh(path: Path, ttl_seconds: int) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < ttl_seconds


def read_price_cache(path: Path) -> dict[str, pd.DataFrame]:
    if not path.exists():
        return {}
    try:
        return long_to_prices(pd.read_parquet(path))
    except Exception:
        return {}


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
    chunk_size: int = 60,
    threads: int = 8,
    ttl_seconds: int = 6 * 60 * 60,
    force_refresh: bool = False,
    max_elapsed_seconds: int = DOWNLOAD_TIME_BUDGET_SECONDS,
    progress_callback: ProgressCallback | None = None,
) -> PriceDownloadResult:
    PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    unique_tickers, invalid_tickers = split_valid_tickers(market, tickers)
    cache_path = PRICE_CACHE_DIR / cache_key(market, unique_tickers, period, interval)
    warnings: list[str] = []

    if invalid_tickers:
        warnings.append(f"Excluded {len(invalid_tickers)} invalid ticker(s) before download.")

    cached_prices = read_price_cache(cache_path)
    cache_is_stale = bool(cached_prices) and not is_fresh(cache_path, ttl_seconds)
    if cached_prices and not force_refresh:
        failures = sorted({ticker for ticker in unique_tickers if ticker not in cached_prices})
        if cache_is_stale:
            warnings.append("Showing local cache first. Use Refresh market data to update from Yahoo.")
        if failures:
            warnings.append(f"Local cache is missing {len(failures)} valid ticker(s).")
        if progress_callback:
            progress_callback(1, 1, "Loaded prices from local cache.")
        return PriceDownloadResult(
            prices=cached_prices,
            failures=failures,
            from_cache=True,
            cache_path=cache_path,
            stale_cache=cache_is_stale,
            warnings=warnings,
            invalid_tickers=invalid_tickers,
        )

    if not unique_tickers:
        return PriceDownloadResult(
            prices={},
            failures=[],
            from_cache=False,
            cache_path=cache_path,
            warnings=warnings,
            invalid_tickers=invalid_tickers,
        )

    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance first: pip install yfinance") from exc

    def download_symbols(symbols: list[str], active_threads: int, timeout: int) -> dict[str, pd.DataFrame]:
        raw = yf.download(
            symbols,
            period=period,
            interval=interval,
            auto_adjust=True,
            group_by="ticker",
            threads=active_threads,
            progress=False,
            actions=False,
            timeout=timeout,
        )
        return split_yfinance_frame(raw, symbols)

    prices: dict[str, pd.DataFrame] = {}
    chunks = [unique_tickers[i : i + chunk_size] for i in range(0, len(unique_tickers), chunk_size)]
    total = max(len(chunks), 1)
    start_time = time.perf_counter()

    for idx, chunk in enumerate(chunks, start=1):
        elapsed = time.perf_counter() - start_time
        if elapsed >= max_elapsed_seconds:
            warnings.append(f"Stopped Yahoo refresh after {int(elapsed)}s and returned partial data/cache.")
            break
        if progress_callback:
            progress_callback(idx - 1, total, f"Downloading chunk {idx}/{total}...")
        split: dict[str, pd.DataFrame] = {}
        try:
            split = download_symbols(chunk, threads, timeout=15)
        except Exception as exc:
            warnings.append(f"Yahoo chunk {idx}/{total} failed: {type(exc).__name__}")
        prices.update(split)

        missing = [ticker for ticker in chunk if ticker not in split]
        retry_chunk = missing[:10]
        if retry_chunk and (time.perf_counter() - start_time) < max_elapsed_seconds:
            try:
                prices.update(download_symbols(retry_chunk, min(4, threads), timeout=10))
            except Exception:
                pass
            if len(missing) > len(retry_chunk):
                warnings.append(f"Skipped retry for {len(missing) - len(retry_chunk)} missing symbol(s) in chunk {idx} to keep loading fast.")
        if progress_callback:
            progress_callback(idx, total, f"Downloaded {idx}/{total} chunks.")

    stale_cache_used = False
    if prices and cached_prices:
        missing = [ticker for ticker in unique_tickers if ticker not in prices and ticker in cached_prices]
        if missing:
            prices.update({ticker: cached_prices[ticker] for ticker in missing})
            stale_cache_used = True
            warnings.append(f"Used stale cache for {len(missing)} symbols that failed to refresh.")
    elif not prices and cached_prices:
        prices = cached_prices
        stale_cache_used = True
        warnings.append("Download failed; using stale local cache.")

    failures = sorted({ticker for ticker in unique_tickers if ticker not in prices})
    long_df = prices_to_long(prices)
    if not long_df.empty:
        long_df.to_parquet(cache_path, index=False)
    return PriceDownloadResult(
        prices=prices,
        failures=failures,
        from_cache=False,
        cache_path=cache_path,
        stale_cache=stale_cache_used,
        warnings=warnings,
        invalid_tickers=invalid_tickers,
    )


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
    if len(clean) < RS_MIN_BARS:
        return np.nan
    quarter = min(63, (len(clean) - 1) // 4)
    if quarter < 20 or len(clean) < quarter * 4 + 1:
        return np.nan
    latest = clean.iloc[-1]
    q1_base = clean.iloc[-1 - quarter]
    q2_base = clean.iloc[-1 - quarter * 2]
    q3_base = clean.iloc[-1 - quarter * 3]
    q4_base = clean.iloc[-1 - quarter * 4]
    if min(latest, q1_base, q2_base, q3_base, q4_base) <= 0:
        return np.nan
    q1 = latest / q1_base - 1
    q2 = q1_base / q2_base - 1
    q3 = q2_base / q3_base - 1
    q4 = q3_base / q4_base - 1
    return float(q1 * 0.4 + q2 * 0.2 + q3 * 0.2 + q4 * 0.2)


def calculate_rs_ratings(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for ticker, df in prices.items():
        if "Close" in df:
            clean_len = int(df["Close"].dropna().shape[0])
            rows.append(
                {
                    "Ticker": ticker,
                    "weighted_return": weighted_quarter_return(df["Close"]),
                    "rs_bars": clean_len,
                    "rs_valid": clean_len >= RS_MIN_BARS,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["Ticker", "weighted_return", "rs_rating", "rs_bars", "rs_valid"])
    valid = out["weighted_return"].notna()
    out["rs_rating"] = 1
    if valid.any():
        ranks = out.loc[valid, "weighted_return"].rank(pct=True, method="average")
        out.loc[valid, "rs_rating"] = np.clip(np.ceil(ranks * 99), 1, 99).astype(int)
    out["rs_valid"] = valid
    return out


def trend_template(df: pd.DataFrame, rs_rating: float | int | None) -> TrendTemplateResult:
    close = df["Close"].dropna()
    if len(close) < 220:
        return TrendTemplateResult(score=0.0, passed=0, total=8, conditions={})

    sma50 = sma(close, 50)
    sma150 = sma(close, 150)
    sma200 = sma(close, 200)
    current = close.iloc[-1]
    high_52w = close.tail(min(252, len(close))).max()
    low_52w = close.tail(min(252, len(close))).min()
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
    contracting = len(depths) >= 2 and depths[0] > depths[-1]
    strictly_contracting = len(depths) >= 2 and all(left > right for left, right in zip(depths, depths[1:]))
    max_depth = max(depths) if depths else None
    last_depth = depths[-1] if depths else None
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
    balanced_vcp = bool(
        len(depths) >= 2
        and contracting
        and last_depth is not None
        and last_depth <= 0.12
        and max_depth is not None
        and max_depth <= 0.35
        and (volume_dry_up or pivot_near)
    )
    candidate_vcp = bool(
        not balanced_vcp
        and len(depths) >= 2
        and contracting
        and max_depth is not None
        and max_depth <= 0.45
        and (pivot_near or volume_dry_up or (last_depth is not None and last_depth <= 0.18))
    )
    label = "VCP" if balanced_vcp else "VCP Candidate" if candidate_vcp else "No VCP"
    score = 0.0
    if contracting:
        score += 8.0
    if strictly_contracting:
        score += 2.0
    if last_depth is not None and last_depth <= 0.12:
        score += 3.0
    if max_depth is not None and max_depth <= 0.35:
        score += 3.0
    if volume_dry_up:
        score += 6.0
    if pivot_near:
        score += 6.0
    if label == "No VCP":
        score = min(score, 8.0)
    elif label == "VCP Candidate":
        score = min(score, 14.0)
    return VCPResult(
        score=min(score, 20.0),
        is_vcp=balanced_vcp,
        contraction_depths=depths,
        pivot=pivot,
        pivot_distance_pct=pivot_distance_pct,
        volume_dry_up=volume_dry_up,
        label=label,
        max_depth=max_depth,
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


def is_pocket_pivot_last(df: pd.DataFrame) -> bool:
    if len(df) < 55:
        return False
    close = df["Close"]
    volume = df["Volume"]
    window = df.iloc[-11:-1]
    down_volume_max = window.loc[window["Close"] < window["Close"].shift(1), "Volume"].max()
    if pd.isna(down_volume_max):
        down_volume_max = 0

    sma10_last = close.tail(10).mean()
    sma50_last = close.tail(50).mean()
    atr20_last = true_range(df).tail(20).mean()
    current_low = df["Low"].iloc[-1]
    if pd.isna(sma10_last) or pd.isna(sma50_last) or pd.isna(atr20_last) or atr20_last <= 0:
        return False

    near_sma = (
        abs(current_low - sma10_last) / sma10_last <= 0.025
        or abs(current_low - sma50_last) / sma50_last <= 0.025
    )
    spread_ok = (df["High"].iloc[-1] - df["Low"].iloc[-1]) <= atr20_last * 2.2
    return bool(close.iloc[-1] > close.iloc[-2] and volume.iloc[-1] > down_volume_max and near_sma and spread_ok)


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


def clamp(value: float, low: float, high: float) -> float:
    if pd.isna(value):
        return low
    return float(min(max(value, low), high))


def relative_volume(df: pd.DataFrame, window: int = 50) -> float:
    if len(df) < 20 or "Volume" not in df:
        return np.nan
    volume_avg = df["Volume"].rolling(window, min_periods=min(20, window)).mean().iloc[-1]
    current_volume = df["Volume"].iloc[-1]
    if pd.isna(volume_avg) or volume_avg <= 0:
        return np.nan
    return float(current_volume / volume_avg)


def volume_score_from_rel(rel_volume: float) -> float:
    if pd.isna(rel_volume):
        return 0.0
    if rel_volume >= 1.8:
        return 10.0
    if rel_volume >= 1.5:
        return 8.5
    if rel_volume >= 1.2:
        return 7.0
    if rel_volume >= 1.0:
        return 5.5
    if rel_volume >= 0.7:
        return 3.0
    return 1.0


def pivot_score_from_distance(distance_pct: float | None) -> float:
    if distance_pct is None or pd.isna(distance_pct):
        return 0.0
    if -2.0 <= distance_pct <= 2.0:
        return 10.0
    if 2.0 < distance_pct <= 5.0:
        return 8.0
    if 5.0 < distance_pct <= 10.0:
        return 4.5
    if -5.0 <= distance_pct < -2.0:
        return 6.0
    return 0.0


def effective_pivot(df: pd.DataFrame, vcp: VCPResult, lookback: int = 20) -> tuple[float | None, float | None]:
    if vcp.pivot is not None and vcp.pivot_distance_pct is not None:
        return vcp.pivot, vcp.pivot_distance_pct
    if len(df) < lookback + 1:
        return None, None
    pivot = float(df["High"].iloc[-lookback - 1 : -1].max())
    current = float(df["Close"].iloc[-1])
    if pivot <= 0:
        return None, None
    return pivot, float((pivot - current) / pivot * 100)


def candle_close_position(df: pd.DataFrame) -> float | None:
    if df.empty:
        return None
    high = float(df["High"].iloc[-1])
    low = float(df["Low"].iloc[-1])
    close = float(df["Close"].iloc[-1])
    spread = high - low
    if spread <= 0:
        return None
    return float((close - low) / spread)


def breakout_age(df: pd.DataFrame, pivot: float | None, lookback: int = 40) -> int | None:
    if pivot is None or pd.isna(pivot) or pivot <= 0 or len(df) < 2:
        return None
    close = df["Close"].tail(lookback + 1)
    crosses = (close > pivot) & (close.shift(1) <= pivot)
    if not crosses.any():
        return None
    last_cross_date = crosses[crosses].index[-1]
    return int(len(close.loc[last_cross_date:]) - 1)


def breakout_status(
    df: pd.DataFrame,
    pivot: float | None,
    rel_vol: float,
    reversal: bool,
    trend_passed: int,
) -> BreakoutStatus:
    if pivot is None or pd.isna(pivot) or pivot <= 0 or len(df) < 55:
        return BreakoutStatus("Watch", False, False, False, False, None, None, None, [])

    close = df["Close"]
    current = float(close.iloc[-1])
    previous = float(close.iloc[-2])
    extension_pct = float((current - pivot) / pivot * 100)
    close_position = candle_close_position(df)
    age = breakout_age(df, pivot)
    sma10_last = close.tail(10).mean()
    sma20_last = close.tail(20).mean()
    sma50_last = close.tail(50).mean()
    above_pivot = current > pivot
    crossed_today = above_pivot and previous <= pivot
    rel_vol_ok = pd.notna(rel_vol) and rel_vol >= 1.2
    close_position_ok = close_position is not None and close_position >= 0.50
    breakout = bool(crossed_today and rel_vol_ok and close_position_ok)

    ma10_ext = (current / sma10_last - 1) * 100 if pd.notna(sma10_last) and sma10_last > 0 else 0.0
    ma20_ext = (current / sma20_last - 1) * 100 if pd.notna(sma20_last) and sma20_last > 0 else 0.0
    holds_short_ma = bool(pd.notna(sma10_last) and pd.notna(sma20_last) and current > sma10_last and current > sma20_last)
    extended = bool(above_pivot and (extension_pct > 8.0 or ma10_ext > 8.0 or ma20_ext > 12.0))
    active = bool(above_pivot and not breakout and not extended and extension_pct <= 8.0 and holds_short_ma)

    flags: list[str] = []
    if reversal:
        flags.append("Bearish reversal")
    if previous > pivot and current < pivot:
        flags.append("Lost pivot")
    if crossed_today and (not rel_vol_ok or not close_position_ok):
        flags.append("Weak breakout")
    if pd.notna(sma50_last) and current < sma50_last:
        flags.append("Below 50SMA")
    if trend_passed < 4:
        flags.append("Weak trend")
    risk = bool(flags and ("Bearish reversal" in flags or "Lost pivot" in flags or "Below 50SMA" in flags))

    if risk:
        state = "Risk"
    elif extended:
        state = "Extended"
    elif breakout:
        state = "Breakout"
    elif active:
        state = "Active"
    else:
        state = "Watch"
    return BreakoutStatus(state, breakout, active, extended, risk, age, extension_pct, close_position, flags)


def classify_state(
    setup_score: float,
    near_pivot: bool,
    trend_passed: int,
    rs_rating: float,
    status: BreakoutStatus,
) -> str:
    if status.state in {"Risk", "Extended", "Breakout", "Active"}:
        return status.state
    if near_pivot and setup_score >= 50 and trend_passed >= 4 and rs_rating >= 45:
        return "Setup"
    return "Watch"


STATE_RANK = {"Risk": 1, "Extended": 2, "Watch": 3, "Active": 4, "Setup": 5, "Breakout": 6}


def risk_flags(
    reversal: bool,
    pivot_distance_pct: float | None,
    rel_vol: float,
    trend_passed: int,
    status: BreakoutStatus | None = None,
) -> str:
    flags = list(status.risk_flags) if status is not None else []
    if reversal:
        flags.append("Bearish reversal")
    if pivot_distance_pct is not None and pd.notna(pivot_distance_pct) and pivot_distance_pct < -6:
        flags.append("Extended")
    if pd.notna(rel_vol) and rel_vol < 0.6:
        flags.append("Low volume")
    if trend_passed < 4:
        flags.append("Weak trend")
    flags = list(dict.fromkeys(flags))
    return ", ".join(flags) if flags else "-"


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
    rs_return_map = dict(zip(rs["Ticker"], rs.get("weighted_return", pd.Series(dtype=float)), strict=False))
    rs_bars_map = dict(zip(rs["Ticker"], rs.get("rs_bars", pd.Series(dtype=int)), strict=False))
    rs_valid_map = dict(zip(rs["Ticker"], rs.get("rs_valid", pd.Series(dtype=bool)), strict=False))

    rows = []
    for ticker, df in scan_prices.items():
        latest = df.iloc[-1]
        rs_rating = float(rs_map.get(ticker, 1))
        weighted_return = rs_return_map.get(ticker, np.nan)
        rs_bars = int(rs_bars_map.get(ticker, len(df)))
        rs_valid = bool(rs_valid_map.get(ticker, False))
        trend = trend_template(df, rs_rating)
        vcp = detect_vcp(df)
        pocket = is_pocket_pivot_last(df)
        ants = ants_indicator(df)
        htf = htf_lite(df)
        reversal = bearish_one_day_reversal(df)

        pivot, pivot_distance_pct = effective_pivot(df, vcp)
        rel_vol = relative_volume(df)
        near_pivot = pivot_distance_pct is not None and 0.0 <= pivot_distance_pct <= 5.0
        status = breakout_status(df, pivot, rel_vol, reversal, trend.passed)

        rs_norm = clamp(rs_rating / 99 * 10, 0, 10)
        trend_norm = clamp(trend.score / 20 * 10, 0, 10)
        pivot_norm = pivot_score_from_distance(pivot_distance_pct)
        rel_volume_norm = volume_score_from_rel(rel_vol)
        base_norm = clamp(vcp.score / 20 * 10 + (1.0 if pocket else 0.0) + (1.0 if htf else 0.0), 0, 10)
        setup_score = (
            rs_norm * 0.25
            + trend_norm * 0.25
            + pivot_norm * 0.20
            + rel_volume_norm * 0.15
            + base_norm * 0.15
        ) * 10
        if reversal:
            setup_score -= 7.0
        if pivot_distance_pct is not None and pd.notna(pivot_distance_pct) and pivot_distance_pct < -6.0:
            setup_score -= 5.0
        if pd.notna(rel_vol) and rel_vol < 0.5:
            setup_score -= 3.0
        if trend.passed < 4:
            setup_score -= 4.0
        setup_score = clamp(setup_score, 0, 100)
        state = classify_state(setup_score, near_pivot, trend.passed, rs_rating, status)
        state_rank = STATE_RANK[state]

        patterns = []
        if vcp.label == "VCP":
            patterns.append("VCP")
        elif vcp.label == "VCP Candidate":
            patterns.append("VCP Candidate")
        if pocket:
            patterns.append("Pocket Pivot")
        if ants:
            patterns.append("Ants")
        if htf:
            patterns.append("HTF")
        if status.is_breakout:
            patterns.append("Breakout")
        if reversal:
            patterns.append("Bearish Reversal")

        rows.append(
            {
                "Ticker": ticker,
                "Last Close": float(latest["Close"]),
                "Volume": int(latest["Volume"]) if pd.notna(latest["Volume"]) else 0,
                "RS": int(rs_rating),
                "weighted_return": None if pd.isna(weighted_return) else round(float(weighted_return), 4),
                "RS Bars": rs_bars,
                "RS Valid": rs_valid,
                "Trend": round(trend.score, 1),
                "Trend Pass": f"{trend.passed}/{trend.total}",
                "VCP": round(vcp.score, 1),
                "Pivot": pivot,
                "Pivot Dist %": None if pivot_distance_pct is None else round(pivot_distance_pct, 2),
                "Rel Volume": None if pd.isna(rel_vol) else round(rel_vol, 2),
                "Volume Score": round(rel_volume_norm, 1),
                "Setup Score": round(setup_score, 1),
                "Composite": round(setup_score, 1),
                "State": state,
                "State Rank": state_rank,
                "Tier": state,
                "Tier Rank": state_rank,
                "Breakout": status.is_breakout,
                "Active": status.is_active,
                "Extended": status.is_extended,
                "Near Pivot": near_pivot,
                "Breakout Age": status.breakout_age,
                "Extension %": None if status.extension_pct is None else round(status.extension_pct, 2),
                "Close Position": None if status.close_position is None else round(status.close_position, 2),
                "VCP Label": vcp.label,
                "Pocket Pivot": pocket,
                "Ants": ants,
                "HTF": htf,
                "Bearish Reversal": reversal,
                "Risk Flag": risk_flags(reversal, pivot_distance_pct, rel_vol, trend.passed, status),
                "Patterns": ", ".join(patterns) if patterns else "-",
            }
        )

    leaderboard = pd.DataFrame(rows)
    if leaderboard.empty:
        return leaderboard, market_state
    return leaderboard.sort_values(["State Rank", "Setup Score", "RS"], ascending=False).reset_index(drop=True), market_state


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
    chart_pivot, _ = effective_pivot(df, vcp)
    if chart_pivot is not None:
        fig.add_hline(y=chart_pivot, line_width=1.3, line_dash="dash", line_color="#673ab7", row=1, col=1)
        fig.add_annotation(
            x=chart_df.index[-1],
            y=chart_pivot,
            text=f"{vcp.label} Pivot {chart_pivot:.2f}",
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

    rel_vol = relative_volume(df)
    reversal = bearish_one_day_reversal(df)
    trend = trend_template(df, rs_rating=99)
    status = breakout_status(df, chart_pivot, rel_vol, reversal, trend.passed)
    state_colors = {
        "Setup": "#1a73e8",
        "Breakout": "#673ab7",
        "Active": "#0f9d58",
        "Extended": "#f9ab00",
        "Risk": "#b3261e",
        "Watch": "#5f6368",
    }
    state_color = state_colors.get(status.state, "#5f6368")
    if status.breakout_age is not None and 0 <= status.breakout_age < len(chart_df):
        breakout_date = chart_df.index[-status.breakout_age - 1]
        fig.add_vline(x=breakout_date, line_width=1.2, line_dash="dot", line_color="#673ab7", row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=[chart_df.index[-1]],
            y=[chart_df["Close"].iloc[-1]],
            mode="markers+text",
            name=f"State: {status.state}",
            marker=dict(color=state_color, size=12, symbol="diamond"),
            text=[status.state],
            textposition="top center",
        ),
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


def pattern_tokens(patterns: str) -> list[str]:
    return [token.strip() for token in str(patterns).split(",") if token.strip() and token.strip() != "-"]


def filter_miss_reason(row: pd.Series, rs_cutoff: int, state_filter: list[str], required_patterns: list[str]) -> str:
    misses = []
    if int(row["RS"]) < rs_cutoff:
        misses.append(f"RS<{rs_cutoff}")
    state = str(row["State"])
    if state_filter and state not in state_filter:
        misses.append(f"State not selected")
    tokens = pattern_tokens(str(row["Patterns"]))
    for pattern in required_patterns:
        if pattern not in tokens:
            misses.append(f"No {pattern}")
    return ", ".join(misses) if misses else "-"


def prepare_display_frames(
    leaderboard: pd.DataFrame,
    meta: pd.DataFrame,
    rs_cutoff: int,
    state_filter: list[str],
    required_patterns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if leaderboard.empty or "Ticker" not in leaderboard.columns:
        return leaderboard.copy(), leaderboard.copy()
    display_base = leaderboard.merge(meta, on="Ticker", how="left")
    display_base["Name"] = display_base["Name"].fillna("")
    display_base["Sector"] = display_base["Sector"].fillna("")
    display_base["Filter Miss"] = display_base.apply(
        lambda row: filter_miss_reason(row, rs_cutoff, state_filter, required_patterns),
        axis=1,
    )
    passing = display_base[display_base["Filter Miss"] == "-"].copy()
    preliminary = display_base[display_base["Filter Miss"] != "-"].copy()
    preliminary = preliminary.sort_values(["Setup Score", "RS"], ascending=False).head(30)
    return passing.reset_index(drop=True), preliminary.reset_index(drop=True)


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

    scan_mode = st.sidebar.selectbox("Scan mode", ["Quick scan", "Full universe"], index=0)
    mode_limit = len(universe_df) if scan_mode == "Full universe" else min(QUICK_SCAN_LIMIT, len(universe_df))
    default_limit = mode_limit if scan_mode == "Full universe" else min(int(config["default_limit"]), mode_limit)
    scan_limit_key = f"max_symbols_{scan_mode.lower().replace(' ', '_')}"
    max_symbols = st.sidebar.number_input(
        "Max symbols to scan",
        min_value=5,
        max_value=max(mode_limit, 5),
        value=default_limit,
        step=10,
        key=scan_limit_key,
    )
    rs_cutoff = st.sidebar.slider("Minimum RS", min_value=1, max_value=99, value=40)
    state_options = ["Setup", "Breakout", "Active", "Watch", "Extended", "Risk"]
    state_filter = st.sidebar.multiselect("State filter", state_options, default=["Setup", "Breakout", "Active"])
    required_patterns = st.sidebar.multiselect(
        "Required pattern",
        ["VCP", "VCP Candidate", "Pocket Pivot", "Ants", "HTF", "Breakout", "Bearish Reversal"],
        default=[],
    )
    force_refresh = st.sidebar.button("Refresh market data", width="stretch")

    selected_universe = universe_df.head(int(max_symbols)).copy()
    tickers = selected_universe["symbol"].tolist()
    tickers_with_benchmark = list(dict.fromkeys([*tickers, benchmark]))
    st.sidebar.caption(f"Universe symbols: {len(tickers)} | Benchmark: {benchmark}")
    st.sidebar.caption("RS is ranked within the loaded scan universe. Use Full universe for broader percentile context.")

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
        if result.stale_cache:
            source = f"{source} + stale cache"
        status.update(label=f"Loaded {len(result.prices)} symbols from {source}", state="complete")
    progress_bar.empty()

    if not result.prices:
        st.error("No price data was loaded. Try a smaller universe, different period, or refresh later.")
        if result.failures:
            st.warning("Failed symbols: " + ", ".join(result.failures[:30]))
        if result.invalid_tickers:
            st.warning("Invalid symbols: " + ", ".join(result.invalid_tickers[:30]))
        return

    long_prices = prices_to_long(result.prices)
    leaderboard, market_state_data = cached_scan(long_prices, benchmark)
    market_state = MarketState(**market_state_data)
    render_market_badge(market_state.status, market_state.color, market_state.message)
    st.sidebar.metric("Distribution Days", market_state.distribution_days)
    st.sidebar.metric("Follow-through Day", "Yes" if market_state.follow_through_day else "No")
    st.sidebar.metric("Loaded Symbols", len(result.prices))
    st.sidebar.metric("Failed Symbols", len(result.failures))
    st.sidebar.metric("Excluded Symbols", len(result.invalid_tickers))

    if leaderboard.empty:
        st.warning("Not enough price history to scan this universe.")
        return

    rs_valid_count = int(leaderboard["RS Valid"].sum()) if "RS Valid" in leaderboard else 0
    rs_total_count = int(len(leaderboard))
    rs_min_bars = int(leaderboard["RS Bars"].min()) if "RS Bars" in leaderboard and not leaderboard.empty else 0
    rs_median = float(leaderboard["RS"].median()) if "RS" in leaderboard and not leaderboard.empty else np.nan
    st.sidebar.metric("RS Valid", f"{rs_valid_count}/{rs_total_count}")
    st.sidebar.metric("RS Universe", rs_total_count)
    if rs_valid_count < rs_total_count:
        st.warning(
            f"RS coverage warning: {rs_valid_count}/{rs_total_count} symbols have enough bars "
            f"(minimum required: {RS_MIN_BARS}, current minimum loaded: {rs_min_bars})."
        )

    meta = selected_universe.rename(columns={"symbol": "Ticker", "name": "Name", "sector": "Sector"})
    display_df, preliminary_df = prepare_display_frames(leaderboard, meta, int(rs_cutoff), state_filter, required_patterns)

    st.subheader("Leaderboard")
    state_filter_label = ", ".join(state_filter) if state_filter else "All"
    st.caption(
        f"Active filters: RS >= {rs_cutoff} | States: {state_filter_label} | "
        f"Required patterns: {', '.join(required_patterns) if required_patterns else 'None'} | "
        f"RS scope: {rs_total_count} loaded symbols"
    )
    with st.expander("State guide", expanded=False):
        st.markdown(
            """
            - **Setup**: Below or near pivot by 0-5%, trend/RS filters are acceptable, and not already extended.
            - **Breakout**: Crossed above pivot today from below, with relative volume >= 1.2x and a close in the upper half of the daily range.
            - **Active**: Already above pivot after a valid breakout, within +8% of pivot, and holding above short moving averages.
            - **Extended**: More than +8% above pivot or materially stretched from short moving averages.
            - **Risk**: Bearish reversal, lost pivot, weak breakout failure, below 50SMA, or weak trend warning.
            - **Watch**: Useful candidate context, but not actionable under the current setup/breakout rules.
            """
        )
    if result.failures:
        st.warning(f"{len(result.failures)} symbols failed or returned empty data. First failures: {', '.join(result.failures[:15])}")
    if result.invalid_tickers:
        st.warning(f"{len(result.invalid_tickers)} invalid ticker(s) were excluded before download: {', '.join(result.invalid_tickers[:15])}")
    for warning in result.warnings:
        st.warning(warning)
    st.caption(f"Local cache: `{result.cache_path}`")
    with st.expander("QA diagnostics", expanded=False):
        st.write(
            {
                "market": market,
                "period": period,
                "scan_mode": scan_mode,
                "loaded_symbols": len(result.prices),
                "failed_symbols": len(result.failures),
                "excluded_symbols": len(result.invalid_tickers),
                "rs_valid": f"{rs_valid_count}/{rs_total_count}",
                "rs_scope": f"{rs_total_count} loaded scan symbols",
                "rs_min_bars": rs_min_bars,
                "rs_median": None if pd.isna(rs_median) else round(rs_median, 2),
                "kospi_seed_symbols": int(len(universe_df)) if market == "KOSPI 200" else None,
            }
        )
        if "RS" in leaderboard and not leaderboard.empty:
            st.dataframe(
                leaderboard[["Ticker", "RS", "RS Bars", "RS Valid", "weighted_return"]].head(30)
                if "weighted_return" in leaderboard.columns
                else leaderboard[["Ticker", "RS", "RS Bars", "RS Valid"]].head(30),
                width="stretch",
                hide_index=True,
            )

    columns = [
        "Ticker",
        "Name",
        "Sector",
        "State",
        "Setup Score",
        "RS",
        "RS Bars",
        "Trend Pass",
        "VCP Label",
        "Pivot",
        "Pivot Dist %",
        "Extension %",
        "Breakout Age",
        "Rel Volume",
        "Risk Flag",
        "Patterns",
        "Last Close",
        "Volume",
    ]

    column_config = {
        "Setup Score": st.column_config.ProgressColumn("Setup Score", min_value=0, max_value=100, format="%.1f"),
        "RS": st.column_config.ProgressColumn("RS", min_value=1, max_value=99, format="%d"),
        "RS Bars": st.column_config.NumberColumn("RS Bars", format="%d"),
        "Pivot": st.column_config.NumberColumn("Pivot", format="%.2f"),
        "Last Close": st.column_config.NumberColumn("Last Close", format="%.2f"),
        "Volume": st.column_config.NumberColumn("Volume", format="%d"),
        "Pivot Dist %": st.column_config.NumberColumn("Pivot Dist %", format="%.2f"),
        "Extension %": st.column_config.NumberColumn("Extension %", format="%.2f"),
        "Breakout Age": st.column_config.NumberColumn("Breakout Age", format="%d"),
        "Rel Volume": st.column_config.NumberColumn("Rel Volume", format="%.2f"),
    }

    event = None
    if display_df.empty:
        st.info("No symbols match the current filters. Use the Preliminary Watchlist below for the closest setups.")
    else:
        event = st.dataframe(
            display_df[columns],
            width="stretch",
            hide_index=True,
            height=430,
            on_select="rerun",
            selection_mode="single-row",
            column_config=column_config,
            key="leaderboard_table",
        )

    preliminary_event = None
    if not preliminary_df.empty:
        st.subheader("Preliminary Watchlist")
        st.info(f"Closest candidates that did not pass the active filters (RS >= {rs_cutoff}, States = {state_filter_label}).")
        preliminary_columns = [*columns[:11], "Filter Miss", *columns[11:]]
        preliminary_event = st.dataframe(
            preliminary_df[preliminary_columns],
            width="stretch",
            hide_index=True,
            height=320,
            on_select="rerun",
            selection_mode="single-row",
            column_config={**column_config, "Filter Miss": st.column_config.TextColumn("Filter Miss")},
            key="preliminary_table",
        )

    selected_source = display_df if not display_df.empty else preliminary_df
    selected_rows = []
    if event is not None and event.selection.rows:
        selected_source = display_df
        selected_rows = event.selection.rows
    elif preliminary_event is not None and preliminary_event.selection.rows:
        selected_source = preliminary_df
        selected_rows = preliminary_event.selection.rows

    if selected_source.empty:
        st.warning("No symbols are available for chart review.")
        return

    row_idx = selected_rows[0] if selected_rows else 0
    selected = selected_source.iloc[row_idx]
    ticker = selected["Ticker"]
    selected_df = result.prices.get(ticker)
    if selected_df is None or selected_df.empty:
        st.warning(f"No chart data for {ticker}.")
        return

    st.subheader(f"{ticker} Detail")
    metric_cols = st.columns(5)
    metric_cols[0].metric("Setup Score", f"{selected['Setup Score']:.1f}")
    metric_cols[1].metric("State", selected["State"])
    metric_cols[2].metric("RS", f"{int(selected['RS'])}")
    metric_cols[3].metric("Extension", "-" if pd.isna(selected["Extension %"]) else f"{selected['Extension %']:.2f}%")
    metric_cols[4].metric("Rel Volume", "-" if pd.isna(selected["Rel Volume"]) else f"{selected['Rel Volume']:.2f}x")
    st.caption(f"VCP: {selected['VCP Label']} | Patterns: {selected['Patterns']} | Risk: {selected['Risk Flag']}")
    st.plotly_chart(make_price_chart(ticker, selected_df), width="stretch")


if __name__ == "__main__":
    main()
