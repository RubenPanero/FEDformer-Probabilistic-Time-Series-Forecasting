import argparse
import os
import pandas as pd
import pandas_ta as ta  # noqa: F401
import logging
from data.alpha_vantage_client import AlphaVantageClient
from data.vix_data import VixDataFetcher

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_financial_dataset(symbol, output_dir, use_mock=False):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Obtener OHLCV
    if use_mock:
        # Modo fallback rápido sin API key
        import yfinance as yf

        logger.info(f"Usando yfinance (mock) para {symbol} porque use_mock=True")
        df = yf.download(symbol, period="5y", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.index = df.index.tz_localize(None)
    else:
        av_client = AlphaVantageClient()
        df = av_client.get_daily_data(symbol, outputsize="full")

    if df.empty:
        raise ValueError(f"No se obtuvieron datos para {symbol}.")

    # 2. Obtener VIX y unir (merge)
    start_date = df.index.min().strftime("%Y-%m-%d")
    end_date = df.index.max().strftime("%Y-%m-%d")

    vix_fetcher = VixDataFetcher()
    vix_df = vix_fetcher.get_vix_data(start_date=start_date, end_date=end_date)

    if not vix_df.empty:
        df = df.join(vix_df, how="left")
        df["VIX_Close"] = df["VIX_Close"].ffill()  # Evita look-ahead: no usar bfill
    else:
        df["VIX_Close"] = 0.0  # Fallback si falla VIX

    # 3. Calcular Indicadores Técnicos con pandas-ta
    logger.info("Calculando indicadores técnicos...")

    # Tendencia
    df.ta.sma(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)

    # Momento
    df.ta.rsi(length=14, append=True)
    df.ta.stoch(k=14, d=3, smooth_k=3, append=True)

    # Volatilidad
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)

    # Volumen
    df.ta.obv(append=True)
    df.ta.vwma(length=20, append=True)

    # 4. Sentimiento (Dummy/Placeholder)
    df["Sentiment_Score"] = 0.0

    # Limpiar NAs generados por ventanas móviles
    df.dropna(inplace=True)

    # Guardar
    output_path = os.path.join(output_dir, f"{symbol}_features.csv")
    df.index.name = "date"
    df.to_csv(output_path)
    logger.info(f"Dataset final guardado en {output_path} con forma {df.shape}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Financial Dataset Builder")
    parser.add_argument(
        "--symbol", type=str, default="GOOGL", help="Ticker symbol to download"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data", help="Output directory"
    )
    parser.add_argument(
        "--use_mock",
        action="store_true",
        help="Usa yfinance para OHLCV en lugar de Alpha Vantage",
    )
    args = parser.parse_args()

    build_financial_dataset(args.symbol, args.output_dir, args.use_mock)
