# Plan Estratégico: Entrenamiento de FEDformer con Datos Financieros

## 1. Selección y Preparación del Dataset
- **Fuente de Datos:** Obtener OHLCV (Open, High, Low, Close, Volume) de activos (ej. Yahoo Finance, Binance, Interactive Brokers).
- **Temporalidad (Timeframe):** Seleccionar una temporalidad coherente con el horizonte de inversión (ej. 1h, 4h, 1D).
- **Indicadores Técnicos (Features):**
  - *Tendencia:* Medias Móviles (SMA, EMA), MACD.
  - *Momento:* RSI, Estocástico.
  - *Volatilidad:* Bandas de Bollinger, ATR (Average True Range).
  - *Volumen:* OBV, VWAP.
- **Limpieza y Estacionariedad:** Manejar valores nulos, outliers y evaluar si es necesario transformar los precios a retornos logarítmicos para mejorar la estacionariedad.

## 2. Configuración del Pipeline de FEDformer (`data/preprocessing.py`)
- **Variables Objetivo (`--targets`):** Generalmente el precio de cierre (`Close`) o los retornos futuros.
- **Manejo de Escalas:** Asegurar que el `RobustScaler` o `StandardScaler` en `preprocessing.py` maneje adecuadamente las diferencias de magnitud entre precios (ej. Bitcoin a 60k) e indicadores (ej. RSI de 0 a 100).
- **Variables Temporales (`--date-col`):** Incluir características del calendario (día de la semana, mes) si existe estacionalidad.

## 3. Definición de la Ventana Temporal (Hyperparameters)
- **`seq-len` (Ventana de Observación):** Cantidad de periodos históricos que el modelo ve para hacer una predicción (ej. 96 periodos).
- **`label-len` (Solapamiento de Decodificación):** Historial reciente visible para el decodificador (ej. 48 periodos).
- **`pred-len` (Horizonte Predictivo):** Cuántos periodos hacia el futuro queremos predecir (ej. 24 periodos). En finanzas, horizontes más cortos suelen ser más precisos.

## 4. Ejecución del Entrenamiento (Walk-Forward)
- **Backtesting Realista:** Utilizar el `WalkForwardTrainer` configurado en `main.py` (parámetro `--splits`). Esto es crítico en series de tiempo financieras para evitar *data leakage* (ver el futuro) y evaluar cómo el modelo se adapta a regímenes cambiantes de mercado.
- **Aceleración (AMP):** Asegurarse de tener `--use-checkpointing` activado si los datasets con muchos indicadores consumen mucha VRAM.

## 5. Evaluación Centrada en el Riesgo
- **Métricas Más Allá del Error Cuadrático:** En lugar de solo mirar el MSE, observar las salidas estocásticas.
- **Simulación (`simulations/portfolio.py` y `risk.py`):**
  - *Ratio de Sharpe / Sortino:* Evaluar si la predicción direccional genera retornos ajustados al riesgo.
  - *VaR y CVaR:* Utilizar el `RiskSimulator` para medir el riesgo de cola de las predicciones probabilísticas.

## 6. Siguientes Pasos (Accionables)
1. **Script de Ingesta:** Crear un script sencillo para descargar datos (ej. con la librería `yfinance`) y calcular indicadores técnicos (ej. con `pandas-ta`).
2. **Prueba de Humo (Smoke Test):** Correr `main.py` con un subconjunto pequeño de este nuevo dataset financiero para verificar que el pipeline procese correctamente todas las características.
