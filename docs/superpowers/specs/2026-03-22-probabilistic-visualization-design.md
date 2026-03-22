# Probabilistic Visualization Design

**Date:** 2026-03-22
**Status:** Approved, pending implementation

## Goal

Visualizar predicciones probabilísticas generadas por `python3 -m inference` con fan charts y calibration plots.

## Decisions

| Decision | Choice | Alternatives considered |
|----------|--------|------------------------|
| Fan chart style | **Clásico** (banda p10-p90 + p50 + GT) | Multi-banda gradual, Spaghetti+banda |
| Data scope | **Rolling overview + zoom última ventana** | Solo última ventana, ventana seleccionable |
| Calibration | **Reliability diagram + PIT histogram** | Solo reliability, solo PIT |
| Code location | **`utils/visualization.py`** funciones puras | Paquete `visualization/`, script standalone |

## Architecture

### File: `utils/visualization.py`

Funciones puras: reciben DataFrames → devuelven `matplotlib.figure.Figure`.

```python
def plot_fan_chart(df: pd.DataFrame, ticker: str) -> Figure:
    """Fan chart con 2 subplots: rolling overview + zoom última ventana."""

def plot_calibration(df: pd.DataFrame, ticker: str) -> Figure:
    """Calibration con 2 subplots: reliability diagram + PIT histogram."""
```

### Integration: `inference/__main__.py`

Flag `--plot` en el CLI de inferencia:
```bash
python3 -m inference --ticker NVDA --csv data/NVDA_features.csv --plot
```

Al finalizar la predicción, genera los gráficos y los guarda como PNG.

### Output

- `results/fan_chart_{ticker}.png`
- `results/calibration_{ticker}.png`
- O ruta custom con `--output-dir`

## Plot 1 — Fan Chart Clásico

**2 subplots verticales:**

### Superior: Rolling Overview
- Para cada timestep, tomar `step=0` de su ventana correspondiente
- Genera serie temporal continua con:
  - Banda p10-p90 (fill semi-transparente azul)
  - Línea p50 mediana (azul sólido)
  - Ground truth (naranja punteada)
- Vista panorámica de calidad predictiva sobre todo el histórico

### Inferior: Zoom Última Ventana
- Los 20 pasos (pred_len) de la última ventana
- Misma visualización: banda p10-p90 + p50 + ground truth
- Caso de uso: "¿qué predice el modelo ahora?"

## Plot 2 — Calibration

**1×2 subplots:**

### Izquierda: Reliability Diagram
- Eje X: niveles nominales (0.1, 0.5, 0.9)
- Eje Y: cobertura empírica (fracción de GT ≤ cuantil)
- Línea diagonal = calibración perfecta
- Puntos por encima = sobre-confianza, por debajo = sub-confianza

### Derecha: PIT Histogram
- Probability Integral Transform
- Histograma de rangos empíricos del ground truth dentro de la distribución predictiva
- Uniforme = bien calibrado
- Forma de U = sub-dispersión (intervalos demasiado estrechos)
- Campana = sobre-dispersión (intervalos demasiado anchos)

## Input Data Format

CSV de inferencia (`results/inference_{ticker}.csv`):
```
window,step,mean_Close,gt_Close,p10_Close,p50_Close,p90_Close
0,0,0.004520,...,-0.039627,0.007764,0.049166
```

- 1609 ventanas × 20 pasos = 32180 filas (NVDA)
- Valores en espacio de log-returns

## Style

- matplotlib, tema oscuro (consistente con plot existente en main.py)
- `MPLBACKEND=Agg` compatible (headless)
- Colores: azul `#4fc3f7` para predicciones, naranja `#ff7043` para ground truth

## Testing

- Tests unitarios con DataFrames sintéticos (sin modelo real)
- Verificar que funciones devuelven Figure válido
- Verificar output PNG generado
- Test de integración: `--plot` flag en CLI

## Architectural Decisions (approved 2026-03-22)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| PIT computation | Interpolación lineal entre p10/p50/p90 | Funciones puras, sin cambios en export CSV; eje etiquetado "approx" |
| Escala ejes | % change via `np.expm1(log_return) * 100` | Más intuitivo que log-returns; no requiere precio base |
| `--output-dir` | Default `results/`, configurable vía CLI | Coexiste con `--output` (CSV) sin conflicto |

## Dependencies

- matplotlib (ya instalado, v3.10.8)
- No requiere dependencias nuevas
