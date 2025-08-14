# Data Directory

Este directorio contiene los datos utilizados por Vanguard-FEDformer.

## Estructura

```
data/
├── sample/           # Datos de ejemplo para demostración
│   └── sp500_sample.csv  # Datos sintéticos del S&P 500
└── README.md         # Este archivo
```

## Datos de Ejemplo

### sp500_sample.csv

Archivo de datos sintéticos que simula precios del S&P 500 con las siguientes características:

- **Formato**: CSV con valores separados por comas
- **Período**: 1000 observaciones
- **Características**: 5 columnas (Open, High, Low, Close, Volume)
- **Patrones**: Incluye tendencia, estacionalidad y cambios de régimen
- **Uso**: Para demostración y pruebas del modelo

## Formato de Datos

Vanguard-FEDformer espera datos en el siguiente formato:

### Entrada
- **Formato**: CSV, Parquet, o NumPy arrays
- **Estructura**: Cada fila es una observación temporal
- **Columnas**: Características de la serie temporal
- **Orden**: Cronológico (más antiguo a más reciente)

### Ejemplo de CSV
```csv
timestamp,open,high,low,close,volume
2023-01-01,100.0,101.5,99.5,100.8,1000000
2023-01-02,100.8,102.0,100.2,101.5,1200000
...
```

## Preprocesamiento

Los datos se preprocesan automáticamente usando la clase `DataPreprocessor`:

1. **Normalización**: Escalado de características
2. **Ventanas deslizantes**: Creación de secuencias de entrada
3. **División temporal**: Train/validation/test splits
4. **Manejo de valores faltantes**: Interpolación o eliminación

## Configuración

La configuración de datos se especifica en los archivos YAML:

```yaml
data:
  sequence_length: 96      # Longitud de secuencia de entrada
  prediction_length: 24    # Longitud de predicción
  batch_size: 32          # Tamaño de batch
  num_workers: 4          # Workers para DataLoader
  features: ["open", "high", "low", "close", "volume"]
```

## Agregar Nuevos Datos

Para agregar nuevos conjuntos de datos:

1. Coloca el archivo en el directorio apropiado
2. Actualiza la configuración si es necesario
3. Ejecuta el preprocesamiento: `python scripts/preprocess.py --data_path path/to/data`
4. Verifica la compatibilidad ejecutando las pruebas

## Notas Importantes

- Los datos deben estar ordenados cronológicamente
- Evita datos con muchos valores faltantes
- Para series financieras, considera usar retornos logarítmicos
- Los datos de alta frecuencia pueden requerir configuración especial