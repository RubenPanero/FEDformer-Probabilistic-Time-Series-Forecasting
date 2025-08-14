# Vanguard-FEDformer: Advanced Probabilistic Time Series Forecasting

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Vanguard-FEDformer es una implementación modular y avanzada del modelo FEDformer (Frequency Enhanced Decomposed Transformer) con capacidades de pronóstico probabilístico, detección de regímenes y análisis de riesgo.

## 🚀 Características Principales

- **Modelo FEDformer Mejorado**: Implementación optimizada con mecanismos de atención Fourier y Wavelet
- **Pronósticos Probabilísticos**: Flujos normalizantes para estimación de incertidumbre
- **Detección de Regímenes**: Algoritmos HMM y GMM para identificar cambios de mercado
- **Análisis de Riesgo**: Métricas VaR, CVaR y simulaciones de Monte Carlo
- **Arquitectura Modular**: Diseño limpio y extensible para investigación y producción
- **Configuraciones Especializadas**: Optimizado para datos financieros y criptomonedas

## 📁 Estructura del Proyecto

```
vanguard_fedformer/
├── __init__.py                    # Paquete principal
├── core/                          # Módulos principales
│   ├── models/                    # Implementaciones del modelo
│   │   ├── fedformer.py          # Modelo FEDformer principal
│   │   ├── flows.py              # Flujos normalizantes
│   │   ├── attention.py          # Mecanismos de atención
│   │   └── components.py         # Capas encoder/decoder
│   ├── data/                      # Manejo de datos
│   │   ├── dataset.py            # Clases de dataset
│   │   ├── preprocessing.py      # Preprocesamiento
│   │   └── regime_detection.py   # Detección de regímenes
│   ├── training/                  # Lógica de entrenamiento
│   │   ├── trainer.py            # Entrenador principal
│   │   ├── losses.py             # Funciones de pérdida
│   │   └── callbacks.py          # Callbacks de entrenamiento
│   └── evaluation/                # Evaluación y análisis
│       ├── metrics.py            # Métricas de evaluación
│       ├── backtesting.py        # Pruebas walk-forward
│       └── risk_analysis.py      # Análisis de riesgo
├── utils/                         # Utilidades
│   ├── config.py                 # Gestión de configuración
│   ├── logging.py                # Sistema de logging
│   └── visualization.py          # Funciones de visualización
├── scripts/                       # Scripts ejecutables
│   ├── train.py                  # Script de entrenamiento
│   ├── evaluate.py               # Script de evaluación
│   └── demo.py                   # Script de demostración
├── configs/                       # Archivos de configuración
│   ├── default.yaml              # Configuración por defecto
│   ├── financial.yaml            # Configuración financiera
│   └── crypto.yaml               # Configuración cripto
├── tests/                         # Suite de pruebas
│   ├── test_models.py            # Pruebas de modelos
│   ├── test_data.py              # Pruebas de datos
│   └── test_training.py          # Pruebas de entrenamiento
├── notebooks/                     # Jupyter notebooks
│   ├── demo.ipynb                # Demostración interactiva
│   ├── tutorial.ipynb            # Guía paso a paso
│   └── advanced_usage.ipynb      # Características avanzadas
├── data/                          # Directorio de datos
│   ├── sample/                   # Datos de ejemplo
│   └── README.md                 # Documentación de datos
└── docs/                          # Documentación
    ├── api/                       # Documentación de API
    ├── tutorials/                 # Guías de usuario
    └── paper/                     # Documentación técnica
```

## 🛠️ Instalación

### Requisitos Previos

- Python 3.8+
- PyTorch 2.0+
- CUDA (opcional, para aceleración GPU)

### Instalación Rápida

```bash
# Clonar el repositorio
git clone https://github.com/yourusername/Vanguard-FEDformer-Advanced-Probabilistic-Time-Series-Forecasting.git
cd Vanguard-FEDformer-Advanced-Probabilistic-Time-Series-Forecasting

# Instalar dependencias
pip install -r requirements.txt

# Instalar en modo desarrollo
pip install -e .
```

## 🚀 Uso Rápido

### 1. Entrenamiento del Modelo

```bash
# Entrenar con configuración por defecto
python vanguard_fedformer/scripts/train.py \
    --data_path data/sample/sp500_sample.csv \
    --config vanguard_fedformer/configs/default.yaml

# Entrenar con configuración financiera
python vanguard_fedformer/scripts/train.py \
    --data_path your_financial_data.csv \
    --config vanguard_fedformer/configs/financial.yaml
```

### 2. Evaluación del Modelo

```bash
# Evaluar modelo entrenado
python vanguard_fedformer/scripts/evaluate.py \
    --model models/vanguard_fedformer.pt \
    --data data/sample/sp500_sample.csv \
    --config vanguard_fedformer/configs/default.yaml
```

### 3. Demostración Interactiva

```bash
# Ejecutar demo
python vanguard_fedformer/scripts/demo.py

# O abrir notebook
jupyter notebook vanguard_fedformer/notebooks/demo.ipynb
```

## 📊 Ejemplo de Uso

```python
from vanguard_fedformer.core.models.fedformer import VanguardFEDformer
from vanguard_fedformer.core.data.dataset import TimeSeriesDataset
from vanguard_fedformer.utils.config import ConfigManager

# Cargar configuración
config = ConfigManager("vanguard_fedformer/configs/financial.yaml")

# Crear modelo
model = VanguardFEDformer(
    d_model=config.model.d_model,
    n_heads=config.model.n_heads,
    n_layers=config.model.n_layers,
    d_ff=config.model.d_ff,
    dropout=config.model.dropout,
    activation=config.model.activation
)

# Crear dataset
dataset = TimeSeriesDataset(
    data_path="your_data.csv",
    sequence_length=config.data.sequence_length,
    prediction_length=config.data.prediction_length,
    batch_size=config.data.batch_size
)

# Entrenar modelo
trainer = VanguardTrainer(model, config)
trainer.train(dataset)
```

## ⚙️ Configuración

### Configuración por Defecto

```yaml
model:
  type: "fedformer"
  d_model: 512
  n_heads: 8
  n_layers: 6
  d_ff: 2048
  dropout: 0.1
  activation: "gelu"

data:
  sequence_length: 96
  prediction_length: 24
  batch_size: 32
  num_workers: 4

training:
  epochs: 100
  learning_rate: 0.0001
  weight_decay: 0.01
  patience: 10
```

### Configuraciones Especializadas

- **`financial.yaml`**: Optimizado para datos financieros diarios
- **`crypto.yaml`**: Optimizado para datos de criptomonedas de alta frecuencia

## 🧪 Pruebas

```bash
# Ejecutar todas las pruebas
python -m pytest vanguard_fedformer/tests/

# Ejecutar pruebas específicas
python -m pytest vanguard_fedformer/tests/test_models.py

# Con cobertura
python -m pytest vanguard_fedformer/tests/ --cov=vanguard_fedformer
```

## 📚 Documentación

- **API Reference**: `docs/api/`
- **Tutoriales**: `docs/tutorials/`
- **Notebooks**: `notebooks/`
- **Configuración**: `configs/`

## 🔬 Características Avanzadas

### Detección de Regímenes

```python
from vanguard_fedformer.core.data.regime_detection import RegimeDetector

detector = RegimeDetector(
    n_regimes=3,
    method="hmm",
    volatility_threshold=0.02
)

regimes = detector.detect(data)
```

### Pronósticos Probabilísticos

```python
from vanguard_fedformer.core.models.flows import NormalizingFlow

flow = NormalizingFlow(
    n_flows=4,
    hidden_dim=128,
    flow_type="real_nvp"
)

# Generar múltiples pronósticos
forecasts = []
for _ in range(100):
    forecast = model(x)
    forecasts.append(forecast)

# Calcular intervalos de confianza
mean_forecast = torch.stack(forecasts).mean(0)
std_forecast = torch.stack(forecasts).std(0)
```

### Análisis de Riesgo

```python
from vanguard_fedformer.core.evaluation.risk_analysis import RiskSimulator

risk_sim = RiskSimulator(config)
risk_metrics = risk_sim.analyze(backtest_results)

print(f"VaR (95%): {risk_metrics['var_95']:.4f}")
print(f"CVaR (95%): {risk_metrics['cvar_95']:.4f}")
print(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.4f}")
```

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 🙏 Agradecimientos

- **FEDformer**: Paper original sobre Frequency Enhanced Decomposed Transformer
- **PyTorch**: Framework de deep learning
- **Comunidad**: Todos los contribuidores y usuarios

## 📞 Contacto

- **Proyecto**: [GitHub Issues](https://github.com/yourusername/Vanguard-FEDformer-Advanced-Probabilistic-Time-Series-Forecasting/issues)
- **Email**: your.email@example.com

---

**⭐ Si este proyecto te es útil, por favor dale una estrella en GitHub!**
