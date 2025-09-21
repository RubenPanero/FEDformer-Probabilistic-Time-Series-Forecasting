# Preflight

## 1. Mapa del proyecto
- `main.py`: CLI principal para entrenar o evaluar configuraciones completas de FEDformer y flujos probabilísticos.【F:AGENTS.md†L5-L18】
- `config.py`: define `FEDformerConfig` y las utilidades de configuración que consumen los módulos del modelo.【F:AGENTS.md†L5-L18】
- `models/`: componentes de modelado; `models/fedformer.py` contiene la clase `Flow_FEDformer`, núcleo del modelo con normalizing flows y lógica de inferencia.【F:AGENTS.md†L5-L10】【F:models/fedformer.py†L33-L186】
- `training/`: bucles de entrenamiento, programadores y checkpoints reutilizados por `main.py`.【F:AGENTS.md†L5-L10】
- `utils/`: utilidades compartidas (métricas, I/O, semillas) invocadas en entrenamiento y evaluación.【F:AGENTS.md†L5-L12】
- `data/`, `reports/` (ignorados en git) y `tests/`: administración de datasets, reportes y la suite de pruebas respetada por PyTest.【F:AGENTS.md†L5-L12】

## 2. Dependencias para ejecutar Pylint
1. Instalar la base del proyecto: `pip install -r requirements.txt` trae PyTorch (`torch>=2.0.0`) y librerías científicas fundamentales (NumPy, pandas, SciPy, scikit-learn) necesarias para que Pylint resuelva los imports del código.【F:requirements.txt†L1-L25】
2. Paquetes opcionales pero recomendados (`matplotlib`, `seaborn`, `wandb`, `tqdm`, `PyYAML`) evitan falsos positivos de "import-error" cuando los módulos de visualización o logging se inspeccionan estáticamente.【F:requirements.txt†L12-L21】
3. Añadir herramientas de linting: `pylint>=3.0.0` (no incluida en `requirements.txt`) y, si se usan fixtures de PyTest en análisis estático, `pylint-pytest` para reconocer marcadores y fixtures personalizados.【F:requirements.txt†L23-L25】
4. Para proyectos con PyTorch es habitual complementar con `typing-extensions` (instalada transitivamente por `torch`) y definir `PYTHONPATH` hacia la raíz para que Pylint encuentre los paquetes locales (`models`, `training`, etc.).【F:requirements.txt†L4-L21】【F:AGENTS.md†L5-L12】

## 3. Reglas de Pylint potencialmente problemáticas y mitigaciones
- `too-many-instance-attributes`/`too-many-public-methods` en `Flow_FEDformer` debido a la cantidad de capas y proyectores registrados en `__init__`. Se puede encapsular subconjuntos en `nn.ModuleDict` o factorizar creadores auxiliares sin modificar el flujo de datos central.【F:models/fedformer.py†L33-L95】
- `too-many-locals` y `too-many-branches` en `forward`, que mezcla preparación de embeddings, checkpointing y proyección de flujos. Dividir en métodos privados (`_apply_regime_embeddings`, `_run_flow_projection`) mantiene la lógica pero reduce la complejidad reportada.【F:models/fedformer.py†L122-L185】
- `broad-except` por el `except Exception` que envuelve el checkpointing; reemplazarlo por excepciones concretas de PyTorch (`RuntimeError`, `ValueError`) conserva el fallback sin suprimir errores no previstos.【F:models/fedformer.py†L145-L159】
- `duplicate-code`/`duplicate-string-formatting` por el bloque duplicado de cabecera e imports al inicio del módulo; basta eliminar el duplicado para satisfacer la regla sin tocar el comportamiento.【F:models/fedformer.py†L1-L30】
- `unused-import` provocado por tipos no utilizados como `Optional`; remover el import o emplearlo en anotaciones mantiene intacta la ejecución.【F:models/fedformer.py†L8-L25】
- `no-member` al acceder a `torch.utils.checkpoint.checkpoint`; puede mitigarse definiendo `generated-members=torch.*` en `.pylintrc` o importando explícitamente `from torch.utils import checkpoint` antes de usarlo, evitando cambios funcionales.【F:models/fedformer.py†L145-L154】

## 4. Valoración y plan
- **Factibilidad:** Muy alta (≈97 %). El repositorio ya delimita responsabilidades y depende de librerías maduras; las mitigaciones listadas no afectan la semántica del modelo.
- **Próximas fases:**
  1. Configurar entorno virtual y aplicar `pip install -r requirements.txt && pip install pylint pylint-pytest`.
  2. Elaborar (o ajustar) `.pylintrc` incluyendo tolerancias específicas (por ejemplo, umbrales de complejidad y `generated-members`).
  3. Refactorizar mínimamente los módulos señalados para cumplir las reglas sin alterar cálculos y validar con `pytest -q`.
  4. Integrar Pylint al pipeline (makefile/CI) y documentar los comandos en `README.md` para el equipo.
