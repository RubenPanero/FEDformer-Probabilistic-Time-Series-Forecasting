---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7, 8]
inputDocuments:
  - CLAUDE.md
  - checkpoints/model_registry.json
  - docs/superpowers/plans/2026-03-24-phase6-hp-optimization.md
  - optuna_studies/kaggle_nvda_phase6/export/best_trial.json
  - optuna_studies/kaggle_nvda_phase6/export/nvda_trials_phase6.csv
workflowType: "architecture"
project_name: "FEDformer-Probabilistic-Time-Series-Forecasting"
user_name: "ruben"
date: "2026-03-25"
status: "complete"
sourceMode: "post-Kaggle Phase 6 P2 — Dual Model Strategy"
---

# Architecture Decision Document
## Phase 6 Post-Kampagnen — Estrategia Dual Model NVDA

Este documento define la arquitectura de implementación de las dos opciones derivadas
del análisis de la campaña Kaggle Phase 6 P2 (30 trials, 2×T4, NVDA).

---

## Contexto de la Decisión

### Hallazgos de la campaña Kaggle
- 27/30 trials completados, 3 podados.
- Mejor trial global (#4): **Sharpe=1.3469**, `pred_len=4`, `seq_len=128`, `batch=32`,
  `clip=0.3`, `e_layers=1`, `d_layers=2`, `n_flow=6`, `dropout=0.2`.
- Mejor trial con `pred_len=20` (trials 5,12,13,14,22): **Sharpe=1.179**, `seq=128`,
  `batch=32`, `clip=0.3`, `e_layers=3`, `d_layers=2`, `n_flow=2`, `dropout=0.2`.
- Trial 0 (canónico enqueued en T4): Sharpe=0.865 vs. local 0.990 — delta estocástico
  esperado por diferencias de entorno CUDA/PyTorch.

### Opciones identificadas
- **Opción A**: Re-entrenar modelo canónico NVDA con `pred_len=20` optimizado (+18.9% Sharpe).
- **Opción B**: Crear modelo experimental "short-horizon" con `pred_len=4` en rama separada.

### Estado actual del registry (`checkpoints/model_registry.json`)
```
specialists:
  NVDA: checkpoint=nvda_canonical.pt  pred_len=20  Sharpe=0.990
  GOOGL: checkpoint=googl_canonical.pt  pred_len=20  Sharpe=0.737
```

La clave de registro es el ticker (`NVDA`, `GOOGL`). El sistema admite actualmente
un único especialista por ticker.

---

## Decisiones Arquitecturales

### Decisión 1: Opción A no requiere cambios de código

**Contexto:** Los parámetros optimizados de `pred_len=20` son completamente configurables
mediante flags CLI ya existentes. El `_save_canonical_specialist` actualiza automáticamente
el registry si el nuevo Sharpe supera al actual (Guardia 2).

**Decisión:** Opción A se implementa en `main` con un nuevo run de entrenamiento.
No se crea rama ni se modifica ningún módulo.

**Implicación:** Si el Sharpe del re-entrenamiento local queda por debajo de 0.990 por
variabilidad estocástica, se debe hacer reset manual del Sharpe en el registry antes de
volver a lanzar (gotcha documentado en CLAUDE.md).

**Comando canónico Opción A:**
```bash
MPLBACKEND=Agg python3 main.py --csv data/NVDA_features.csv --targets "Close" \
  --seq-len 128 --pred-len 20 --batch-size 32 --splits 4 \
  --return-transform log_return --metric-space returns \
  --gradient-clip-norm 0.3 --seed 7 \
  --e-layers 3 --d-layers 2 --n-flow-layers 2 --flow-hidden-dim 64 \
  --label-len 48 --dropout 0.2 \
  --save-results --save-canonical --no-show
```

---

### Decisión 2: Opción B usa clave de registry `NVDA_SH`

**Contexto:** El registry indexa por clave de ticker (`specialists[key]`). Para soportar
un segundo modelo NVDA con `pred_len=4` sin colisionar con el canónico, necesitamos una
clave distinta. Alternativas evaluadas:

| Alternativa | Ventajas | Desventajas |
|---|---|---|
| Añadir `--model-alias` a `main.py` | Limpio, extensible | Requiere código nuevo + test + argparse |
| Usar `NVDA_SH` como alias via un CSV renombrado/symlink | Sin código | Hack frágil, confuso |
| Patch manual del registry tras entrenamiento | Sin código, explícito | Proceso manual propenso a errores |
| Extender schema con campo `variants` | Diseño correcto a largo plazo | Scope amplio, no justificado aún |

**Decisión:** Añadir `--model-alias` a `main.py` en la rama de Opción B.

**Justificación:**
- El patrón `if args.X is not None: config.X = args.X` con `default=None` es exactamente
  el patrón ya en uso en la codebase.
- No requiere cambios al schema del registry — el alias reemplaza simplemente el valor
  derivado del CSV basename.
- Es testeable: `test_finetune.py` debe reflejar el nuevo flag.
- Permite inferencia con `--ticker NVDA_SH` de forma limpia.

**Checkpoint naming para Opción B:**
- Checkpoint: `checkpoints/nvda_sh_canonical.pt`
- Preprocessing: `checkpoints/nvda_sh_preprocessing/`
- Registry key: `"NVDA_SH"`

**Comando canónico Opción B:**
```bash
MPLBACKEND=Agg python3 main.py --csv data/NVDA_features.csv --targets "Close" \
  --model-alias NVDA_SH \
  --seq-len 128 --pred-len 4 --batch-size 32 --splits 4 \
  --return-transform log_return --metric-space returns \
  --gradient-clip-norm 0.3 --seed 7 \
  --e-layers 1 --d-layers 2 --n-flow-layers 6 --flow-hidden-dim 64 \
  --label-len 48 --dropout 0.2 \
  --save-results --save-canonical --no-show
```

**Inferencia Opción B:**
```bash
python3 -m inference --ticker NVDA_SH --csv data/NVDA_features.csv
```

---

### Decisión 3: Estrategia de ramas

| Tarea | Rama | Justificación |
|---|---|---|
| Opción A: re-entrenamiento | `main` | Solo run de datos, sin código |
| Opción B: `--model-alias` + test + run | `feat/short-horizon-nvda` | Cambio de código experimental |

**Regla de merge para la rama de Opción B:**
La rama se mergea a `main` SOLO si se cumplen todos estos criterios:
1. Tests pasan (350 fast) incluyendo el nuevo test de `--model-alias`.
2. El re-entrenamiento local con pred_len=4 produce Sharpe > 0.5 (mínimo útil).
3. Se documenta un caso de uso concreto para el horizonte de 4 días.
4. La inferencia CLI con `--ticker NVDA_SH` funciona end-to-end.

---

### Decisión 4: Protocolo de comparación entre opciones

**Problema:** `pred_len=4` y `pred_len=20` son horizontes de predicción distintos.
Sus Sharpes de CV (1.347 vs 1.179) **no son directamente comparables** — un horizonte
de 4 días es intrínsecamente más fácil de predecir que uno de 20 días.

**Protocolo de evaluación independiente:**

| Métrica | Opción A (pred_len=20) | Opción B (pred_len=4) |
|---|---|---|
| Sharpe walk-forward CV | ~1.179 (Kaggle) | ~1.347 (Kaggle) |
| Sharpe re-entrenamiento local | A medir | A medir |
| MaxDD objetivo | < −55% | < −60% (más agresivo) |
| coverage_80 objetivo | ≥ 80% | ≥ 80% |
| Caso de uso principal | Swing trading 20d | Day-to-week trading 4d |

**No se debe afirmar que Opción B "supera" a Opción A** basándose solo en Sharpe de CV.
Son productos diferentes.

---

### Decisión 5: Guardia de la evaluación estocástica

El run local puede producir Sharpe diferente al de la campaña Kaggle (el canónico
actual obtuvo 0.865 en T4 vs 0.990 local con los mismos params).

**Regla:** Si el re-entrenamiento local de Opción A produce Sharpe < 0.990, NO se
actualiza el canónico automáticamente. Se compara el run local 3× (seeds 7, 42, 123)
y se usa la mediana antes de decidir.

---

## Archivos Afectados

### Opción A (rama `main`)
- **Solo datos**: No se modifica ningún archivo de código.
- `checkpoints/model_registry.json` — actualizado automáticamente vía `--save-canonical`.
- `checkpoints/nvda_canonical.pt` — sobreescrito.
- `checkpoints/nvda_preprocessing/` — sobreescrito.
- `CLAUDE.md`, `memory/MEMORY.md` — actualizar tabla de modelos canónicos.

### Opción B (rama `feat/short-horizon-nvda`)
- **Modificar**: `main.py` — añadir `--model-alias` al argparse.
- **Modificar**: `tests/test_finetune.py` — añadir `model_alias=None` al `argparse.Namespace`.
- **Añadir test**: verificar que `--model-alias NVDA_SH` produce `specialist_key="NVDA_SH"`.
- `checkpoints/model_registry.json` — nueva entrada `NVDA_SH`.
- `checkpoints/nvda_sh_canonical.pt` — nuevo checkpoint.
- `checkpoints/nvda_sh_preprocessing/` — nuevos artifacts.
- `CLAUDE.md`, `memory/MEMORY.md` — documentar modelo experimental.

---

## Plan de Implementación Paralela

### Workstream A — Re-entrenamiento canónico NVDA (main, ~10 min)

```
[A1] Verificar que el run de Opción A puede lanzarse (params OK, sin código)
[A2] Lanzar re-entrenamiento con nuevos HPs (pred_len=20 optimizado)
[A3] Verificar Sharpe resultante vs 0.990 (Guardia 2)
     → Si Sharpe > 0.990: registry actualiza automáticamente ✓
     → Si Sharpe < 0.990: reset Sharpe en registry + relanzar
[A4] Validar inference end-to-end: python3 -m inference --ticker NVDA ...
[A5] Actualizar CLAUDE.md + MEMORY.md con nuevos resultados
[A6] Commit en main
```

### Workstream B — Modelo short-horizon (rama feat/short-horizon-nvda)

```
[B1] git checkout -b feat/short-horizon-nvda
[B2] TDD — escribir test RED para --model-alias en test_finetune.py
[B3] Añadir --model-alias a main.py (argparse + apply_flags)
     → Convención: if args.model_alias is not None: specialist_key = args.model_alias
[B4] Actualizar argparse.Namespace en test_finetune.py (model_alias=None)
[B5] Ejecutar tests → GREEN (350 fast)
[B6] Lanzar re-entrenamiento con pred_len=4 + --model-alias NVDA_SH (~10 min)
[B7] Verificar que registry crea entrada "NVDA_SH" correctamente
[B8] Validar inference: python3 -m inference --ticker NVDA_SH --csv data/NVDA_features.csv
[B9] Evaluar criterios de merge (Sharpe > 0.5, caso de uso definido)
[B10] Si criterios OK: merge a main (vía push-safe)
      Si no: dejar rama como experimental
```

### Dependencias y paralelismo

```
A1 ──► A2 ──► A3 ──► A4 ──► A5 ──► A6
B1 ──► B2 ──► B3 ──► B4 ──► B5 ──► B6 ──► B7 ──► B8 ──► B9 ──► B10

A y B son COMPLETAMENTE INDEPENDIENTES.
Pueden ejecutarse en paralelo (sesiones de terminal separadas).
```

---

## Impacto en Módulos Existentes

### `main.py` — `_save_canonical_specialist`
Actualmente deriva el `specialist_key` del basename del CSV:
```
"NVDA_features.csv" → key = "NVDA"
```

Con `--model-alias`, la lógica debe ser:
```python
specialist_key = args.model_alias if args.model_alias else derive_from_csv(args.csv)
```

Este cambio es localizado y backward-compatible: `default=None` preserva el comportamiento
actual cuando no se pasa el flag.

### `tests/test_model_registry.py`
No requiere cambios — los tests del registry trabajan con claves arbitrarias.

### `inference/__main__.py`
No requiere cambios — ya acepta cualquier key válida via `--ticker`.

### `tests/test_inference.py`
Puede necesitar un test adicional para `--ticker NVDA_SH` si se mergea a main.

---

## Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| Re-entrenamiento A produce Sharpe < 0.990 | Media | Medio | Regla de 3 seeds (Decisión 5) |
| `pred_len=4` no útil en práctica (4 días de horizonte) | Media | Bajo | La rama es experimental, no bloquea |
| `--model-alias` introduce regresión en Guardia 2 del registry | Baja | Alto | Test TDD cubre el path + adversarial review antes de merge |
| Confusión NVDA vs NVDA_SH en inference CLI | Baja | Bajo | Documentar en CLAUDE.md y --help |

---

## Gates de Validación

### Gate A — Opción A lista para producción
- [ ] Re-entrenamiento local Sharpe ≥ 0.990 (o 3-seed median > 0.990)
- [ ] Inference NVDA end-to-end OK
- [ ] `ruff check .` + `ruff format .` limpios (sin código cambiado → trivialmente OK)
- [ ] Commit en main, pusheado

### Gate B — Opción B lista para merge
- [ ] Test `--model-alias` pasa en suite (350 fast)
- [ ] Registry crea entrada `NVDA_SH` correctamente
- [ ] Inference `--ticker NVDA_SH` devuelve predicciones válidas (no ceros)
- [ ] Sharpe de re-entrenamiento local NVDA_SH ≥ 0.5
- [ ] Caso de uso documentado en CLAUDE.md
- [ ] `adversarial-review` ejecutado antes de merge

---

## Próximos Pasos Inmediatos

1. **Ahora mismo** → Lanzar Workstream A (sin código, solo training run).
2. **En paralelo** → Iniciar Workstream B en nueva rama con TDD de `--model-alias`.
3. **Post-A** → Si Sharpe A ≥ 0.990: commit en main. Si no: regla de 3 seeds.
4. **Post-B** → Evaluar criterios de merge; decidir si NVDA_SH va a producción.
