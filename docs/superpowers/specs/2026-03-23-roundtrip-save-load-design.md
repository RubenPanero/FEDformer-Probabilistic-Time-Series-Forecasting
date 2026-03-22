# Spec: Test Roundtrip Save-Load (#9)

## Objetivo

Verificar que el ciclo completo `train → save_canonical → load_specialist → predict`
produce resultados deterministas y no pierde información crítica en la serialización.

## Contexto

Deferred #9 del backlog. El bug de `enc_in/dec_in` (sesión 19, commit `c4331c3`) demostró
que la cadena save→load puede corromper silenciosamente parámetros de arquitectura.
No existe test end-to-end que cubra esta cadena.

## Cadena bajo test

```
1. Flow_FEDformer(config)           → modelo original
2. trainer.save_checkpoint()        → best_model_fold_N.pt
3. preprocessor.save_artifacts()    → schema.json + metadata.json + scaler.pkl
4. register_specialist()            → model_registry.json + {ticker}_canonical.pt
5. load_specialist(ticker)          → (model_loaded, config_loaded, preprocessor_loaded)
6. model_loaded(input)              → output idéntico al original
```

## Archivo

`tests/test_roundtrip.py` — archivo dedicado.

## Modelo tiny

Para mantener el test rápido (<3s), usar dimensiones mínimas:
- `d_model=32`, `n_heads=2`, `d_ff=64`
- `e_layers=1`, `d_layers=1`
- `seq_len=16`, `pred_len=4`, `label_len=8`
- `enc_in=5`, `dec_in=5`
- `n_flow_layers=2`, `flow_hidden_dim=16`

## Tests

### T1: `test_config_roundtrip`
Guarda config_dict en registry JSON, reconstruye con `_build_config`, verifica que
todos los arch params coinciden: `d_model`, `n_heads`, `d_ff`, `e_layers`, `d_layers`,
`modes`, `dropout`, `n_flow_layers`, `flow_hidden_dim`, `enc_in`, `dec_in`, `label_len`,
`seq_len`, `pred_len`, `target_features`, `return_transform`, `seed`.

### T2: `test_preprocessor_artifacts_roundtrip`
Crea PreprocessingPipeline, lo fittea con un DataFrame sintético, guarda artifacts,
carga en un pipeline nuevo, verifica igualdad de:
- `scaler` (transform de un vector conocido → mismo resultado)
- `target_indices`, `feature_columns`, `numeric_columns`
- `return_transform`, `last_prices`
- `outlier_bounds`, `fill_values`, `fit_stats`

### T3: `test_model_state_dict_roundtrip`
Crea Flow_FEDformer tiny, guarda state_dict en checkpoint format, carga en modelo
nuevo, pasa el mismo input → output bitwise idéntico (`torch.equal`).

### T4: `test_full_roundtrip_save_load_predict`
Test de integración completo:
1. Crea CSV sintético en `tmp_path` con 5 features + columna Date
2. Crea FEDformerConfig tiny apuntando al CSV
3. Crea PreprocessingPipeline, fit con el DataFrame
4. Crea Flow_FEDformer, forward pass → captura output original
5. Guarda checkpoint (formato `save_checkpoint`)
6. Guarda preprocessing artifacts
7. Llama `register_specialist` con registry en `tmp_path`
8. Llama `load_specialist` apuntando al registry de `tmp_path`
9. Verifica: config params coinciden, preprocessor state coincide
10. Forward pass con modelo cargado → `torch.allclose` con output original

### T5: `test_enc_in_dec_in_survives_roundtrip`
Regresión para el bug de sesión 19: `__post_init__` sobreescribe `enc_in`/`dec_in`
leyendo el CSV. Verifica que los valores del registry prevalecen post-load.

## Fixtures

- `tiny_config(tmp_path)` → FEDformerConfig con dimensiones mínimas + CSV sintético
- `fitted_preprocessor(tiny_config)` → PreprocessingPipeline fitteado
- `tiny_model(tiny_config)` → Flow_FEDformer en eval mode
- `saved_checkpoint(tmp_path, tiny_model)` → path al .pt guardado
- `registry_entry(tmp_path, ...)` → dict de registry con paths reales

## Dependencias

Solo `pytest`, `torch`, `pandas`, `numpy` — sin mocks de red, sin GPU requerida.

## Criterios de aceptación

- 5 tests passing
- Tiempo total < 5s
- No marca `@slow`
- CI verde (incluido en `ci.yml` sin cambios al workflow)
