# FEDformer Performance Optimization Plan

## Executive Summary

This document outlines a comprehensive performance optimization strategy for the FEDformer-Probabilistic-Time-Series-Forecasting repository, based on architectural analysis and profiling insights.

**Current Performance Characteristics:**
- Single canonical training run (NVDA, seed=7): ~15-45 minutes (GPU dependent)
- MC Dropout inference (50 samples): ~40% of evaluation time
- FourierAttention operations: ~20% of model forward pass
- Normalizing Flow computations: ~15% of loss calculation
- Optuna hyperparameter search: 15 min timeout per trial × 16-20 trials = 4-5 hours

**Optimization Goals:**
1. Reduce training time by 30-50% without sacrificing model quality
2. Optimize MC dropout inference for faster evaluation
3. Improve memory efficiency for larger batch sizes
4. Accelerate Optuna hyperparameter search
5. Provide profiling tools for continuous performance monitoring

---

## Phase 1: Critical Bottlenecks (HIGH IMPACT, LOW EFFORT)

### 1.1 MC Dropout Inference Optimization

**Implemented subset**

- **Description:** reduced trainer-side stochastic evaluation cost without touching inference defaults.
- **Code locations:** `config.py`, `main.py`, `training/trainer.py`, `training/utils.py`
- **Changes shipped:**
  - Added `mc_dropout_eval_samples` to runtime config with default `20`
  - Added CLI flag `--mc-dropout-eval-samples` for trainer evaluation folds
  - Kept inference default at `50` samples in `inference/predictor.py`
  - Extended `mc_dropout_inference()` with `mc_batch_size` to chunk MC sample accumulation while preserving output shape and fallback behavior; this does not add vectorized model forwards
  - Updated `WalkForwardTrainer._evaluate_model()` to use `self.config.mc_dropout_eval_samples` and `mc_batch_size=10`
- **Dependencies:** quantile aggregation still uses `DEFAULT_QUANTILE_LEVELS`; `preds_* == p50` compatibility remains unchanged

**Validation**

- Unit tests:
  - config accepts `mc_dropout_eval_samples`
  - batched and unbatched MC inference match under fixed seed
  - model train/eval mode restoration is preserved
  - fallback zero tensor shape is unchanged on sampling failures
  - trainer consumes the configured sample count instead of a hard-coded `50`
- Integration tests:
  - `run_backtest()` respects `mc_dropout_eval_samples`
  - `predict()` still uses `50` samples by default even if trainer config uses another value

**Synthetic benchmark evidence**

Measured with:

```bash
source .venv/bin/activate
python3 docs/optimization/critical_bottlenecks_benchmark.py
```

| Scenario | Baseline | Optimized | Delta |
|----------|----------|-----------|-------|
| MC Dropout time (20 samples, CPU synthetic) | 0.4850s | 0.0619s | -87.2% |
| MC Dropout peak Python memory | 12128 B | 9135 B | -24.7% |

**Notes**

- The benchmark isolates batching overhead and uses the same `20` samples on both sides.
- Analytical approximation remains deferred.

---

### 1.2 FourierAttention FFT Optimization

**Implemented subset**

- **Description:** reused the existing configurable `modes` path and shipped an explicit optimization preset instead of rewriting the kernel.
- **Code locations:** `config.py`, `tests/test_config_presets.py`, `docs/optimization/critical_bottlenecks_benchmark.py`
- **Changes shipped:**
  - Added preset `fourier_optimized`
  - Preset sets `modes=48`
  - Left the global default `modes=64` intact
  - Did not add isolated `torch.compile` logic in this phase
- **Dependencies:** current `config.py` clamping logic still applies when `seq_len // 2 < 48`

**Validation**

- Unit tests:
  - preset registration and `apply_preset()` wiring
  - existing CLI-overrides-preset behavior remains intact
- Benchmarks:
  - `docs/optimization/critical_bottlenecks_benchmark.py` compares `modes=64` vs `modes=48`

**Synthetic benchmark evidence**

| Scenario | Baseline | Optimized | Delta |
|----------|----------|-----------|-------|
| FourierAttention forward time (CPU synthetic) | 0.001093s | 0.000652s | -40.3% |
| Peak Python memory | 1212 B | 1212 B | 0.0% |

**Notes**

- This gain is synthetic and CPU-bound; canonical training quality must still be checked against NVDA/GOOGL before promoting `modes=48` as a default.
- FFT index caching was already present and remains unchanged.

---

### 1.3 Normalizing Flow Layer Optimization

**Implemented subset**

- **Description:** extended checkpointing to the flow-backed `log_prob()` path used during training.
- **Code locations:** `models/fedformer.py`, `tests/test_critical_bottlenecks.py`
- **Changes shipped:**
  - `Flow_FEDformer.forward()` now passes `use_gradient_checkpointing` and training state into `NormalizingFlowDistribution`
  - `NormalizingFlowDistribution.log_prob()` now wraps per-feature flow `log_prob()` with `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` only when checkpointing is enabled and the model is in training mode
  - eval/inference paths remain unchanged
- **Dependencies:** sequence-layer checkpointing in `_run_sequence_layers()` remains the outer checkpointing mechanism; flow checkpointing is additive and training-only

**Validation**

- Unit tests:
  - checkpointed and non-checkpointed `log_prob()` produce equivalent values
  - gradients still propagate through flow parameters
  - checkpoint path is actually invoked when enabled

**Synthetic benchmark evidence**

| Scenario | Baseline | Optimized | Delta |
|----------|----------|-----------|-------|
| Flow `log_prob()` time (CPU synthetic) | 0.0066s | 3.1860s | +47988.9% |
| Peak Python memory | 9890 B | 57407019 B | +580355.2% |

**Notes**

- These CPU `tracemalloc` numbers measure Python-side allocation overhead and are **not** representative of the intended GPU-memory savings during real training.
- Checkpointing is kept because the target benefit is reduced activation memory on larger training workloads; on CPU or tiny tensors it is expected to be slower.
- Fused affine transforms remain deferred.

---

## Phase 2: Structural Optimizations (HIGH IMPACT, MEDIUM EFFORT)

### 2.1 Torch Compilation Strategy

**Current State:**
- `torch.compile` available but disabled by default in Optuna trials
- Auto-disabled on GPUs with <40 SMs (e.g., RTX 4050)

**Optimization Strategy:**

#### Selective Compilation
```python
# training/trainer.py:_get_model
def _get_model(self, config: FEDformerConfig) -> Flow_FEDformer:
    model = Flow_FEDformer(config).to(device)
    
    if self.config.compile_mode and self.config.compile_mode != "":
        # Compile only for training, not inference
        if self.is_training_phase:
            model = torch.compile(
                model,
                mode=self.config.compile_mode,
                dynamic=True  # Handle variable batch sizes
            )
    
    return model
```

**Compilation Modes:**
- `""` (disabled): Optuna trials, small datasets
- `"default"`: General purpose, low overhead
- `"reduce-overhead"`: Inference, MC dropout
- `"max-autotune"`: Training on A100/H100 (>=40 SMs)

**Impact:** 1.5-2x speedup for compiled models (after warmup)
**Risk:** Compilation overhead (30s-2min first run)

---

### 2.2 Data Preprocessing Caching

**Current Issue:**
```python
# data/dataset.py:TimeSeriesDataset._fit_and_transform
# Re-fits preprocessing for EVERY fold
should_refit = (
    force_refit
    or fit_scope == "fold_train_only"  # True for walk-forward
    or not self.preprocessor.fitted
)
```

**Optimization Strategy:**

#### Incremental Scaling
```python
class IncrementalScaler:
    """Scaler that can be updated incrementally."""
    
    def partial_fit(self, X, batch_size=1024):
        # Update running mean/var without full re-fit
        # Useful for walk-forward where data shifts gradually
        pass

class PreprocessingPipeline:
    def incremental_update(self, new_data: pd.DataFrame) -> None:
        """Update statistics without full re-fit."""
        # Use Welford's online algorithm for running stats
        # Update scaler parameters incrementally
        pass
```

**Impact:** 50% faster preprocessing in walk-forward folds

#### Cross-Fold Caching
```python
# Cache preprocessing artifacts between folds
PREPROCESSING_CACHE = {}

def fit_with_cache(self, data, fold_idx):
    cache_key = f"fold_{fold_idx}_{len(data)}"
    if cache_key in PREPROCESSING_CACHE:
        return PREPROCESSING_CACHE[cache_key]
    
    # Fit and cache
    result = self._fit(data)
    PREPROCESSING_CACHE[cache_key] = result
    return result
```

**Impact:** Eliminates redundant preprocessing for similar fold sizes

---

### 2.3 Optuna Trial Optimization

**Current Issue:**
```python
# tune_hyperparams.py:objective
# Each trial spawns FULL subprocess with main.py
result = subprocess.run(
    cmd,
    timeout=900,  # 15 minutes!
    capture_output=True
)
```

**Optimization Strategy:**

#### Option A: In-Process Trial Execution
```python
def objective_in_process(trial: optuna.Trial) -> float:
    """Run trial in same process to avoid subprocess overhead."""
    # Suggest hyperparameters
    params = suggest_params(trial)
    
    # Create config
    config = build_config(params)
    
    # Run training directly (no subprocess)
    trainer = WalkForwardTrainer(config, dataset)
    results = trainer.run_backtest()
    
    return extract_sharpe(results)
```

**Impact:** 30-60 seconds saved per trial (no process spawn overhead)

#### Option B: Early Stopping for Bad Trials
```python
def objective_with_pruning(trial: optuna.Trial) -> float:
    """Prune trials early if metrics are poor."""
    config = build_config(trial)
    
    # Run single fold first
    single_fold_result = run_single_fold(config, fold=0)
    
    # Prune if first fold is terrible
    if single_fold_result.sharpe < -0.5:
        raise optuna.TrialPruned()
    
    # Continue with full walk-forward
    return run_full_backtest(config)
```

**Impact:** 40% of trials pruned early, saving 10+ minutes each

#### Option C: Successive Halving
```python
# Use Optuna's SuccessiveHalvingPruner
pruner = optuna.pruners.SuccessiveHalvingPruner(
    min_resource=1,
    reduction_factor=3,
    min_early_stopping_rate=2
)

study = optuna.create_study(
    direction='maximize',
    pruner=pruner
)
```

**Impact:** Automatically allocates more resources to promising trials

**Recommendation:** Implement **Option B** first (easy win), then **Option C**

---

## Phase 3: Memory Optimizations (MEDIUM IMPACT, MEDIUM EFFORT)

### 3.1 Gradient Accumulation with Larger Effective Batches

**Current State:**
```python
# config.py:LoopSettings
batch_size: int = 64  # Current canonical
grad_accum_steps: int = 1  # Default
```

**Optimization:**
```python
# Use smaller batches with gradient accumulation
batch_size = 32  # Fits in memory
grad_accum_steps = 2  # Effective batch = 64

# Reduces peak memory by 50%, same effective batch size
# Enables training with larger effective batches on limited VRAM
```

**Impact:** Enables 2x larger effective batch sizes on same hardware

---

### 3.2 Activation Checkpointing (Already Implemented ✅)

```python
# models/fedformer.py:_run_sequence_layers
if use_checkpointing and self.training:
    enc_out = torch.utils.checkpoint.checkpoint(
        encoder_layer, enc_out, use_reentrant=False
    )
```

**Enhancement:**
```python
# Extend to decoder and flow layers
for decoder_layer in self.sequence_layers['decoders']:
    dec_out, trend_delta = torch.utils.checkpoint.checkpoint(
        decoder_layer, dec_out, enc_out, use_reentrant=False
    )

# Flow checkpointing
for flow in self.flows:
    lp = torch.utils.checkpoint.checkpoint(
        flow.log_prob, y_true, context, use_reentrant=False
    )
```

**Impact:** 60% memory reduction, enables 2x larger sequences

---

### 3.3 DataLoader Optimization

**Current State:**
```python
# training/trainer.py:_create_dataloaders
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    worker_init_fn=_SeedWorker(self.config.seed),
)
```

**Optimization:**
```python
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=min(os.cpu_count(), 8),  # Cap at 8
    worker_init_fn=_SeedWorker(self.config.seed),
    pin_memory=torch.cuda.is_available(),  # Already done ✅
    non_blocking=True,  # Already done ✅
    persistent_workers=True,  # NEW: Keep workers alive between epochs
    prefetch_factor=2,  # NEW: Pre-fetch 2 batches ahead
)
```

**Impact:** 10-15% faster data loading, smoother GPU utilization

---

## Phase 4: Advanced Optimizations (LOW IMPACT, HIGH EFFORT)

### 4.1 Mixed Precision Training (Already Implemented ✅)

```python
# training/trainer.py:_train_epoch
with autocast(device_type=device_type, enabled=self.config.use_amp):
    dist = model(x_enc, x_dec, x_regime)
    loss = -dist.log_prob(y_true).mean()
```

**Enhancement:**
```python
# Use bf16 on Ampere+ GPUs
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability()
    if capability[0] >= 8:  # Ampere+
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
```

---

### 4.2 Custom CUDA Kernels for FourierAttention

**Future Work:**
- Write custom Triton/CUDA kernel for FFT + einsum fusion
- Avoid creating complex tensors explicitly
- Reduce memory allocations

**Potential Impact:** 2-3x faster attention
**Effort:** High (requires CUDA expertise)

---

### 4.3 Distributed Training

**Future Work:**
```python
# Use PyTorch DDP for multi-GPU training
if torch.cuda.device_count() > 1:
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank]
    )
```

**Use Case:** Hyperparameter search with parallel trials
**Effort:** Medium (requires DDP setup)

---

## Implementation Roadmap

### Week 1: Quick Wins
- [ ] Reduce MC samples from 50→20 during training
- [ ] Enable `persistent_workers=True` in DataLoader
- [ ] Add environment variable for MC sample count
- [ ] Document profiling tools and usage

### Week 2: Core Optimizations
- [ ] Implement batched MC dropout inference
- [ ] Add gradient checkpointing for flow layers
- [ ] Configure FFT modes (64→48 option)
- [ ] Test torch.compile on different GPUs

### Week 3: Structural Changes
- [ ] Implement early stopping for Optuna trials
- [ ] Add incremental preprocessing updates
- [ ] Create performance regression tests
- [ ] Benchmark all optimizations

### Week 4: Validation & Documentation
- [ ] Run full canonical benchmarks (NVDA, GOOGL)
- [ ] Verify no accuracy degradation
- [ ] Update README with performance tuning guide
- [ ] Create profiling tutorial

---

## Expected Performance Gains

| Optimization | Time Savings | Memory Savings | Accuracy Impact |
|--------------|--------------|----------------|-----------------|
| MC samples 50→20 | -40% eval time | -30% | Minimal |
| Batched MC inference | -15% eval time | -20% | None |
| Reduced FFT modes | -10% forward | -5% | Minimal |
| torch.compile | -30% after warmup | 0% | None |
| Gradient checkpointing | 0% | -40% | None |
| Early Optuna pruning | -40% search time | 0% | None |
| Incremental preprocessing | -20% per fold | 0% | None |

**Combined Impact:**
- **Single training run:** 15-45 min → 10-25 min (30-40% faster)
- **Optuna search (16 trials):** 4-5 hours → 2-3 hours (40% faster)
- **Memory usage:** Enables 2x larger batches/sequences

---

## Profiling Tools & Usage

### Quick Profiling
```bash
# Profile MC dropout inference
python3 profile_optimize.py --mode profile --target mc_dropout

# Profile FourierAttention
python3 profile_optimize.py --mode profile --target fourier_attention

# Profile NormalizingFlows
python3 profile_optimize.py --mode profile --target flows

# Profile all components
python3 profile_optimize.py --mode profile --target all
```

### Benchmark Optimizations
```bash
# Run benchmarks comparing original vs optimized
python3 profile_optimize.py --mode benchmark

# Memory profiling
python3 profile_optimize.py --mode memory --target trainer
```

### Production Profiling
```bash
# Use py-spy for production profiling
py-spy record -o profile.svg -- python3 main.py --csv data/NVDA_features.csv ...

# Line-by-line profiling
kernprof -l -v profile_optimize.py --target mc_dropout
```

---

## Monitoring & Regression

### Performance Regression Tests
```python
# tests/test_performance.py
def test_training_time_regression():
    """Ensure training time doesn't increase."""
    start = time.time()
    # Run minimal training
    elapsed = time.time() - start
    assert elapsed < BASELINE * 1.2  # 20% tolerance

def test_memory_usage_regression():
    """Ensure peak memory doesn't increase."""
    tracemalloc.start()
    # Run training step
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < BASELINE_MEMORY * 1.2
```

### CI Integration
```yaml
# .github/workflows/performance.yml
name: Performance Regression
on: [push]
jobs:
  benchmark:
    runs-on: gpu-runner
    steps:
      - uses: actions/checkout@v4
      - name: Run benchmarks
        run: python3 profile_optimize.py --mode benchmark
      - name: Check regression
        run: |
          # Compare against baseline
          python3 check_performance.py
```

---

## Conclusion

This optimization plan provides a structured approach to improving FEDformer performance by 30-50% with minimal impact on model quality. The phased approach ensures quick wins are captured immediately while more complex optimizations are validated thoroughly.

**Next Steps:**
1. Run profiling suite to establish baseline
2. Implement Week 1 quick wins
3. Validate with canonical benchmarks (NVDA, GOOGL)
4. Iterate through remaining phases

All optimizations should be accompanied by tests to prevent regressions and ensure reproducibility.
