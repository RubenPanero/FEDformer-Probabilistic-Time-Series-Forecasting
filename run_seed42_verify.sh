#!/usr/bin/env bash
# Verifica seed=42 con el per-fold reseed fix activo.
# Objetivo: determinar si seed=42 es intrínsecamente pobre o recuperable
# con el código actual.
set -euo pipefail
WORKDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORKDIR"
source .venv/bin/activate

echo "=== Verificación seed=42 — per-fold reseed fix ==="
echo "Inicio: $(date '+%H:%M:%S')"
START=$(date +%s)

MPLBACKEND=Agg python3 main.py \
  --csv data/NVDA_features.csv \
  --targets "Close" \
  --seq-len 96 --pred-len 20 --batch-size 64 --splits 4 \
  --return-transform log_return --metric-space returns \
  --gradient-clip-norm 0.5 \
  --seed 42 \
  --save-results --no-show

END=$(date +%s)
ELAPSED=$((END - START))
printf "Tiempo total: %d min %d seg\n" $((ELAPSED / 60)) $((ELAPSED % 60))

echo ""
echo "=== Métricas portfolio (seed=42, con fix) ==="
PORTFOLIO=$(ls -t results/portfolio_metrics_*.csv | head -1)
cat "$PORTFOLIO"

echo ""
echo "=== Comparativa histórica ==="
echo "  seed=42  sin fix  (Mar-12, regresión):   Sharpe -1.127 | Sortino -1.478 | MaxDD -93.2%"
echo "  seed=42  con fix  (Mar-13, run local):    Sharpe -0.525 | Sortino -0.717 | MaxDD -73.8%"
echo "  seed=7   con fix  (Mar-14, Kaggle 2xT4):  Sharpe +1.060 | Sortino +1.940 | MaxDD -55.9%"
echo "  seed=42  con fix  (este run) ------------> ver CSV arriba"
echo ""
echo "Interpretación:"
echo "  Si Sharpe ≈ -0.525 → seed=42 intrínsecamente pobre con código actual"
echo "  Si Sharpe > 0      → varianza de ejecución; seed=42 recuperable"
