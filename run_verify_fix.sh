#!/usr/bin/env bash
set -euo pipefail
WORKDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORKDIR"
source .venv/bin/activate

echo "=== Verificación fix per-fold reseed — seed=42 ==="
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
echo "=== Métricas portfolio ==="
PORTFOLIO=$(ls -t results/portfolio_metrics_*.csv | head -1)
echo "Archivo: $PORTFOLIO"
cat "$PORTFOLIO"

echo ""
echo "=== Referencia baseline (seed=42, pre-regresión) ==="
echo "  Sharpe:  +0.609"
echo "  Sortino: +0.993"
echo "  MaxDD:   -72.69%"
