# Makefile — FEDformer targets de calidad de código

PROJDIR := $(shell pwd)
PYTHON  := python3

.PHONY: lint lint-fix ci-check test claude-review

## Verificar lint sin modificar archivos (rápido)
lint:
	ruff check . && ruff format --check .

## Auto-fix lint con ruff directamente (sin Claude, rápido)
lint-fix:
	ruff check . --fix && ruff format .

## Paridad CI completa (pre-commit canónico)
ci-check:
	ruff check . --fix && ruff format . && \
	pylint --errors-only models/ training/ data/ utils/ inference/ && \
	pytest -q -m "not slow"

## Tests rápidos
test:
	pytest -q -m "not slow"

## Claude headless — revisión semántica on-demand
## Uso: make claude-review  |  make claude-review FILES="inference/loader.py utils/helpers.py"
FILES ?= $(shell git diff --name-only HEAD~1..HEAD 2>/dev/null | grep '\.py$$' | tr '\n' ' ')
claude-review:
	@if [ -z "$(FILES)" ]; then \
	    echo "ℹ️  Sin archivos Python modificados en el último commit."; \
	    echo "    Usa: make claude-review FILES=\"path/to/file.py\""; \
	else \
	    echo "▶  Claude headless revisando: $(FILES)"; \
	    claude --dangerously-skip-permissions -p \
	      "Revisa estos archivos Python: $(FILES). Busca: imports no usados, variables sin usar, f-strings corregibles, errores de lógica. Corrige lo que puedas con Edit y muestra un resumen de máximo 5 líneas." \
	      --allowedTools "Bash,Edit,Read" \
	      --output-format text; \
	fi
