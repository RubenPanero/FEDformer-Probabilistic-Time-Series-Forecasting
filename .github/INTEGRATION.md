# GitHub Actions Integration Guide

Este archivo documenta la integración de GitHub Actions con el proyecto FEDformer y cómo los workflows validan las correcciones críticas.

## 📁 Estructura de Workflows

```
.github/
├── workflows/
│   ├── critical-fixes.yml        # Validación de 5 correcciones críticas
│   ├── compatibility.yml         # Tests de compatibilidad multi-versión
│   ├── security.yml              # Análisis de seguridad y calidad
│   └── README.md                 # Este archivo
└── INTEGRATION.md                # Este archivo
```

## 🔄 Flujo de Validación

### 1. Trigger Manual (Local)
```bash
# Simular validación local antes de push
cd /home/nexus/PROYECTOS\ PYTHON/FEDFORMER/FEDformer-Probabilistic-Time-Series-Forecasting

# Ejecutar tests críticos
python -m pytest tests/test_critical_fixes.py -v

# Ejecutar validaciones estáticas
python tests/validate_fixes.py

# Analizar código
flake8 . --count --select=E9,F63,F7,F82
black --check .
```

### 2. Push a GitHub
```bash
git add .
git commit -m "Critical fixes + GitHub Actions workflows"
git push origin main
```

### 3. Validación Automática
GitHub Actions ejecutará automáticamente:
- **critical-fixes.yml**: Verifica 5 correcciones (Python 3.9, 3.10, 3.11)
- **compatibility.yml**: Tests de compatibilidad
- **security.yml**: Análisis de seguridad

### 4. Monitoreo
- Abre GitHub Actions tab
- Selecciona el workflow
- Verifica estado (✅ exitoso / ❌ fallido)
- Revisa logs para detalles

## ✅ Validaciones Ejecutadas

### Critical Fixes Workflow (critical-fixes.yml)

Ejecuta para cada Python version (3.9, 3.10, 3.11):

**Fix 1: Walk-forward Data Leakage Prevention**
```python
# Verificar línea 394 en training/trainer.py
train_indices = list(range(fold_idx * split_size))
# Evita que fold N de entrenamiento incluya datos de fold N+1
```

**Fix 2: RegimeDetector Volatility Calculation**
```python
# Verificar línea 28-52 en data/dataset.py
rolling_vol = pd.DataFrame(returns).rolling(...).std(ddof=1)
# Usa .std() (desviación estándar) en lugar de .mean()
```

**Fix 3: Fourier Attention Determinism**
```python
# Verificar línea 86-93 en models/layers.py
generator = torch.Generator()
seed = (seq_len * 1009 + self.modes * 1013) % (2**31 - 1)
# Garantiza reproducibilidad usando seed determinístico
```

**Fix 4: Trend Projection Validation**
```python
# Verificar línea 160-167 en models/fedformer.py
raise RuntimeError(f"Trend shape mismatch...")
# Valida tendencia antes de proyectar en lugar de crear nn.Linear ad-hoc
```

**Fix 5: Log-Det Jacobian Normalization**
```python
# Verificar línea 105-119 en models/flows.py
log_det_jacobian = log_det_jacobian / n_layers
# Normaliza para evitar escalado exponencial con profundidad del flow
```

### Compatibility Workflow (compatibility.yml)

Verifica que el código funciona en múltiples entornos:

1. **Module Imports** - Todos los módulos principales importan sin error
2. **Config Initialization** - FEDformerConfig se inicializa correctamente
3. **RegimeDetector** - Volatility fix funciona con datos aleatorios
4. **Fourier Attention** - Indices son determinísticos
5. **Flow_FEDformer** - Forward pass sin errores
6. **NormalizingFlow** - Log-prob scaling es numéricamente estable
7. **No Breaking Changes** - Métodos clave aún existen

### Security Workflow (security.yml)

Ejecuta análisis de calidad y seguridad:

1. **Code Formatting** - Black
2. **Import Order** - isort
3. **Linting** - flake8
4. **Security Scanning** - Detecta patrones inseguros
5. **Dependency Check** - Verifica vulnerabilidades
6. **Fixes Integrity** - Confirma que todas las correcciones están presentes

## 🎯 Estado de los Workflows

### Cómo Verificar Estado

**Opción 1: GitHub Web UI**
1. Ve a tu repositorio en GitHub
2. Click en **Actions**
3. Selecciona workflow más reciente
4. Ve el status y logs

**Opción 2: GitHub CLI**
```bash
gh run list --repo YOUR_USERNAME/FEDformer-Probabilistic-Time-Series-Forecasting
gh run view RUN_ID --repo YOUR_USERNAME/FEDformer-Probabilistic-Time-Series-Forecasting
```

**Opción 3: Badges en README**
Añade a tu README.md:
```markdown
## CI/CD Status

![Critical Fixes](https://github.com/YOUR_USERNAME/FEDformer-Probabilistic-Time-Series-Forecasting/actions/workflows/critical-fixes.yml/badge.svg)
![Compatibility](https://github.com/YOUR_USERNAME/FEDformer-Probabilistic-Time-Series-Forecasting/actions/workflows/compatibility.yml/badge.svg)
![Security](https://github.com/YOUR_USERNAME/FEDformer-Probabilistic-Time-Series-Forecasting/actions/workflows/security.yml/badge.svg)
```

## 🔐 Secretos y Variables

Si necesitas secretos (API keys, credentials):

**Cómo añadir:**
1. Ve a Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Nombre: `MY_SECRET`
4. Valor: `***`

**Uso en workflow:**
```yaml
- name: Use secret
  env:
    MY_SECRET: ${{ secrets.MY_SECRET }}
  run: |
    echo "Secret is set"
```

## 📊 Monitoreo y Alertas

### Notificaciones Automáticas
GitHub te notificará en:
- Email cuando un workflow falla
- GitHub Notifications
- (Opcional) Slack/Discord si lo configuras

### Configurar Notificaciones
1. Settings → Notifications
2. "Actions" → Selecciona preferencias
3. Guarda cambios

## 🐛 Debugging de Workflows Fallidos

### Paso 1: Revisar Logs
1. Click en workflow fallido
2. Expande el step que falló
3. Lee el output detallado

### Paso 2: Reproducir Localmente
```bash
# Reproduce el mismo ambiente
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Ejecuta la validación que falló
python -m pytest tests/test_critical_fixes.py -v
```

### Paso 3: Verificar Código
```bash
# Si fix #1 falló
grep -n "train_indices = list(range(fold_idx * split_size))" training/trainer.py

# Si fix #2 falló
grep -n ".std(ddof=1)" data/dataset.py
```

### Paso 4: Corregir y Re-push
```bash
# Haz cambios localmente
# ...edita archivos...

# Valida localmente
python tests/validate_fixes.py

# Push para re-ejecutar workflows
git add .
git commit -m "Fix: [describe fix]"
git push origin main
```

## 🚀 Optimizaciones de Workflows

### Caché de Dependencias
Los workflows ya usan caché de pip para velocidad:
```yaml
- uses: actions/setup-python@v4
  with:
    cache: 'pip'
```

### Matriz de Python Versions
Para testear múltiples versiones:
```yaml
strategy:
  matrix:
    python-version: ['3.9', '3.10', '3.11']
```

### Ejecución Condicional
Para ejecutar solo en ciertos casos:
```yaml
if: contains(github.event.head_commit.message, 'run-all-tests')
```

## 📝 Mantenimiento de Workflows

### Actualizar Versiones de Python
Edita la sección `matrix` cuando lances soporte para nuevas versiones:
```yaml
python-version: ['3.9', '3.10', '3.11', '3.12']
```

### Añadir Nuevos Tests
1. Crea test en `tests/`
2. Añade step en workflow .yml:
```yaml
- name: Run new test
  run: python -m pytest tests/new_test.py -v
```

### Cambiar Schedule
Para security.yml:
```yaml
schedule:
  - cron: '0 12 * * *'  # Todos los días a 12:00 UTC
```

## 🔗 Integraciones Útiles

### GitHub API para Automatización
```bash
# Get últimas ejecuciones
curl -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/YOUR_USERNAME/FEDformer-Probabilistic-Time-Series-Forecasting/actions/runs

# Triggerear workflow manualmente
gh workflow run critical-fixes.yml --repo YOUR_USERNAME/FEDformer-Probabilistic-Time-Series-Forecasting
```

### Webhook para Notificaciones
Configura webhook en Settings → Webhooks para:
- Push events
- Pull request events
- Workflow run completions

## 📚 Referencias

- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [Status Badges](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/adding-a-workflow-status-badge)
- [Caching Dependencies](https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows)

## ✨ Mejoras Futuras

- [ ] Añadir workflow para coverage reports
- [ ] Integrar con Codecov
- [ ] Añadir workflow para building documentation
- [ ] Integrar con PyPI para auto-releases
- [ ] Setup auto-updates para dependencias
- [ ] Crear workflow para performance benchmarks

---

**Última actualización:** $(date)
**Workflows operacionales:** 3 ✅
**Status general:** Production Ready
