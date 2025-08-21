<# Verify and Smoke Script for Vanguard-FEDformer (PowerShell) #>
# Steps:
# 1) Create/activate virtual environment
# 2) Install dependencies
# 3) Run unit tests (pytest)
# 4) Run a smoke test with Nvidia CSV data
# 5) Optional: produce a simple report

$Root = 'C:\Users\rbglz\Documents\GitHub\Vanguard-FEDformer-Advanced-Probabilistic-Time-Series-Forecasting'
$VenvDir = Join-Path $Root '.venv'
$ReportsDir = Join-Path $Root 'reports'
$TestResults = Join-Path $ReportsDir 'test_results.txt'
$CsvPath = Join-Path $Root 'data\nvidia_stock_2024-08-20_to_2025-08-20.csv'

# 1) Create/activate venv
if (-Not (Test-Path $VenvDir)) {
    Write-Host 'Creating virtual environment...' -ForegroundColor Cyan
    python -m venv $VenvDir
}

Write-Host 'Activating virtual environment...'
& "$VenvDir\Scripts\Activate.ps1" | Out-Null

# 2) Install dependencies
Write-Host 'Installing dependencies...'
& "$VenvDir\Scripts\pip.exe" install --upgrade pip
& "$VenvDir\Scripts\pip.exe" install -r "$Root\requirements.txt"

# 3) Run unit tests
if (-Not (Test-Path $ReportsDir)) { New-Item -ItemType Directory -Path $ReportsDir | Out-Null }
Write-Host 'Running unit tests...'
pytest -q | Tee-Object -FilePath $TestResults -Encoding utf8
Write-Host "Test results saved to $TestResults" -ForegroundColor Green

# 4) Smoke test with Nvidia CSV
Write-Host 'Launching smoke test with Nvidia CSV...'
# Determine a good target column and date column from the header
$FirstLine = Get-Content -Path $CsvPath -TotalCount 1
$Headers = $FirstLine -split ','
$PossibleTargets = @('Close','close','close_price','Adj Close')
$Target = $null
foreach ($h in $PossibleTargets) {
    if ($Headers -contains $h) { $Target = $h; break }
}
if (-not $Target) {
    foreach ($h in $Headers) {
        if ($h -notmatch 'date|Date|time|timestamp') { $Target = $h; break }
    }
}
$dateCol = $null
foreach ($h in $Headers) {
    if ($h -match '^[dD]ate$' -or $h -match '^[dD]ate.*') { $dateCol = $h; break }
}

$SmokeCmd = "python main.py --csv `"$CsvPath`" --targets `"$Target`""
if ($dateCol) { $SmokeCmd += " --date-col `"$dateCol`"" }
$SmokeCmd += " --pred-len 4 --seq-len 16 --label-len 8 --epochs 1 --batch-size 4 --splits 2 --seed 123 --deterministic --no-show"

Write-Host "Executing: $SmokeCmd" -ForegroundColor Yellow
Invoke-Expression $SmokeCmd

Write-Host 'Verification complete.' -ForegroundColor Green
