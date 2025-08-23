<# Verify local environment for Vanguard-FEDformer project (PowerShell script) #>
<# Steps>
# 1) Create/activate virtual environment
# 2) Install dependencies
# 3) Run unit tests (pytest)
# 4) Run a smoke test with synthetic data
# 5) Produce a short report
#> 

$Root = 'C:\Users\rbglz\Documents\GitHub\Vanguard-FEDformer-Advanced-Probabilistic-Time-Series-Forecasting'
$VenvPath = Join-Path $Root '.venv'
$PythonExe = Join-Path $VenvPath 'Scripts\python.exe'

# 1) Create/activate venv
if (-Not (Test-Path $VenvPath)) {
    Write-Host 'Creating virtual environment...' -ForegroundColor Cyan
    & 'python' -m venv $VenvPath
}

Write-Host 'Activating virtual environment...'
& "$VenvPath\Scripts\Activate.ps1" | Out-Null

# 2) Install dependencies
Write-Host 'Installing dependencies...'
& "$VenvPath\Scripts\pip.exe" install --upgrade pip
& "$VenvPath\Scripts\pip.exe" install -r "$Root\requirements.txt"

# 3) Ensure tests scaffold is present
Write-Host 'Running unit tests (pytest)...'
$TestResultsPath = Join-Path $Root 'test_results.txt'
( pytest -q 2>&1 ) | Tee-Object -FilePath $TestResultsPath
Write-Host "Test results written to $TestResultsPath"

# 4) Smoke test data
$SmokeCSV = Join-Path $Root 'data\smoke_test.csv'
if (-Not (Test-Path $SmokeCSV)) {
    Write-Host 'Generating synthetic smoke test data...'
    $date = Get-Date
    $rows = 40
    $lines = @('date,close_price,volume')
    for ($i=0; $i -lt $rows; $i++) {
        $d = $date.AddDays($i).ToString('yyyy-MM-dd')
        $close = [Math]::Round(100 + (Get-Random -Minimum 0 -Maximum 20) + $i*0.2, 2)
        $vol = [Math]::Round(1000 + (Get-Random -Minimum 0 -Maximum 500), 0)
        $lines += "$d,$close,$vol"
    }
    $lines | Set-Content -Path $SmokeCSV
}

Write-Host 'Running smoke test (small dataset)...'
& pytest -q tests/test_flows.py -q -k 'roundtrip' 2>&1 | Tee-Object -FilePath ($Root + '\smoke_test.txt')

# Optional: full smoke run of main with the synthetic data (uncomment if desired)
<#
Write-Host 'Running full smoke run with synthetic data...'
& $PythonExe main.py --csv $SmokeCSV --targets 'close_price' --date-col 'date' --pred-len 4 --seq-len 16 --label-len 8 --epochs 1 --batch-size 4 --splits 2 --seed 123 --no-show 2>&1 | Tee-Object -FilePath ($Root + '\smoke_run.txt')
#>

Write-Host 'Verification script completed.'
