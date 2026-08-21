$ErrorActionPreference = 'Stop'
$venv = Join-Path $PSScriptRoot '.venv'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing .venv. Run: py -3.11 -m venv .venv"
}
if ((Get-Item -LiteralPath $venv -Force).LinkType) {
    throw ".venv is a link or junction. Recreate it inside this project; virtual environments are not portable."
}
& $python -c "import duckdb"
if ($LASTEXITCODE -ne 0) {
    throw "The project environment is incomplete. Reinstall with: .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}

& $python (Join-Path $PSScriptRoot 'src\download_bts.py')
& $python (Join-Path $PSScriptRoot 'src\download_nws_hazards.py')
& $python (Join-Path $PSScriptRoot 'src\download_faa.py')
& $python (Join-Path $PSScriptRoot 'src\build_dataset.py')
& $python (Join-Path $PSScriptRoot 'src\sample_model_data.py')
& $python (Join-Path $PSScriptRoot 'src\eda.py')
& $python (Join-Path $PSScriptRoot 'src\enrich_features.py')
& $python (Join-Path $PSScriptRoot 'src\train_models.py')
& $python (Join-Path $PSScriptRoot 'src\missingness_audit.py')
& $python (Join-Path $PSScriptRoot 'src\train_delay_thresholds.py')
& $python (Join-Path $PSScriptRoot 'src\multicollinearity_and_shapes.py')
& $python (Join-Path $PSScriptRoot 'src\make_report.py')
