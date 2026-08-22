$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Error "Virtual environment not found. Run .\setup.ps1 first."
}

& .\.venv\Scripts\Activate.ps1

$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:TOKENIZERS_PARALLELISM = "false"

python agent.py