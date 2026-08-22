$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

py -3.11 -m venv .venv
& .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host "Setup completed successfully."