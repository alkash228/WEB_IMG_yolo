$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
    & .\venv\Scripts\Activate.ps1
    pip install -r requirements.txt
} else {
    & .\venv\Scripts\Activate.ps1
}

Write-Host "Starting WEB IMG YOLO on http://localhost:8091"
python server.py
