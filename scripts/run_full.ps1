# One-command launcher for full project run on Windows (PowerShell)
# - Creates a local venv
# - Installs dependencies (API + PDF generation)
# - Tries to start Docker DB (optional; continues if unavailable)
# - Restarts any server on port 5000
# - Starts the Flask app and opens browser pages

param()
$ErrorActionPreference = 'Stop'

function Write-Info($msg){ Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Warn($msg){ Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err($msg){ Write-Host "[ERROR] $msg" -ForegroundColor Red }

# Move to project root (script lives in scripts/)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

Write-Info "Project root: $ProjectRoot"

# Create venv if missing
$venvPy = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
  Write-Info "Creating virtual environment (.venv)"
  python -m venv .venv
}
if (-not (Test-Path $venvPy)) {
  Write-Err "Virtual environment creation failed. Ensure Python 3 is installed and in PATH."
  exit 1
}

# Install dependencies
Write-Info "Upgrading pip and installing packages"
& $venvPy -m pip install --upgrade pip
& $venvPy -m pip install -r requirements.txt
# Extra runtime libs (non-fatal if some optional ML libs fail)
& $venvPy -m pip install flask-cors pandas numpy scikit-learn 2>$null | Out-Null
try { & $venvPy -m pip install xgboost lightgbm 2>$null | Out-Null } catch { Write-Warn "Optional ML packages xgboost/lightgbm not installed" }

# Try Docker (optional)
$dockerOk = $false
try {
  Write-Info "Attempting to start Docker services (optional)"
  docker compose up -d 2>$null | Out-Null
  $dockerOk = $true
} catch {
  try {
    docker-compose up -d 2>$null | Out-Null
    $dockerOk = $true
  } catch {
    Write-Warn "Docker is not available or images could not be pulled. Continuing without database."
  }
}

# Stop existing server on port 5000 (if any)
try {
  $existing = Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($existing) {
    Write-Warn "Port 5000 is in use by PID $($existing.OwningProcess). Stopping it."
    Stop-Process -Id $existing.OwningProcess -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
  }
} catch {}

# Start the Flask app
Write-Info "Starting Flask app..."
$proc = Start-Process -FilePath $venvPy -ArgumentList "start_server.py" -WorkingDirectory $ProjectRoot -PassThru -WindowStyle Minimized
Write-Info "Server PID: $($proc.Id)"

# Wait for readiness (up to ~120s)
$ready = $false
for ($i=0; $i -lt 60; $i++) {
  Start-Sleep -Seconds 2
  try {
    $r = Invoke-WebRequest -Uri 'http://localhost:5000/' -UseBasicParsing -TimeoutSec 3
    if ($r.StatusCode -eq 200) { $ready = $true; break }
  } catch {}
}

if ($ready) {
  Write-Info "Server is up at http://localhost:5000/"
  Start-Process "http://localhost:5000/simple"
  Start-Process "http://localhost:5000/download-report"
  if ($dockerOk) { Write-Info "Docker started. Database-backed endpoints will work once healthy." } else { Write-Warn "Running without database. Simple interface and PDF reports will work." }
  Write-Info "To stop the server later: Stop-Process -Id $($proc.Id)"
  exit 0
} else {
  Write-Warn "Server did not respond in time, but the process is still running (PID $($proc.Id)). It may still be starting."
  Write-Info "Try opening http://localhost:5000/simple in your browser after a few more seconds."
  Write-Info "To stop the server later: Stop-Process -Id $($proc.Id)"
  exit 0
}
