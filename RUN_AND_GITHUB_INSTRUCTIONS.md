# How to Run the Program and Upload to GitHub (Windows, PowerShell)

This document provides complete, step-by-step instructions to run the Deforestation Risk Analysis program and to upload it to GitHub using only local tools and the GitHub website (no AI tools required).

You can run the program in two ways:
- Option A: Quick Demo (no database setup)
- Option B: Full System (with PostgreSQL/PostGIS via Docker)

---

## 1) Prerequisites

Install these on Windows (10/11):
- Python 3.8+ (recommended: 3.12)
- Git
- Docker Desktop (required for the full system with PostgreSQL)
- PowerShell (already included in Windows)

Optional for data fetching (not needed for the demo):
- API keys (NASA, GBIF) if you plan to fetch real data instead of using built-in simulation

---

## 2) Open the Project Folder in PowerShell

Run PowerShell and change to the project folder:

```powershell
cd "C:\Users\prana\Desktop\Deforestation"
```

---

## 3) Option A — Quick Demo (No Database)
This mode launches an interactive web demo at http://localhost:5000/simple without requiring Docker or a database.

1. Create and activate a virtual environment
   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   If activation is blocked, run this once and retry activation:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```

2. Upgrade pip and install required packages
   ```powershell
   python -m pip install --upgrade pip
   pip install flask numpy pandas flask-cors
   ```

3. Start the demo
   ```powershell
   python quick_demo.py
   ```

4. Open the web interface
   - Web: http://localhost:5000/simple

5. Stop the server
   - Press Ctrl+C in the PowerShell window

---

## 4) Option B — Full System (Database + API + Dashboard)
This mode uses PostgreSQL with PostGIS through Docker and runs the full Flask API and dashboard.

1. Start Docker Desktop
   - Ensure Docker Desktop is running before continuing.

2. Start the database stack (PostgreSQL + pgAdmin)
   From the project folder:
   ```powershell
   docker compose up -d
   ```
   If your Docker supports the legacy syntax, this also works:
   ```powershell
   docker-compose up -d
   ```
   This starts:
   - Postgres/PostGIS on port 5432 with database deforestation_db
   - pgAdmin on http://localhost:8080 (email: admin@deforestation.local, password: admin123)

3. Create and activate a virtual environment
   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install Python dependencies
   Minimal API/database dependencies:
   ```powershell
   pip install -r backend/requirements.txt
   pip install flask-cors pandas numpy
   ```
   For ML features and training (optional):
   ```powershell
   pip install scikit-learn xgboost lightgbm tensorflow
   ```
   Notes:
   - If geopandas is needed later and fails to build, skip it for now; it is not required to run the API and dashboard.

5. Verify database connectivity (optional)
   The app is preconfigured to use:
   ```
   postgresql://postgres:yourpassword@localhost:5432/deforestation_db
   ```
   This matches docker-compose.yml. If you change credentials in docker-compose.yml, update the connection string in app.py accordingly.

6. Launch the full system
   Quick system setup without fetching new data or retraining models:
   ```powershell
   python launch_system.py --skip-data --skip-training
   ```
   Or start the Flask app directly (API + dashboard):
   ```powershell
   python app.py
   ```

7. Use the system
   - API Root: http://localhost:5000/
   - Dashboard: http://localhost:5000/dashboard
   - Simple Interface: http://localhost:5000/simple

8. Stop the system
   - Stop Flask: Ctrl+C in the PowerShell window
   - Stop containers:
     ```powershell
     docker compose down
     ```

---

## 5) Configuration (Optional)

- Database credentials: Change the connection string in app.py if needed.
- Real API keys for data fetchers: create a .env (do NOT commit this file)
  Example .env content (if you plan to run data/fetch_*.py scripts):
  ```env
  NASA_API_KEY=your_nasa_api_key
  GBIF_USER=your_gbif_username
  GBIF_PASSWORD=your_gbif_password
  ```
  To set temporary environment variables in PowerShell for the current session:
  ```powershell
  $env:NASA_API_KEY = "your_nasa_api_key"
  $env:GBIF_USER = "your_gbif_username"
  $env:GBIF_PASSWORD = "your_gbif_password"
  ```

---

## 6) Troubleshooting

- Port 5000 already in use:
  ```powershell
  netstat -ano | findstr ":5000"
  # Identify the PID and stop that process in Task Manager or with:
  Stop-Process -Id <PID> -Force
  ```

- PowerShell won’t activate the venv:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\.venv\Scripts\Activate.ps1
  ```

- Docker database not reachable:
  ```powershell
  docker compose ps
  docker compose logs postgres
  ```
  Ensure Docker Desktop is running and try again.

- Missing Python packages:
  ```powershell
  pip install -r backend/requirements.txt
  pip install flask-cors pandas numpy scikit-learn xgboost lightgbm tensorflow
  ```

---

## 7) Upload (Publish) This Project to GitHub (No AI Tools)

These steps use only Git, PowerShell, and the GitHub website.

1. Create a new repository on GitHub
   - Go to https://github.com and sign in
   - Click “New” repository
   - Name it (e.g., Deforestation)
   - Choose Public or Private
   - Do NOT initialize with a README if you plan to push an existing local repo (this folder already contains a Git repo)
   - Click “Create repository” and keep the page open — you’ll need the “https://github.com/<username>/<repo>.git” URL

2. In PowerShell, from the project folder
   ```powershell
   cd "C:\Users\prana\Desktop\Deforestation"
   git status
   ```
   If it says “not a git repository,” initialize it (this folder already has .git, so you can usually skip this):
   ```powershell
   git init
   ```

3. Configure your Git identity (only once per machine)
   ```powershell
   git config --global user.name "Your Name"
   git config --global user.email "you@example.com"
   ```

4. Ensure useful ignores are present
   - .gitignore in this project already excludes virtual envs, large data/model files, and secrets.
   - Do not commit .env, venv/, data dumps, or large model binaries.

5. Add, commit, and set the main branch
   ```powershell
   git add -A
   git commit -m "Initial commit: Deforestation Risk Analysis System"
   git branch -M main
   ```

6. Add the GitHub remote (replace with your actual URL)
   ```powershell
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   ```
   If a remote named origin already exists and you need to change it:
   ```powershell
   git remote set-url origin https://github.com/<your-username>/<your-repo>.git
   ```

7. Push to GitHub
   ```powershell
   git push -u origin main
   ```
   - If you created the repo on GitHub with a README and get a non-fast-forward error, run:
     ```powershell
     git pull --rebase origin main
     git push -u origin main
     ```

8. Verify
   - Refresh your GitHub repository page; your files should be visible

Optional (only if you later decide to version large binaries):
- Consider Git LFS for large model/data files: https://git-lfs.github.com/

---

## 8) Quick Reference Commands

- One-command run (recommended):
  - Double-click: run_full.cmd
  - Or from CMD/PowerShell:
    ```cmd
    run_full.cmd
    ```
  This will: create a venv, install deps, try Docker (optional), start the server without debug, and open your browser. If Docker isn’t available, the app still runs (simple interface and PDF reports work without the database).

- Start demo (no DB):
  ```powershell
  .\\.venv\\Scripts\\Activate.ps1
  python quick_demo.py
  ```

- Start full API + dashboard manually:
  ```powershell
  .\\.venv\\Scripts\\Activate.ps1
  python start_server.py
  ```
  Optional (for database-backed endpoints):
  ```powershell
  docker compose up -d   # or: docker-compose up -d
  ```

- Stop services:
  ```powershell
  # stop Flask
  Get-NetTCPConnection -LocalPort 5000 -State Listen | Select-Object -First 1 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
  # stop database services
  docker compose down
  ```

- First-time Git publish:
  ```powershell
  git add -A
  git commit -m "Initial commit"
  git branch -M main
  git remote add origin https://github.com/<your-username>/<your-repo>.git
  git push -u origin main
  ```

---

All steps above rely solely on local tools (PowerShell, Python, Git, Docker) and the GitHub website — no AI tools are used or required.

