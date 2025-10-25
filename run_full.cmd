@echo off
REM Simple one-command runner for CMD users (no need to change execution policy)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_full.ps1"
