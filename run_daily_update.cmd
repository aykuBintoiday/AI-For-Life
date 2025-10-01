@echo off
setlocal
cd /d D:\FEandBE\travel-ai
if not exist ".\logs" mkdir ".\logs"
".\.venv\Scripts\python.exe" ".\daily_update.py" >> ".\logs\daily_update.log" 2>&1
endlocal
set CE_MODEL=BAAI/bge-reranker-v2-m3
set DIVERSITY_COS=0.9
