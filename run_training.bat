@echo off
REM Activate venv if exists, then run training
if exist .venv\Scripts\activate.bat (
  call .venv\Scripts\activate.bat
)
python -m src.train_model
pause
