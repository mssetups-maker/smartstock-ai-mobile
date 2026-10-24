@echo off
chcp 65001 >nul
cd /d "%~dp0"
python --version >nul 2>&1
if errorlevel 1 (
  echo Python이 설치되어 있지 않습니다. https://www.python.org/downloads/ 에서 Python을 설치해 주세요.
  pause
  exit /b 1
)
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo 패키지 설치 오류가 발생했습니다.
  pause
  exit /b 1
)
python -m streamlit run app.py
pause
