@echo off
cd /d "%~dp0"
if exist "%~dp0..\work\venv\Scripts\python.exe" (
  "%~dp0..\work\venv\Scripts\python.exe" "%~dp0ES_Simulation_Demo.py"
) else if exist "%~dp0.demo_venv\Scripts\python.exe" (
  "%~dp0.demo_venv\Scripts\python.exe" "%~dp0ES_Simulation_Demo.py"
) else (
  python "%~dp0ES_Simulation_Demo.py"
)
if errorlevel 1 (
  echo Simulation failed. Check that the project dataset and virtual environment are present.
) else (
  start "" "%~dp0ES_Simulation_Preview.png"
)
pause
