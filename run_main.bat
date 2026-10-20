@echo off
REM Ejecuta main.py con el entorno virtual local del proyecto (Proyecto\venv)

cd /d "%~dp0"
set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] No se encontro el entorno virtual en "%~dp0venv".
    echo Crea uno con:  py -3.12 -m venv venv
    echo y luego:       venv\Scripts\python.exe -m pip install opencv-python mediapipe==0.10.18
    pause
    exit /b 1
)

if not exist "%~dp0memes" (
    echo [AVISO] No se encontro la carpeta "memes" junto a main.py.
)

echo Iniciando programa de deteccion de gestos...
"%PYTHON_EXE%" "%~dp0main.py"
pause
