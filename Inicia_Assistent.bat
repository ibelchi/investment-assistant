@echo off
title Assistent Personal d'Anàlisi d'Inversió
echo ===================================================
echo Iniciant l'Assistent Personal d'Analisi d'Inversio...
echo ===================================================
echo.
echo Carregant l'entorn virtual i Streamlit...

cd /d "%~dp0"
call venv\Scripts\activate.bat
streamlit run app.py

echo.
echo S'ha tancat l'aplicacio.
pause
