@echo off
REM Starts the Ollama-powered FRONTLINE app from its project-local virtual environment.
cd /d "%~dp0"
.venv\Scripts\python.exe -m streamlit run app.py --server.headless=true --server.port=8501
