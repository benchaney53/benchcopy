@echo off
REM Start the local development server and open browser
REM Double-click this file to start developing

echo Starting Bench Development Server...
echo.

start "" C:\ProgramData\anaconda3\python.exe serve.py

timeout /t 2 /nobreak >nul

start "" http://localhost:8000/

echo.
echo Server is running at http://localhost:8000/
echo Press any key to stop the server...
pause >nul
