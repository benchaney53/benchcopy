@echo off
echo Generating tools manifest...
python generate-manifest.py
if %ERRORLEVEL% EQU 0 (
    echo.
    echo Success! Manifest generated.
    echo You can now commit and push the changes.
) else (
    echo.
    echo Error generating manifest.
)
pause
