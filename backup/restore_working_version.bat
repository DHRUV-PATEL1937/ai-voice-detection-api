@echo off
echo ========================================
echo RESTORING TO WORKING VERSION
echo ========================================
echo.

echo Stopping any running servers...
taskkill /F /IM python.exe 2>nul

echo.
echo Removing deployment files...
del Procfile 2>nul
del railway.json 2>nul
del nixpacks.toml 2>nul
del render.yaml 2>nul
rmdir /s /q scripts\download_model.py 2>nul

echo.
echo Restoring original files...
echo Please manually replace the files with the code above

echo.
echo ========================================
echo RESTORATION COMPLETE
echo ========================================
echo.
echo Next steps:
echo 1. Copy the original app/main.py code above
echo 2. Copy the original app/detector.py code above  
echo 3. Copy the original static/script.js code above
echo 4. Run: uvicorn app.main:app --reload
echo.
pause