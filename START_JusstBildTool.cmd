@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>&1
if %errorlevel%==0 (
  set "PY=py"
) else (
  where python >nul 2>&1
  if %errorlevel%==0 (
    set "PY=python"
  ) else (
    echo Python wurde nicht gefunden.
    echo.
    echo Das Tool kann ohne vorhandenes Python nicht gestartet werden.
    pause
    exit /b 1
  )
)

%PY% -c "import PIL, cv2, numpy" >nul 2>&1
if not %errorlevel%==0 (
  echo Benoetigte Bildmodule werden einmalig eingerichtet...
  %PY% -m pip install --user --disable-pip-version-check pillow opencv-python numpy
  if not %errorlevel%==0 (
    echo.
    echo Installation der Bildmodule ist fehlgeschlagen.
    pause
    exit /b 1
  )
)

%PY% "%~dp0JusstBildTool.py"
if not %errorlevel%==0 (
  echo.
  echo Das BildTool wurde mit einem Fehler beendet.
  pause
)
