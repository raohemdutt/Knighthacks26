@echo off
rem Keeps python -m lab pool running. The scheduled task that register_pool.ps1 creates runs this file.
rem
rem From the repo root (the clone this file sits in) it does three things:
rem   1. Starts python -m lab pool, using the clone's .venv python when there is one and python from
rem      PATH otherwise. Everything the pool prints is appended to lab\state\logs\pool.log, with a
rem      line from this file each time it starts the pool, sees it exit or waits for STOP.
rem   2. When the pool exits, for any reason, waits RESTART_SECONDS and starts it again.
rem   3. While lab\state\STOP exists it starts nothing and looks again every POLL_SECONDS. The pool
rem      itself starts no more sittings once STOP is set and exits when the running ones have ended;
rem      it does not wait for a resume. So python -m lab stop halts the pool, python -m lab resume
rem      lets this file start it again, and the scheduled task is never touched.
rem Arguments are handed to the pool: run_pool.cmd --units 12 runs python -m lab pool --units 12.
rem
rem The pool always uses this clone's lab\state, so a LAB_STATE_DIR set in the user environment is
rem cleared here: it would move the ledger and the pool lock away from the STOP file and the log
rem that this file watches and writes.
setlocal
cd /d "%~dp0..\.." || exit /b 1
set "RESTART_SECONDS=60"
set "POLL_SECONDS=30"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set LAB_STATE_DIR=
if exist ".venv\Scripts\python.exe" (set "PY=.venv\Scripts\python.exe") else (set "PY=python")
set "STATE=lab\state"
set "LOG=%STATE%\logs\pool.log"
if not exist "%STATE%\logs" mkdir "%STATE%\logs"
set "WAS_STOPPED="

:again
if exist "%STATE%\STOP" goto stopped
if defined WAS_STOPPED call :note "STOP is gone, starting the pool"
set "WAS_STOPPED="
call :note "starting python -m lab pool"
"%PY%" -m lab pool %* >> "%LOG%" 2>&1
set "CODE=%errorlevel%"
call :note "python -m lab pool exited with code %CODE%"
if exist "%STATE%\STOP" goto stopped
call :note "starting it again in %RESTART_SECONDS% seconds"
call :wait %RESTART_SECONDS%
goto again

:stopped
if not defined WAS_STOPPED call :note "STOP exists, so the pool stays down until it is removed (python -m lab resume)"
set "WAS_STOPPED=1"
call :wait %POLL_SECONDS%
goto again

:note
rem One line in the log: the local time (the clock the pool's own lines use), then the message.
set "NOW=%date% %time%"
for /f "delims=" %%t in ('powershell -NoProfile -NonInteractive -Command "Get-Date -Format s"') do set "NOW=%%t"
>> "%LOG%" echo %NOW% run_pool: %~1
exit /b 0

:wait
rem ping to the local address waits one second per count and, unlike timeout, works without a console.
set /a "PINGS=%~1+1"
ping -n %PINGS% 127.0.0.1 > nul
exit /b 0
