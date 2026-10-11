@echo off
rem Keeps python -m lab pool running. The scheduled task that register_pool.ps1 creates runs this file.
rem
rem From the repo root (the clone this file sits in) it does three things:
rem   1. Starts python -m lab pool, using the clone's .venv python when there is one and python from
rem      PATH otherwise. Everything the pool prints is appended to logs\pool.log in the state dir,
rem      with a line from this file each time it starts the pool, sees it exit or waits for STOP.
rem   2. When the pool exits, for any reason, waits RESTART_SECONDS and starts it again.
rem   3. While STOP exists in the state dir it starts nothing and looks again every POLL_SECONDS.
rem      The pool itself starts no more sittings once STOP is set and exits when the running ones
rem      have ended; it does not wait for a resume. So python -m lab stop halts the pool,
rem      python -m lab resume lets this file start it again, and the scheduled task is never
rem      touched.
rem Arguments are handed to the pool: run_pool.cmd --units 12 runs python -m lab pool --units 12.
rem
rem The state dir is the one python -m lab stop, resume and pool status use: LAB_STATE_DIR when it is
rem set and not empty, this clone's lab\state otherwise. A relative LAB_STATE_DIR is read from the
rem repo root, as the pool started from here reads it. The variable is left as it is, so the pool
rem keeps its ledger and lock in the same dir as the STOP file and the log this file watches and
rem writes, and python -m lab stop from a shell with the same LAB_STATE_DIR halts it.
setlocal
cd /d "%~dp0..\.." || exit /b 1
set "RESTART_SECONDS=60"
set "POLL_SECONDS=30"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
if exist ".venv\Scripts\python.exe" (set "PY=.venv\Scripts\python.exe") else (set "PY=python")
set "STATE=lab\state"
if defined LAB_STATE_DIR set "STATE=%LAB_STATE_DIR%"
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
