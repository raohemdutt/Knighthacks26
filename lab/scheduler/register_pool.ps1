# Registers the lab pool with Windows Task Scheduler: ONE task, AtlasLab-Pool, that starts
# python -m lab pool when you log on and starts it again whenever it exits. This script never
# starts the pool itself; it starts at your next logon, or now with:
#
#   Start-ScheduledTask -TaskName AtlasLab-Pool
#
# Run it once, from the clone the pool should run in (usually atlas\), in a normal PowerShell:
#
#   powershell -ExecutionPolicy Bypass -File lab\scheduler\register_pool.ps1
#
# Running it again replaces the task (for example after the clone moves). If Windows answers
# "access denied", run it from an elevated PowerShell instead.
#
# Remove the task with:  lab\scheduler\register_pool.ps1 -Remove
# That also stops the task if it is running. To let the running sittings hand off first, run
# python -m lab stop, wait until python -m lab pool status says no pool runs, then -Remove,
# then python -m lab resume.
#
# What the task runs. It starts lab\scheduler\run_pool.cmd through conhost --headless, so no
# console window appears, in your interactive session (the agent CLI needs your login). That file
# starts python -m lab pool from this clone, appends everything it prints to
# <state dir>\logs\pool.log, and starts the pool again 60 seconds after it exits, whatever the
# reason. The task is also set to restart if that file itself ends in failure (every minute, up to
# 999 times, the most Task Scheduler allows), never to stop for running too long (the 72 hour
# default would end a pool that works), to run on battery, and to ignore a second start while one
# runs.
#
# STOP halts the pool without touching the task. python -m lab stop creates <state dir>\STOP; the
# pool starts no more sittings and exits once the running ones have ended; run_pool.cmd then waits
# with the task still registered and running. python -m lab resume removes the file and
# run_pool.cmd starts the pool again within 30 seconds. The state dir is LAB_STATE_DIR when it is
# set and not empty, and this clone's lab\state otherwise; run_pool.cmd, the pool it starts and
# python -m lab stop, resume and pool status all read it the same way, and run_pool.cmd leaves the
# variable as it is. So run stop and resume from a shell with the same LAB_STATE_DIR as your user
# environment (the one the task inherits at logon), or with none when that has none.
#
# Rehearse without starting anything: python -m lab pool --dry-run prints the schedule the
# pool would start and starts nothing; python -m lab pool status says whether a pool runs.

param([switch]$Remove)

$ErrorActionPreference = "Stop"

$taskName = "AtlasLab-Pool"
$taskPath = "\"
$wrapper = Join-Path $PSScriptRoot "run_pool.cmd"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$description = "Keeps python -m lab pool running for the Atlas lab. Halt it with python -m lab stop, resume with python -m lab resume. Registered by lab\scheduler\register_pool.ps1."

function Get-PoolTask {
    Get-ScheduledTask -TaskName $taskName -TaskPath $taskPath -ErrorAction SilentlyContinue
}

if ($Remove) {
    try {
        $existing = Get-PoolTask
        if ($null -eq $existing) {
            Write-Output "not registered: $taskName"
            exit 0
        }
        if ($existing.State -eq "Running") {
            Stop-ScheduledTask -TaskName $taskName -TaskPath $taskPath
            Write-Output "stopped the running $taskName"
        }
        Unregister-ScheduledTask -TaskName $taskName -TaskPath $taskPath -Confirm:$false
    } catch {
        Write-Output "FAILED to remove ${taskName}: $($_.Exception.Message)"
        exit 1
    }
    if ($null -eq (Get-PoolTask)) {
        Write-Output "removed $taskName"
        exit 0
    }
    Write-Output "SURVIVED ${taskName}: still registered"
    exit 1
}

$settingsArgs = @{
    AllowStartIfOnBatteries    = $true
    DontStopIfGoingOnBatteries = $true
    MultipleInstances          = "IgnoreNew"
    ExecutionTimeLimit         = [TimeSpan]::Zero
    RestartCount               = 999
    RestartInterval            = (New-TimeSpan -Minutes 1)
}

try {
    if (-not (Test-Path -LiteralPath $wrapper)) {
        throw "run_pool.cmd is not next to this script: $wrapper"
    }
    $conhost = Join-Path $env:SystemRoot "System32\conhost.exe"
    $action = New-ScheduledTaskAction -Execute $conhost -Argument "--headless `"$wrapper`"" -WorkingDirectory $repoRoot
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
    $principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet @settingsArgs
    Register-ScheduledTask -TaskName $taskName -TaskPath $taskPath -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description $description -Force | Out-Null
} catch {
    Write-Output "FAILED to register ${taskName}: $($_.Exception.Message)"
    Write-Output "If Windows says access is denied, run this script again from an elevated PowerShell."
    exit 1
}

$registered = Get-PoolTask
if ($null -eq $registered) {
    Write-Output "MISSING ${taskName}: not registered"
    exit 1
}
Write-Output "registered ${taskName}: starts python -m lab pool at logon of $user and starts it again whenever it exits"
Write-Output "state: $($registered.State); nothing was started"
Write-Output "log: $(Join-Path $repoRoot 'lab\state\logs\pool.log'), or logs\pool.log under LAB_STATE_DIR when your user environment sets it"
Write-Output "halt it without touching the task: python -m lab stop; let it run again: python -m lab resume"
