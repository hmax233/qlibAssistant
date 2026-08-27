$ErrorActionPreference = 'Stop'

$TaskName = 'Qlib_Alpha360_E6_Combined_260828'
$Root = 'E:\qlibAssistant\.qlibAssistant\remote_runs\alpha360_e6_full_260828'
$Worker = Join-Path $Root 'run_alpha360_e6_combined_windows.ps1'

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    throw "Task $TaskName already exists; inspect it instead of starting a duplicate."
}
if (-not (Test-Path $Worker)) { throw "Worker script is missing: $Worker" }

$Action = New-ScheduledTaskAction `
    -Execute 'powershell.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$Worker`"" `
    -WorkingDirectory $Root
$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 12) `
    -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Principal $Principal `
    -Settings $Settings `
    -Description 'Wait for E0-E5 Selection, train E6, freeze E0-E6, then open held-out Test once.' | Out-Null
Start-ScheduledTask -TaskName $TaskName
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName, State
