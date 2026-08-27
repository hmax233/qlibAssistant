$ErrorActionPreference = 'Stop'

$TaskName = 'Qlib_Alpha360_E6_Store_260828'
$Root = 'E:\qlibAssistant\.qlibAssistant\remote_runs\alpha360_e6_full_260828'
$Worker = Join-Path $Root 'run_alpha360_e6_store_windows.ps1'
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    throw "Task already exists: $TaskName"
}
$Action = New-ScheduledTaskAction `
    -Execute 'powershell.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$Worker`"" `
    -WorkingDirectory $Root
$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 6) `
    -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries
Register-ScheduledTask `
    -TaskName $TaskName -Action $Action -Principal $Principal -Settings $Settings `
    -Description 'Build the full E6 A+US Alpha360 store with strict previous-US-close alignment.' | Out-Null
Start-ScheduledTask -TaskName $TaskName
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName,State
