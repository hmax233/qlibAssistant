param(
    [string]$Root = 'E:\qlibAssistant\.qlibAssistant\remote_runs\alpha360_cross_stock_fold3_120m_260827',
    [string]$Name = 'Qlib_Alpha360_CrossStock_Fold3_120m_260827'
)
$ErrorActionPreference = 'Stop'
if (Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue) {
    throw "Task already exists; inspect it instead of launching a duplicate: $Name"
}
$Runner = Join-Path $Root 'script\run_alpha360_cross_stock_windows.ps1'
$Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$Runner`" -Root `"$Root`""
$Action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $Arguments -WorkingDirectory $Root
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 48) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $Name -Action $Action -Principal $Principal -Settings $Settings -Description 'User priority Alpha360 temporal/cross-stock Transformer; frozen stock embedding; joint Gaussian NLL; one-shot.' | Out-Null
Start-ScheduledTask -TaskName $Name
Get-ScheduledTask -TaskName $Name | Select-Object TaskName,State
