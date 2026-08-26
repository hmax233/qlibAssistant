param(
    [string]$Root = 'E:\qlibAssistant\.qlibAssistant\remote_runs\alpha360_cross_stock_fold3_120m_260827',
    [string]$Name = 'Qlib_Alpha360_Finalize_260827'
)
$ErrorActionPreference = 'Stop'
if (Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue) {
    throw "One-shot finalizer already exists: $Name"
}
$Python = 'E:\Miniconda\envs\qlibass\python.exe'
$Arguments = "-u $Root\script\finalize_alpha360_cross_stock.py --root $Root"
$Action = New-ScheduledTaskAction -Execute $Python -Argument $Arguments -WorkingDirectory $Root
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 49) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $Name -Action $Action -Principal $Principal -Settings $Settings -Description 'One-shot lightweight Alpha360 completion monitor; audit and package only after training finishes; never restarts training.' | Out-Null
Start-ScheduledTask -TaskName $Name
Get-ScheduledTask -TaskName $Name | Select-Object TaskName,State
