param(
    [string]$Root = 'E:\qlibAssistant\.qlibAssistant\remote_runs\alpha360_cross_stock_fold3_120m_260827',
    [string]$Name = 'Qlib_Alpha360_Resume_IO_260827'
)
$ErrorActionPreference = 'Stop'
if (Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue) {
    throw "Resume task already exists; inspect before retrying: $Name"
}
$Active = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -like "*$Root*train_alpha360_cross_stock.py*"
}
if ($Active) { throw 'A training process is still active; refusing a duplicate' }
if (!(Test-Path "$Root\run\last_checkpoint.pt")) { throw 'Missing complete checkpoint' }
if (Test-Path "$Root\data\last_failure.json") { throw 'Archive and review failure before explicitly resuming' }
$Python = 'E:\Miniconda\envs\qlibass\python.exe'
$Arguments = "-u $Root\script\train_alpha360_cross_stock.py train --data $Root\data --output $Root\run --device cuda --epochs 50 --patience 10 --threads 4 --resume --resume-io-repair --log-file $Root\resume_io.log"
$Action = New-ScheduledTaskAction -Execute $Python -Argument $Arguments -WorkingDirectory $Root
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 48) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $Name -Action $Action -Principal $Principal -Settings $Settings -Description 'Explicit checkpoint continuation after atomic status publication repair; preserves optimizer/RNG and logs source provenance.' | Out-Null
Start-ScheduledTask -TaskName $Name
Get-ScheduledTask -TaskName $Name | Select-Object TaskName,State
