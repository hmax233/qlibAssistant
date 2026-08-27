param([switch]$ResumeBlindProtocolMigration)
$ErrorActionPreference = 'Stop'

$TaskName = 'Qlib_Alpha360_V2_Full50_260828'
$Root = 'E:\qlibAssistant\.qlibAssistant\remote_runs\alpha360_cross_stock_fold3_120m_v2_260828'
$Data = 'E:\qlibAssistant\.qlibAssistant\remote_runs\alpha360_cross_stock_fold3_120m_260827\data'
$Python = 'E:\Miniconda\envs\qlibass\python.exe'
$Entry = Join-Path $Root 'script\train_alpha360_cross_stock.py'
$Output = Join-Path $Root 'run'
$Log = Join-Path $Root 'train.log'

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    throw "Task $TaskName already exists; inspect it instead of starting a duplicate."
}
if ($ResumeBlindProtocolMigration -and -not (Test-Path (Join-Path $Output 'last_checkpoint.pt'))) {
    throw 'Blind-protocol resume requested but last_checkpoint.pt is missing.'
}

$Arguments = @(
    '-u', $Entry, 'train',
    '--data', $Data,
    '--output', $Output,
    '--device', 'cuda',
    '--threads', '4',
    '--epochs', '50',
    '--learning-rate', '0.0003',
    '--min-learning-rate', '0.000001',
    '--warmup-epochs', '3',
    '--warmup-start-factor', '0.3333333333333333',
    '--stock-embedding-width', '64',
    '--date-batch-size', '4',
    '--selection-metric', 'close1_close2_rank_ic',
    '--log-file', $Log
)
if ($ResumeBlindProtocolMigration) {
    $Arguments += @('--resume', '--resume-blind-protocol-migration')
}
$Arguments = $Arguments -join ' '

$Action = New-ScheduledTaskAction -Execute $Python -Argument $Arguments -WorkingDirectory $Root
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 6) `
    -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Principal $Principal `
    -Settings $Settings `
    -Description 'Alpha360 v2 full 50 epochs; held-out Test remains locked until the frozen ensemble manifest.' | Out-Null

Start-ScheduledTask -TaskName $TaskName
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName, State
