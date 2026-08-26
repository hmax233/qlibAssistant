param(
    [Parameter(Mandatory=$true)][string]$Root,
    [string]$Python = 'E:\Miniconda\envs\qlibass\python.exe'
)
$ErrorActionPreference = 'Stop'
$Entry = Join-Path $Root 'script\train_alpha360_cross_stock.py'
$Provider = Join-Path $Root 'provider'
$Data = Join-Path $Root 'data'
$Log = Join-Path $Root 'train.log'
# One foreground child at a time. Benchmark failure prevents formal training.
& $Python -u $Entry pipeline --provider $Provider --data $Data --output (Join-Path $Root 'benchmark') --benchmark-only --log-file $Log
if ($LASTEXITCODE -ne 0) { throw "Export/benchmark failed with exit code $LASTEXITCODE. See $Log" }
& $Python -u $Entry train --data $Data --output (Join-Path $Root 'run') --log-file $Log
if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE. See $Log" }
