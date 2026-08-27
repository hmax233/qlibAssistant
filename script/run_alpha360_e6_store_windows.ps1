$ErrorActionPreference = 'Stop'

$Root = 'E:\qlibAssistant\.qlibAssistant\remote_runs\alpha360_e6_full_260828'
$Python = 'E:\Miniconda\envs\qlibass\python.exe'
$Builder = Join-Path $Root 'build_alpha360_cross_market_store.py'
$AStore = 'E:\qlibAssistant\.qlibAssistant\remote_runs\alpha360_cross_stock_fold3_120m_260827\data'
$UsRaw = Join-Path $Root 'us_raw'
$Universe = Join-Path $Root 'us_universe.csv'
$Output = Join-Path $Root 'store'
$Status = Join-Path $Root 'pipeline_status.json'
$Log = Join-Path $Root 'build.log'

function Write-Status([hashtable]$Value) {
    $Value['updated'] = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $Temporary = "$Status.tmp"
    $Value | ConvertTo-Json -Depth 10 | Set-Content $Temporary -Encoding UTF8
    Move-Item -Force $Temporary $Status
}

Start-Transcript -Path $Log -Append | Out-Null
try {
    if (Test-Path (Join-Path $Output 'manifest.json')) {
        $Manifest = Get-Content (Join-Path $Output 'manifest.json') -Raw | ConvertFrom-Json
        if ($Manifest.status -ne 'complete') {
            throw 'Existing E6 store manifest is not complete.'
        }
        Write-Status @{status='complete'; output=$Output; resumed=$true}
        exit 0
    }
    Write-Status @{status='building'; output=$Output; us_files=(Get-ChildItem $UsRaw -Filter *.parquet).Count}
    & $Python -u $Builder `
        --a-store $AStore `
        --us-raw $UsRaw `
        --universe $Universe `
        --output $Output
    if ($LASTEXITCODE -ne 0) {
        throw "E6 store builder failed with exit code $LASTEXITCODE"
    }
    $Manifest = Get-Content (Join-Path $Output 'manifest.json') -Raw | ConvertFrom-Json
    if ($Manifest.status -ne 'complete') {
        throw 'Builder exited successfully but complete manifest is absent.'
    }
    Write-Status @{
        status='complete'; output=$Output;
        input_fingerprint=$Manifest.input_fingerprint;
        us_stock_count=$Manifest.us_source.selected_stock_count;
        coverage=$Manifest.coverage
    }
} catch {
    Write-Status @{status='failed'; error=$_.Exception.Message; output=$Output}
    throw
} finally {
    Stop-Transcript | Out-Null
}
