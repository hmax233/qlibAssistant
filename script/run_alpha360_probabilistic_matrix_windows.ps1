$ErrorActionPreference = 'Stop'

$Root = 'E:\qlibAssistant\.qlibAssistant\remote_runs\alpha360_probabilistic_matrix_260828'
$Data = 'E:\qlibAssistant\.qlibAssistant\remote_runs\alpha360_cross_stock_fold3_120m_260827\data'
$E0Status = 'E:\qlibAssistant\.qlibAssistant\remote_runs\alpha360_cross_stock_fold3_120m_v2_260828\run\status.json'
$Python = 'E:\Miniconda\envs\qlibass\python.exe'
$Entry = Join-Path $Root 'script\train_alpha360_decoupled.py'
$PipelineStatus = Join-Path $Root 'pipeline_status.json'

function Write-PipelineStatus([hashtable]$Value) {
    $Value['updated'] = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $Temporary = "$PipelineStatus.tmp"
    $Value | ConvertTo-Json -Depth 8 | Set-Content -Path $Temporary -Encoding UTF8
    Move-Item -Force $Temporary $PipelineStatus
}

Write-PipelineStatus @{status='waiting_for_e0'; e0_status=$E0Status}
while ($true) {
    if (-not (Test-Path $E0Status)) {
        throw "E0 status is missing: $E0Status"
    }
    $E0 = Get-Content $E0Status -Raw | ConvertFrom-Json
    if ($E0.status -eq 'completed') { break }
    if ($E0.status -in @('failed', 'error')) {
        throw "E0 failed; refusing to start E1-E5"
    }
    Start-Sleep -Seconds 30
}

$Experiments = @(
    @{Id='E1_shared_four_head'; Mode='shared_four_head'; Horizon=$null},
    @{Id='E2_single_open1_close2'; Mode='single_horizon'; Horizon='open1_close2'},
    @{Id='E3_single_close1_open2'; Mode='single_horizon'; Horizon='close1_open2'},
    @{Id='E4_single_open1_open2'; Mode='single_horizon'; Horizon='open1_open2'},
    @{Id='E5_single_close1_close2'; Mode='single_horizon'; Horizon='close1_close2'}
)

foreach ($Experiment in $Experiments) {
    $Output = Join-Path $Root $Experiment.Id
    $Status = Join-Path $Output 'status.json'
    if (Test-Path $Status) {
        $Existing = Get-Content $Status -Raw | ConvertFrom-Json
        if ($Existing.status -eq 'selection_ready') {
            Write-PipelineStatus @{status='skipped_completed'; experiment=$Experiment.Id}
            continue
        }
    }
    $Arguments = @(
        '-u', $Entry, 'train',
        '--data', $Data,
        '--output', $Output,
        '--model-mode', $Experiment.Mode,
        '--device', 'cuda',
        '--threads', '4',
        '--epochs', '50',
        '--learning-rate', '0.0003',
        '--min-learning-rate', '0.000001',
        '--weight-decay', '0.0001',
        '--warmup-epochs', '3',
        '--warmup-start-factor', '0.3333333333333333',
        '--date-batch-size', '4',
        '--target-scale', '100',
        '--log-file', (Join-Path $Output 'train.log')
    )
    if ($Experiment.Horizon) {
        $Arguments += @('--horizon', $Experiment.Horizon)
    }
    if (Test-Path (Join-Path $Output 'last_checkpoint.pt')) {
        $Arguments += '--resume'
    } elseif (Test-Path $Output) {
        $Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
        Rename-Item $Output ("$($Experiment.Id)_incomplete_without_checkpoint_$Stamp")
    }
    Write-PipelineStatus @{
        status='training'; experiment=$Experiment.Id; output=$Output;
        gradient_clipping=$false; gradient_accumulation=$false; date_batch_size=4
    }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        Write-PipelineStatus @{status='failed'; experiment=$Experiment.Id; exit_code=$LASTEXITCODE}
        throw "$($Experiment.Id) failed with exit code $LASTEXITCODE"
    }
}

Write-PipelineStatus @{status='selection_ready'; experiments=@($Experiments.Id); test_read=$false}

