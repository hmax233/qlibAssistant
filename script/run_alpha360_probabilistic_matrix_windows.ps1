$ErrorActionPreference = 'Stop'

$Root = 'E:\qlibAssistant\.qlibAssistant\remote_runs\alpha360_probabilistic_matrix_260828'
$Data = 'E:\qlibAssistant\.qlibAssistant\remote_runs\alpha360_cross_stock_fold3_120m_260827\data'
$E0Run = 'E:\qlibAssistant\.qlibAssistant\remote_runs\alpha360_cross_stock_fold3_120m_v2_260828\run'
$E0Status = Join-Path $E0Run 'status.json'
$E0Candidate = Join-Path $Root 'E0_joint_three_leg'
$Python = 'E:\Miniconda\envs\qlibass\python.exe'
$TrainEntry = Join-Path $Root 'script\train_alpha360_decoupled.py'
$Materializer = Join-Path $Root 'script\materialize_alpha360_joint_checkpoints.py'
$Selector = Join-Path $Root 'script\select_alpha360_probabilistic_ensemble.py'
$Protocol = Join-Path $Root 'script\alpha360_experiments\fixed_fold3_probabilistic_v1.json'
$PipelineStatus = Join-Path $Root 'pipeline_status.json'
$SelectionDirectory = Join-Path $Root 'selection'
$SelectionManifest = Join-Path $SelectionDirectory 'selection_manifest.json'
$TestEvaluation = Join-Path $Root 'test_evaluation'
$Transcript = Join-Path $Root 'pipeline.log'

$Experiments = @(
    @{Id='E1_shared_four_head'; Mode='shared_four_head'; Horizon=$null},
    @{Id='E2_single_open1_close2'; Mode='single_horizon'; Horizon='open1_close2'},
    @{Id='E3_single_close1_open2'; Mode='single_horizon'; Horizon='close1_open2'},
    @{Id='E4_single_open1_open2'; Mode='single_horizon'; Horizon='open1_open2'},
    @{Id='E5_single_close1_close2'; Mode='single_horizon'; Horizon='close1_close2'}
)

function Write-PipelineStatus([hashtable]$Value) {
    $Value['updated'] = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $Temporary = "$PipelineStatus.tmp"
    $Value | ConvertTo-Json -Depth 12 | Set-Content -Path $Temporary -Encoding UTF8
    Move-Item -Force $Temporary $PipelineStatus
}

function Invoke-CheckedPython([object[]]$Arguments, [string]$Stage) {
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Stage"
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Stage failed with exit code $LASTEXITCODE"
    }
}

Start-Transcript -Path $Transcript -Append | Out-Null
try {
    Write-PipelineStatus @{status='waiting_for_e0'; e0_status=$E0Status; test_read=$false}
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

    # E0 is one joint three-leg Gaussian, but every horizon must use its own
    # best-valid RankIC checkpoint.  This step opens Selection-valid only.
    if (-not (Test-Path (Join-Path $E0Candidate 'selection_valid_predictions.csv'))) {
        if (Test-Path $E0Candidate) {
            throw "Incomplete E0 candidate directory exists: $E0Candidate"
        }
        Write-PipelineStatus @{status='materializing_e0_selection'; test_read=$false}
        Invoke-CheckedPython @(
            '-u', $Materializer, 'selection',
            '--source-run', $E0Run, '--data', $Data, '--output', $E0Candidate,
            '--device', 'cuda', '--bf16', '--threads', '4'
        ) 'Materialize E0 horizon-specific Selection-valid predictions'
    }

    foreach ($Experiment in $Experiments) {
        $Output = Join-Path $Root $Experiment.Id
        $Status = Join-Path $Output 'status.json'
        if (Test-Path $Status) {
            $Existing = Get-Content $Status -Raw | ConvertFrom-Json
            if ($Existing.status -eq 'selection_ready') {
                Write-PipelineStatus @{
                    status='skipped_selection_ready'; experiment=$Experiment.Id; test_read=$false
                }
                continue
            }
        }
        $Arguments = @(
            '-u', $TrainEntry, 'train', '--data', $Data, '--output', $Output,
            '--model-mode', $Experiment.Mode, '--device', 'cuda', '--threads', '4',
            '--epochs', '50', '--learning-rate', '0.0003',
            '--min-learning-rate', '0.000001', '--weight-decay', '0.0001',
            '--warmup-epochs', '3', '--warmup-start-factor', '0.3333333333333333',
            '--date-batch-size', '4', '--target-scale', '100',
            '--log-file', (Join-Path $Output 'train.log')
        )
        if ($Experiment.Horizon) { $Arguments += @('--horizon', $Experiment.Horizon) }
        if (Test-Path (Join-Path $Output 'last_checkpoint.pt')) {
            $Arguments += '--resume'
        } elseif (Test-Path $Output) {
            $Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
            Rename-Item $Output ("$($Experiment.Id)_incomplete_without_checkpoint_$Stamp")
        }
        Write-PipelineStatus @{
            status='training'; experiment=$Experiment.Id; output=$Output;
            gradient_clipping=$false; gradient_accumulation=$false;
            date_batch_size=4; test_read=$false
        }
        Invoke-CheckedPython $Arguments "Train $($Experiment.Id)"
    }

    if (-not (Test-Path $SelectionManifest)) {
        New-Item -ItemType Directory -Force -Path $SelectionDirectory | Out-Null
        $SelectionArguments = @(
            '-u', $Selector, 'select', '--protocol', $Protocol,
            '--candidate', "E0_joint_three_leg=$E0Candidate"
        )
        foreach ($Experiment in $Experiments) {
            $SelectionArguments += @(
                '--candidate', "$($Experiment.Id)=$(Join-Path $Root $Experiment.Id)"
            )
        }
        $SelectionArguments += @('--output', $SelectionManifest)
        Write-PipelineStatus @{status='selecting_ensemble'; test_read=$false}
        Invoke-CheckedPython $SelectionArguments 'Select ensemble on Selection-valid and freeze manifest'
    }

    # Test access begins only after the immutable component manifest exists.
    $Frozen = Get-Content $SelectionManifest -Raw | ConvertFrom-Json
    $SelectedCandidates = @(
        $Frozen.selections.PSObject.Properties |
            ForEach-Object { @($_.Value.selected_components) } |
            Sort-Object -Unique
    )
    Write-PipelineStatus @{
        status='materializing_selected_test'; selected_candidates=$SelectedCandidates;
        selection_manifest=$SelectionManifest; test_read=$true
    }

    if ($SelectedCandidates -contains 'E0_joint_three_leg') {
        if (-not (Test-Path (Join-Path $E0Candidate 'test_predictions.csv'))) {
            Invoke-CheckedPython @(
                '-u', $Materializer, 'evaluate-test', '--source-run', $E0Run,
                '--data', $Data, '--output', $E0Candidate,
                '--selection-manifest', $SelectionManifest,
                '--candidate-name', 'E0_joint_three_leg',
                '--device', 'cuda', '--bf16', '--threads', '4'
            ) 'Materialize selected E0 Test horizons'
        }
    }

    foreach ($Experiment in $Experiments) {
        if ($SelectedCandidates -notcontains $Experiment.Id) { continue }
        $Output = Join-Path $Root $Experiment.Id
        if (Test-Path (Join-Path $Output 'test_predictions.csv')) { continue }
        $Arguments = @(
            '-u', $TrainEntry, 'evaluate-test', '--data', $Data, '--output', $Output,
            '--model-mode', $Experiment.Mode, '--device', 'cuda', '--threads', '4',
            '--target-scale', '100', '--selection-manifest', $SelectionManifest,
            '--candidate-name', $Experiment.Id,
            '--log-file', (Join-Path $Output 'test.log')
        )
        if ($Experiment.Horizon) { $Arguments += @('--horizon', $Experiment.Horizon) }
        Invoke-CheckedPython $Arguments "Evaluate selected Test horizons for $($Experiment.Id)"
    }

    if (-not (Test-Path (Join-Path $TestEvaluation 'test_predictions.csv'))) {
        Invoke-CheckedPython @(
            '-u', $Selector, 'evaluate', '--manifest', $SelectionManifest,
            '--output-directory', $TestEvaluation
        ) 'Evaluate frozen ensemble on Test'
    }
    Write-PipelineStatus @{
        status='test_ready'; experiments=@($Experiments.Id);
        selected_candidates=$SelectedCandidates; selection_manifest=$SelectionManifest;
        selection_predictions=(Join-Path $SelectionDirectory 'selection_valid_ensemble_predictions.csv');
        test_predictions=(Join-Path $TestEvaluation 'test_predictions.csv');
        test_summary=(Join-Path $TestEvaluation 'test_summary.csv'); test_read=$true
    }
} catch {
    Write-PipelineStatus @{
        status='failed'; error=$_.Exception.Message;
        selection_manifest_exists=(Test-Path $SelectionManifest)
    }
    throw
} finally {
    Stop-Transcript | Out-Null
}
