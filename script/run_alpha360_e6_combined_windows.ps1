$ErrorActionPreference = 'Stop'

$BaseRoot = 'E:\qlibAssistant\.qlibAssistant\remote_runs\alpha360_probabilistic_matrix_260828'
$BaseStatus = Join-Path $BaseRoot 'pipeline_status.json'
$BaseData = 'E:\qlibAssistant\.qlibAssistant\remote_runs\alpha360_cross_stock_fold3_120m_260827\data'
$E0Run = 'E:\qlibAssistant\.qlibAssistant\remote_runs\alpha360_cross_stock_fold3_120m_v2_260828\run'
$E6Root = 'E:\qlibAssistant\.qlibAssistant\remote_runs\alpha360_e6_full_260828'
$E6Data = Join-Path $E6Root 'store'
$E6Output = Join-Path $E6Root 'run'
$Python = 'E:\Miniconda\envs\qlibass\python.exe'
$TrainEntry = Join-Path $BaseRoot 'script\train_alpha360_decoupled.py'
$E6TrainEntry = Join-Path $E6Root 'script\train_alpha360_cross_market.py'
$Materializer = Join-Path $BaseRoot 'script\materialize_alpha360_joint_checkpoints.py'
$Selector = Join-Path $E6Root 'script\select_alpha360_probabilistic_ensemble.py'
$Protocol = Join-Path $E6Root 'script\alpha360_experiments\fixed_fold3_probabilistic_cross_market_v1.json'
$PipelineStatus = Join-Path $E6Root 'pipeline_status.json'
$SelectionDirectory = Join-Path $E6Root 'selection_e0_e6'
$SelectionManifest = Join-Path $SelectionDirectory 'selection_manifest.json'
$TestEvaluation = Join-Path $E6Root 'test_evaluation_e0_e6'
$Transcript = Join-Path $E6Root 'pipeline.log'
$E0Candidate = Join-Path $BaseRoot 'E0_joint_three_leg'

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
    if ($LASTEXITCODE -ne 0) { throw "$Stage failed with exit code $LASTEXITCODE" }
}

function Test-PythonValidation([object[]]$Arguments, [string]$Stage) {
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Validate: $Stage"
    & $Python @Arguments 2>&1 | ForEach-Object { Write-Host $_ }
    $Code = $LASTEXITCODE
    return ($Code -eq 0)
}

function Move-DirectoryToArchive([string]$Path, [string]$Label) {
    if (-not (Test-Path $Path)) { return }
    $Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $Archive = Join-Path (Split-Path $Path -Parent) "${Label}_$Stamp"
    Move-Item -Path $Path -Destination $Archive
    Write-Host "Archived incomplete directory: $Path -> $Archive"
}

function Move-CandidateTestArtifactsToArchive([string]$Output) {
    $Names = @(
        'test_predictions.csv', 'test_summary.csv', 'test_access.json',
        'test_completion_audit.json', 'test_materialization_manifest.json'
    )
    $Files = @()
    foreach ($Name in $Names) {
        $Path = Join-Path $Output $Name
        if (Test-Path $Path) { $Files += Get-Item $Path }
    }
    $Files += @(Get-ChildItem -Path $Output -Filter 'test_*_daily_metrics.csv' -File -ErrorAction SilentlyContinue)
    if ($Files.Count -eq 0) { return }
    $Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $Archive = Join-Path $Output "incomplete_test_$Stamp"
    New-Item -ItemType Directory -Path $Archive | Out-Null
    foreach ($File in $Files) { Move-Item -Path $File.FullName -Destination $Archive }
}

function Test-CandidateSelection([string]$Name, [string]$Directory) {
    $Arguments = @(
        '-u', $Selector, 'validate-candidate', '--protocol', $Protocol,
        '--candidate-name', $Name, '--candidate-directory', $Directory
    )
    return (Test-PythonValidation -Arguments $Arguments -Stage "Selection candidate $Name")
}

function Test-CandidateTest([string]$Name) {
    $Arguments = @(
        '-u', $Selector, 'validate-candidate-test', '--manifest', $SelectionManifest,
        '--candidate-name', $Name
    )
    return (Test-PythonValidation -Arguments $Arguments -Stage "Test artifacts for $Name")
}

Start-Transcript -Path $Transcript -Append | Out-Null
try {
    Write-PipelineStatus @{
        status='waiting_for_e0_e5_selection'; base_status=$BaseStatus;
        test_access_authorized=$false; test_read=$false
    }
    while ($true) {
        if (-not (Test-Path $BaseStatus)) { throw "Base pipeline status is missing: $BaseStatus" }
        $Base = Get-Content $BaseStatus -Raw | ConvertFrom-Json
        if ($Base.status -eq 'selection_ready') { break }
        if ($Base.status -eq 'test_ready') {
            throw 'Base pipeline already opened Test; refusing to claim a blind E0-E6 comparison'
        }
        if ($Base.status -eq 'failed') { throw "Base E0-E5 pipeline failed: $($Base.error)" }
        Start-Sleep -Seconds 30
    }

    if (-not (Test-CandidateSelection 'E0_joint_three_leg' $E0Candidate)) {
        throw 'E0 candidate failed cross-market protocol authentication'
    }
    foreach ($Experiment in $Experiments) {
        $Output = Join-Path $BaseRoot $Experiment.Id
        if (-not (Test-CandidateSelection $Experiment.Id $Output)) {
            throw "Base candidate failed cross-market protocol authentication: $($Experiment.Id)"
        }
    }

    $E6Ready = $false
    if (Test-Path (Join-Path $E6Output 'status.json')) {
        $Existing = Get-Content (Join-Path $E6Output 'status.json') -Raw | ConvertFrom-Json
        if ($Existing.status -eq 'selection_ready') {
            $E6Ready = Test-CandidateSelection 'E6_a_us_four_head' $E6Output
            if (-not $E6Ready) {
                Move-DirectoryToArchive $E6Output 'run_invalid_selection_ready'
            }
        }
    }
    if (-not $E6Ready) {
        $Arguments = @(
            '-u', $E6TrainEntry, 'train', '--data', $E6Data, '--output', $E6Output,
            '--device', 'cuda', '--bf16', '--threads', '4',
            '--learning-rate', '0.0003', '--minimum-learning-rate', '0.000001',
            '--weight-decay', '0.0001', '--warmup-epochs', '3',
            '--warmup-start-factor', '0.3333333333333333',
            '--date-batch-size', '4', '--target-scale', '100',
            '--log-file', (Join-Path $E6Output 'train.log')
        )
        if (Test-Path (Join-Path $E6Output 'last_checkpoint.pt')) {
            $Arguments += '--resume'
        } elseif (Test-Path $E6Output) {
            Move-DirectoryToArchive $E6Output 'run_incomplete_without_checkpoint'
        }
        Write-PipelineStatus @{
            status='training_e6'; output=$E6Output; date_batch_size=4;
            gradient_clipping=$false; gradient_accumulation=$false;
            test_access_authorized=$false; test_read=$false
        }
        Invoke-CheckedPython $Arguments 'Train E6 A+US cross-market model for 50 epochs'
        if (-not (Test-CandidateSelection 'E6_a_us_four_head' $E6Output)) {
            throw 'Fresh E6 Selection candidate failed authentication'
        }
    }

    $SelectionReady = $false
    if (Test-Path $SelectionDirectory) {
        if (Test-Path $SelectionManifest) {
            $Validation = @('-u', $Selector, 'validate-selection', '--manifest', $SelectionManifest)
            $SelectionReady = Test-PythonValidation -Arguments $Validation -Stage 'E0-E6 freeze'
        }
        if (-not $SelectionReady) {
            Move-DirectoryToArchive $SelectionDirectory 'selection_e0_e6_invalid_or_incomplete'
        }
    }
    if (-not $SelectionReady) {
        $Arguments = @(
            '-u', $Selector, 'select', '--protocol', $Protocol,
            '--candidate', "E0_joint_three_leg=$E0Candidate"
        )
        foreach ($Experiment in $Experiments) {
            $Arguments += @('--candidate', "$($Experiment.Id)=$(Join-Path $BaseRoot $Experiment.Id)")
        }
        $Arguments += @('--candidate', "E6_a_us_four_head=$E6Output", '--output', $SelectionManifest)
        Write-PipelineStatus @{
            status='selecting_e0_e6_ensemble'; test_access_authorized=$false; test_read=$false
        }
        Invoke-CheckedPython $Arguments 'Select and freeze E0-E6 ensemble on Selection-valid'
        Invoke-CheckedPython @(
            '-u', $Selector, 'validate-selection', '--manifest', $SelectionManifest
        ) 'Authenticate E0-E6 Selection freeze'
    }

    $Frozen = Get-Content $SelectionManifest -Raw | ConvertFrom-Json
    $SelectedCandidates = @(
        $Frozen.selections.PSObject.Properties |
            ForEach-Object { @($_.Value.selected_components) } |
            Sort-Object -Unique
    )
    Write-PipelineStatus @{
        status='materializing_selected_test'; selected_candidates=$SelectedCandidates;
        selection_manifest=$SelectionManifest; test_access_authorized=$true;
        test_materialization_started=$false; test_read=$false
    }

    if ($SelectedCandidates -contains 'E0_joint_three_leg') {
        if (-not (Test-CandidateTest 'E0_joint_three_leg')) {
            Move-CandidateTestArtifactsToArchive $E0Candidate
            Write-PipelineStatus @{
                status='materializing_selected_test'; candidate='E0_joint_three_leg';
                test_access_authorized=$true; test_materialization_started=$true; test_read=$false
            }
            Invoke-CheckedPython @(
                '-u', $Materializer, 'evaluate-test', '--source-run', $E0Run,
                '--data', $BaseData, '--output', $E0Candidate,
                '--selection-manifest', $SelectionManifest,
                '--candidate-name', 'E0_joint_three_leg',
                '--device', 'cuda', '--bf16', '--threads', '4'
            ) 'Materialize selected E0 Test horizons'
        }
        if (-not (Test-CandidateTest 'E0_joint_three_leg')) { throw 'E0 Test audit failed' }
    }

    foreach ($Experiment in $Experiments) {
        if ($SelectedCandidates -notcontains $Experiment.Id) { continue }
        $Output = Join-Path $BaseRoot $Experiment.Id
        if (Test-CandidateTest $Experiment.Id) { continue }
        Move-CandidateTestArtifactsToArchive $Output
        Write-PipelineStatus @{
            status='materializing_selected_test'; candidate=$Experiment.Id;
            test_access_authorized=$true; test_materialization_started=$true; test_read=$false
        }
        $Arguments = @(
            '-u', $TrainEntry, 'evaluate-test', '--data', $BaseData, '--output', $Output,
            '--model-mode', $Experiment.Mode, '--device', 'cuda', '--threads', '4',
            '--target-scale', '100', '--selection-manifest', $SelectionManifest,
            '--candidate-name', $Experiment.Id,
            '--log-file', (Join-Path $Output 'test.log')
        )
        if ($Experiment.Horizon) { $Arguments += @('--horizon', $Experiment.Horizon) }
        Invoke-CheckedPython $Arguments "Evaluate selected Test horizons for $($Experiment.Id)"
        if (-not (Test-CandidateTest $Experiment.Id)) { throw "Test audit failed: $($Experiment.Id)" }
    }

    if ($SelectedCandidates -contains 'E6_a_us_four_head') {
        if (-not (Test-CandidateTest 'E6_a_us_four_head')) {
            Move-CandidateTestArtifactsToArchive $E6Output
            Write-PipelineStatus @{
                status='materializing_selected_test'; candidate='E6_a_us_four_head';
                test_access_authorized=$true; test_materialization_started=$true; test_read=$false
            }
            Invoke-CheckedPython @(
                '-u', $E6TrainEntry, 'evaluate-test', '--data', $E6Data,
                '--output', $E6Output, '--device', 'cuda', '--bf16', '--threads', '4',
                '--target-scale', '100', '--selection-manifest', $SelectionManifest,
                '--candidate-name', 'E6_a_us_four_head',
                '--log-file', (Join-Path $E6Output 'test.log')
            ) 'Evaluate selected E6 Test horizons'
        }
        if (-not (Test-CandidateTest 'E6_a_us_four_head')) { throw 'E6 Test audit failed' }
    }

    Write-PipelineStatus @{
        status='selected_candidate_test_complete'; selected_candidates=$SelectedCandidates;
        test_access_authorized=$true; test_materialization_started=$true; test_read=$true
    }

    $AggregateReady = $false
    if (Test-Path $TestEvaluation) {
        $Validation = @(
            '-u', $Selector, 'validate-test-evaluation', '--manifest', $SelectionManifest,
            '--output-directory', $TestEvaluation
        )
        $AggregateReady = Test-PythonValidation -Arguments $Validation -Stage 'E0-E6 Test aggregate'
        if (-not $AggregateReady) {
            Move-DirectoryToArchive $TestEvaluation 'test_evaluation_e0_e6_invalid_or_incomplete'
        }
    }
    if (-not $AggregateReady) {
        Invoke-CheckedPython @(
            '-u', $Selector, 'evaluate', '--manifest', $SelectionManifest,
            '--output-directory', $TestEvaluation
        ) 'Evaluate frozen E0-E6 ensemble on Test'
    }
    Invoke-CheckedPython @(
        '-u', $Selector, 'validate-test-evaluation', '--manifest', $SelectionManifest,
        '--output-directory', $TestEvaluation
    ) 'Authenticate complete E0-E6 Test evaluation'
    Write-PipelineStatus @{
        status='test_ready'; selected_candidates=$SelectedCandidates;
        selection_manifest=$SelectionManifest;
        selection_predictions=(Join-Path $SelectionDirectory 'selection_valid_ensemble_predictions.csv');
        test_predictions=(Join-Path $TestEvaluation 'test_predictions.csv');
        test_summary=(Join-Path $TestEvaluation 'test_summary.csv');
        test_access_authorized=$true; test_materialization_started=$true; test_read=$true
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
