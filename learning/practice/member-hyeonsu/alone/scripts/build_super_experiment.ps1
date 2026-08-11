param(
    [string]$TrainPath = "data\processed\슈퍼_train_qa_complete.csv",
    [string]$ValidationPath = "data\processed\슈퍼_validation_qa_complete.csv",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# 기존 산출물을 실수로 덮어쓰지 않도록 먼저 모두 확인합니다.
$targets = @(
    "docs\super_data_quality.md", "results\super_intent_distribution.csv",
    "results\super_duplicate_questions.csv", "results\super_short_questions.csv",
    "data\experiment\super_train_sample_5000.csv", "data\experiment\super_train_short_question_candidates.csv",
    "results\super_train_sample_distribution.csv", "data\experiment\super_validation_sample_500.csv",
    "results\super_validation_sample_distribution.csv", "docs\experiment_dataset_design.md"
)
foreach ($target in $targets) {
    if ((Test-Path -LiteralPath $target) -and -not $Force) {
        throw "기존 파일이 있어 중단합니다: $target (확인 후 의도적으로 교체하려면 -Force를 사용하세요.)"
    }
}

foreach ($directory in @("docs", "results", "data\experiment")) {
    if (-not (Test-Path -LiteralPath $directory)) { [void](New-Item -ItemType Directory -Path $directory) }
}

# 정적 타입 C# 처리기를 컴파일하여 대용량 CSV도 빠르게 분석합니다.
Add-Type -Path (Join-Path $PSScriptRoot "SuperExperimentProcessor.cs")
$resolvedTrainPath = [System.IO.Path]::GetFullPath($TrainPath)
$resolvedValidationPath = [System.IO.Path]::GetFullPath($ValidationPath)
if (-not (Test-Path -LiteralPath $resolvedTrainPath)) { throw "Training 입력 파일이 없습니다: $resolvedTrainPath" }
if (-not (Test-Path -LiteralPath $resolvedValidationPath)) { throw "Validation 입력 파일이 없습니다: $resolvedValidationPath" }
$workspaceRoot = (Get-Location).Path
if ([string]::IsNullOrWhiteSpace($workspaceRoot)) { throw "작업 폴더의 절대경로를 확인할 수 없습니다." }
try {
    [SuperExperimentProcessor]::Run($resolvedTrainPath, $resolvedValidationPath, $workspaceRoot)
}
catch {
    # 리플렉션 호출 내부의 실제 C# 오류 위치까지 표시하여 문제를 쉽게 찾을 수 있게 합니다.
    if ($_.Exception.InnerException) {
        Write-Host $_.Exception.InnerException.StackTrace
        throw $_.Exception.InnerException
    }
    throw
}
