# AI Hub 고객상담 데이터의 발화를 QA 단위로 묶는 전처리 스크립트입니다.
# 초보자도 안전하게 실행할 수 있도록 기존 결과물은 기본적으로 덮어쓰지 않습니다.
param(
    [string]$InputPath = "Training\라벨링데이터_train\슈퍼_train.csv",
    [string]$OutputPath = "data\processed\슈퍼_train_qa.csv",
    [string]$CompleteOutputPath = "data\processed\슈퍼_train_qa_complete.csv",
    [string]$ReportPath = "docs\data_analysis.md",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# 기존 결과가 있으면 명시적인 -Force 없이는 절대 덮어쓰지 않습니다.
foreach ($target in @($OutputPath, $CompleteOutputPath, $ReportPath)) {
    if ((Test-Path -LiteralPath $target) -and -not $Force) {
        throw "기존 파일이 있어 중단합니다: $target (내용을 확인한 뒤 -Force를 사용하세요.)"
    }
}
foreach ($dir in @((Split-Path $OutputPath), (Split-Path $CompleteOutputPath), (Split-Path $ReportPath))) {
    if ($dir -and -not (Test-Path -LiteralPath $dir)) { [void](New-Item -ItemType Directory -Path $dir) }
}

# 대용량 데이터는 컴파일된 처리기를 사용해 빠르게 그룹화합니다.
$processorSource = Join-Path $PSScriptRoot "SuperQaProcessor.cs"
Add-Type -Path $processorSource
[SuperQaProcessor]::Run(
    (Resolve-Path -LiteralPath $InputPath).Path,
    (Join-Path (Get-Location) $OutputPath),
    (Join-Path (Get-Location) $CompleteOutputPath),
    (Join-Path (Get-Location) $ReportPath)
)
return

function Get-SafeEncoding {
    param([string]$Path)

    # BOM이 있으면 그 정보를 우선 사용합니다.
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
        return [PSCustomObject]@{ Name = "UTF-8 with BOM"; Encoding = [System.Text.UTF8Encoding]::new($true, $true) }
    }
    if ($bytes.Length -ge 2 -and $bytes[0] -eq 0xFF -and $bytes[1] -eq 0xFE) {
        return [PSCustomObject]@{ Name = "UTF-16 LE"; Encoding = [System.Text.UnicodeEncoding]::new($false, $true, $true) }
    }
    if ($bytes.Length -ge 2 -and $bytes[0] -eq 0xFE -and $bytes[1] -eq 0xFF) {
        return [PSCustomObject]@{ Name = "UTF-16 BE"; Encoding = [System.Text.UnicodeEncoding]::new($true, $true, $true) }
    }

    # BOM이 없으면 먼저 UTF-8을 엄격하게 검증합니다. 깨진 바이트가 있으면 예외가 발생합니다.
    try {
        $strictUtf8 = [System.Text.UTF8Encoding]::new($false, $true)
        [void]$strictUtf8.GetString($bytes)
        return [PSCustomObject]@{ Name = "UTF-8 (no BOM)"; Encoding = $strictUtf8 }
    }
    catch [System.Text.DecoderFallbackException] {
        # AI Hub CSV에서 자주 사용되는 CP949를 대안으로 사용합니다.
        return [PSCustomObject]@{ Name = "CP949"; Encoding = [System.Text.Encoding]::GetEncoding(949) }
    }
}

function ConvertTo-CsvField {
    param([AllowEmptyString()][string]$Value)
    if ($null -eq $Value) { $Value = "" }
    # 쉼표, 따옴표, 줄바꿈이 있어도 CSV가 깨지지 않도록 항상 따옴표로 감쌉니다.
    return '"' + $Value.Replace('"', '""') + '"'
}

function Get-DocumentId {
    param([string]$Category, [string]$ConversationId, [string]$QaNumber)
    # 세 원본 키를 URI 안전 문자열로 바꿔 직접 조합합니다. 같은 세 값에는 언제나 같은 ID가 생깁니다.
    $safeCategory = [System.Uri]::EscapeDataString($Category)
    $safeConversation = [System.Uri]::EscapeDataString($ConversationId)
    $safeQaNumber = [System.Uri]::EscapeDataString($QaNumber)
    return "${safeCategory}__${safeConversation}__${safeQaNumber}"
}

$resolvedInput = (Resolve-Path -LiteralPath $InputPath).Path
foreach ($target in @($OutputPath, $CompleteOutputPath, $ReportPath)) {
    if ((Test-Path -LiteralPath $target) -and -not $Force) {
        throw "기존 파일이 있어 중단합니다: $target (의도한 덮어쓰기라면 -Force를 사용하세요.)"
    }
}
foreach ($dir in @((Split-Path $OutputPath), (Split-Path $CompleteOutputPath), (Split-Path $ReportPath))) {
    if ($dir -and -not (Test-Path -LiteralPath $dir)) { [void](New-Item -ItemType Directory -Path $dir) }
}

$encodingInfo = Get-SafeEncoding -Path $resolvedInput
Write-Host "감지된 인코딩: $($encodingInfo.Name)"

# TextFieldParser는 발화문 안에 쉼표나 줄바꿈이 있어도 CSV를 안전하게 읽습니다.
Add-Type -AssemblyName Microsoft.VisualBasic
$parser = [Microsoft.VisualBasic.FileIO.TextFieldParser]::new($resolvedInput, $encodingInfo.Encoding, $true)
$parser.TextFieldType = [Microsoft.VisualBasic.FileIO.FieldType]::Delimited
$parser.SetDelimiters(",")
$parser.HasFieldsEnclosedInQuotes = $true

$headers = $parser.ReadFields()
$required = @("발화자", "발화문", "카테고리", "QA번호", "QA여부", "인텐트", "상담번호", "상담내순번")
foreach ($name in $required) {
    if ($headers -notcontains $name) { throw "필수 컬럼이 없습니다: $name" }
}
$index = @{}
for ($i = 0; $i -lt $headers.Count; $i++) { $index[$headers[$i]] = $i }

Write-Host "`n실제 컬럼명 ($($headers.Count)개):"
Write-Host ($headers -join ", ")
Write-Host "`n앞부분 10개 행:"

$groups = @{}
$conversationIds = [System.Collections.Generic.HashSet[string]]::new()
$utteranceCount = 0L
$sampleRows = [System.Collections.Generic.List[object]]::new()

try {
    while (-not $parser.EndOfData) {
        $fields = $parser.ReadFields()
        $utteranceCount++
        if ($fields.Count -ne $headers.Count) {
            throw "$utteranceCount 번째 데이터 행의 컬럼 수($($fields.Count))가 헤더($($headers.Count))와 다릅니다."
        }

        $row = @{}
        foreach ($name in $required) { $row[$name] = $fields[$index[$name]] }
        if ($sampleRows.Count -lt 10) { $sampleRows.Add([PSCustomObject]$row) }

        $category = $row["카테고리"].Trim()
        $conversation = $row["상담번호"].Trim()
        $qaNumber = $row["QA번호"].Trim()
        [void]$conversationIds.Add($conversation)
        $separator = [char]31
        $key = "$category$separator$conversation$separator$qaNumber"
        if (-not $groups.ContainsKey($key)) {
            $groups[$key] = [PSCustomObject]@{
                Category = $category; Conversation = $conversation; QaNumber = $qaNumber
                Questions = [System.Collections.Generic.List[object]]::new()
                Answers = [System.Collections.Generic.List[object]]::new()
                Intents = [System.Collections.Generic.List[object]]::new()
            }
        }

        $order = 0L
        [void][long]::TryParse($row["상담내순번"], [ref]$order)
        $item = [PSCustomObject]@{ Order = $order; Text = $row["발화문"].Trim() }
        $qaType = $row["QA여부"].Trim().ToLowerInvariant()
        if ($qaType -eq "q") { $groups[$key].Questions.Add($item) }
        elseif ($qaType -eq "a") { $groups[$key].Answers.Add($item) }
        if (-not [string]::IsNullOrWhiteSpace($row["인텐트"])) {
            $groups[$key].Intents.Add([PSCustomObject]@{ Order = $order; Text = $row["인텐트"].Trim() })
        }
    }
}
finally { $parser.Close() }

$sampleRows | Format-Table $required -AutoSize -Wrap | Out-Host

$csvHeader = @("document_id", "category", "conversation_id", "qa_number", "intent", "question", "answer", "question_length", "answer_length")
$utf8Bom = [System.Text.UTF8Encoding]::new($true)
$allWriter = [System.IO.StreamWriter]::new((Join-Path (Get-Location) $OutputPath), $false, $utf8Bom)
$completeWriter = [System.IO.StreamWriter]::new((Join-Path (Get-Location) $CompleteOutputPath), $false, $utf8Bom)
$headerLine = ($csvHeader | ForEach-Object { ConvertTo-CsvField $_ }) -join ","
$allWriter.WriteLine($headerLine)
$completeWriter.WriteLine($headerLine)

$bothCount = 0L; $questionOnlyCount = 0L; $answerOnlyCount = 0L; $bothEmptyCount = 0L; $shortQuestionCount = 0L
$questionLengthSum = 0L; $answerLengthSum = 0L
$intentCounts = @{}

try {
    foreach ($key in $groups.Keys) {
        $group = $groups[$key]
        # 발화가 하나뿐인 그룹은 정렬 작업을 생략하고, 둘 이상일 때만 상담내순번으로 정렬합니다.
        if ($group.Questions.Count -eq 0) { $question = "" }
        elseif ($group.Questions.Count -eq 1) { $question = $group.Questions[0].Text }
        else { $question = (($group.Questions | Sort-Object Order | ForEach-Object Text) -join " ").Trim() }
        if ($group.Answers.Count -eq 0) { $answer = "" }
        elseif ($group.Answers.Count -eq 1) { $answer = $group.Answers[0].Text }
        else { $answer = (($group.Answers | Sort-Object Order | ForEach-Object Text) -join " ").Trim() }
        # 한 QA 그룹에 인텐트가 여러 개면 대화 순서상 가장 먼저 나오는 비어 있지 않은 값을 사용합니다.
        $intent = ""
        $firstIntentOrder = [long]::MaxValue
        foreach ($intentItem in $group.Intents) {
            if ($intentItem.Order -lt $firstIntentOrder) {
                $firstIntentOrder = $intentItem.Order
                $intent = $intentItem.Text
            }
        }
        $qLength = $question.Length; $aLength = $answer.Length
        $questionLengthSum += $qLength; $answerLengthSum += $aLength
        if ($qLength -le 4) { $shortQuestionCount++ }
        $intentKey = if ([string]::IsNullOrWhiteSpace($intent)) { "(빈 인텐트)" } else { $intent }
        if (-not $intentCounts.ContainsKey($intentKey)) { $intentCounts[$intentKey] = 0L }
        $intentCounts[$intentKey]++

        $hasQuestion = $qLength -gt 0; $hasAnswer = $aLength -gt 0
        if ($hasQuestion -and $hasAnswer) { $bothCount++ }
        elseif ($hasQuestion) { $questionOnlyCount++ }
        elseif ($hasAnswer) { $answerOnlyCount++ }
        else { $bothEmptyCount++ }

        $record = @(
            (Get-DocumentId $group.Category $group.Conversation $group.QaNumber), $group.Category,
            $group.Conversation, $group.QaNumber, $intent, $question, $answer, [string]$qLength, [string]$aLength
        )
        $escapedRecord = [string[]]::new($record.Count)
        for ($fieldIndex = 0; $fieldIndex -lt $record.Count; $fieldIndex++) {
            $escapedRecord[$fieldIndex] = ConvertTo-CsvField ([string]$record[$fieldIndex])
        }
        $line = $escapedRecord -join ","
        $allWriter.WriteLine($line)
        if ($hasQuestion -and $hasAnswer) { $completeWriter.WriteLine($line) }
    }
}
finally { $allWriter.Dispose(); $completeWriter.Dispose() }

$groupCount = [long]$groups.Count
$emptySideCount = $questionOnlyCount + $answerOnlyCount + $bothEmptyCount
$avgQuestion = if ($groupCount) { [math]::Round($questionLengthSum / $groupCount, 2) } else { 0 }
$avgAnswer = if ($groupCount) { [math]::Round($answerLengthSum / $groupCount, 2) } else { 0 }
$intentLines = $intentCounts.GetEnumerator() |
    Sort-Object -Property @{ Expression = "Value"; Descending = $true }, @{ Expression = "Name"; Descending = $false } |
    ForEach-Object { "- $($_.Name): $($_.Value)" }

$report = @"
# 슈퍼 Training 데이터 QA 전처리 분석

- 원본 파일: ``$InputPath``
- 감지된 인코딩: **$($encodingInfo.Name)**
- 그룹 기준: ``카테고리 + 상담번호 + QA번호``
- 그룹 내 정렬: ``상담내순번`` 오름차순
- 질문/답변 합치기: 각각 ``QA여부=q`` / ``QA여부=a`` 발화를 공백 하나로 연결
- 인텐트 선택: 그룹을 순서대로 보았을 때 첫 비공백 인텐트
- 평균 길이 모수: 빈 질문/답변을 포함한 전체 QA 그룹

## 전체 통계

| 항목 | 개수 |
|---|---:|
| 전체 발화 수 | $utteranceCount |
| 전체 상담 수 | $($conversationIds.Count) |
| 전체 QA 그룹 수 | $groupCount |
| 질문과 답변이 모두 존재하는 그룹 수 | $bothCount |
| 질문만 있는 그룹 수 | $questionOnlyCount |
| 답변만 있는 그룹 수 | $answerOnlyCount |
| 질문과 답변이 모두 빈 그룹 수 | $bothEmptyCount |
| 질문이나 답변이 빈 그룹 수 | $emptySideCount |
| 평균 질문 길이 | $avgQuestion자 |
| 평균 답변 길이 | $avgAnswer자 |
| 질문 길이가 4자 이하인 그룹 수 | $shortQuestionCount |

## 인텐트별 QA 개수

$($intentLines -join "`n")

## 생성 파일

- 전체 QA 그룹: ``$OutputPath``
- 질문과 답변이 모두 있는 QA: ``$CompleteOutputPath``
"@
[System.IO.File]::WriteAllText((Join-Path (Get-Location) $ReportPath), $report, [System.Text.UTF8Encoding]::new($true))

Write-Host "`n=== 전처리 통계 ==="
Write-Host "전체 발화 수: $utteranceCount"
Write-Host "전체 상담 수: $($conversationIds.Count)"
Write-Host "전체 QA 그룹 수: $groupCount"
Write-Host "질문+답변 그룹 수: $bothCount"
Write-Host "질문만 있는 그룹 수: $questionOnlyCount"
Write-Host "답변만 있는 그룹 수: $answerOnlyCount"
Write-Host "질문과 답변이 모두 빈 그룹 수: $bothEmptyCount"
Write-Host "질문이나 답변이 빈 그룹 수: $emptySideCount"
Write-Host "평균 질문 길이: $avgQuestion자"
Write-Host "평균 답변 길이: $avgAnswer자"
Write-Host "질문 길이 4자 이하: $shortQuestionCount"
Write-Host "`n인텐트별 QA 개수:"
$intentLines | ForEach-Object { Write-Host $_ }
Write-Host "`n저장 완료: $OutputPath"
Write-Host "완전한 QA 저장 완료: $CompleteOutputPath"
Write-Host "분석 문서 저장 완료: $ReportPath"
