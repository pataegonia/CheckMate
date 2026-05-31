[CmdletBinding()]
param(
    [string]$FunctionName = "checkmate-ai-grading",
    [string]$RoleArn = "",
    [string]$Region = $(if ($env:AWS_REGION) { $env:AWS_REGION } elseif ($env:AWS_DEFAULT_REGION) { $env:AWS_DEFAULT_REGION } else { "us-east-1" }),
    [string]$Profile = $env:AWS_PROFILE,
    [string]$ZipPath = "build/checkmate-ai-grading-bedrock.zip",
    [string]$Runtime = "python3.12",
    [string]$Handler = "ai_grading.lambda_handler.handler",
    [int]$Timeout = 300,
    [int]$MemorySize = 1024,
    [string]$Architecture = "x86_64",
    [string]$AppInternalToken = $(if ($env:APP_INTERNAL_TOKEN) { $env:APP_INTERNAL_TOKEN } else { "change-me-in-prod" }),
    [string]$BedrockModelId = $(if ($env:BEDROCK_MODEL_ID) { $env:BEDROCK_MODEL_ID } else { "us.anthropic.claude-sonnet-4-6" }),
    [string]$CropsBucket = $env:CHECKMATE_CROPS_BUCKET,
    [string]$CropsPrefix = $(if ($env:CHECKMATE_CROPS_PREFIX) { $env:CHECKMATE_CROPS_PREFIX } else { "graded-crops" }),
    [switch]$SkipPackage
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-AwsCli {
    param([string[]]$AwsCommand)

    & aws @GlobalAwsArgs @AwsCommand
    if ($LASTEXITCODE -ne 0) {
        $RenderedCommand = ($GlobalAwsArgs + $AwsCommand) -join " "
        throw "AWS CLI failed: aws $RenderedCommand"
    }
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if (-not $SkipPackage) {
    $PythonVersion = $Runtime -replace "^python", ""
    $PythonAbi = "cp" + ($PythonVersion -replace "\.", "")
    & (Join-Path $PSScriptRoot "package_lambda_bedrock.ps1") `
        -OutputPath $ZipPath `
        -PythonVersion $PythonVersion `
        -PythonAbi $PythonAbi
}

if ([System.IO.Path]::IsPathRooted($ZipPath)) {
    $ResolvedZipPath = [System.IO.Path]::GetFullPath($ZipPath)
} else {
    $ResolvedZipPath = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $ZipPath))
}

if (-not (Test-Path -LiteralPath $ResolvedZipPath)) {
    throw "Lambda zip not found: $ResolvedZipPath"
}

$GlobalAwsArgs = @()
if ($Region) {
    $GlobalAwsArgs += @("--region", $Region)
}
if ($Profile) {
    $GlobalAwsArgs += @("--profile", $Profile)
}

$ZipFileUri = "fileb://$($ResolvedZipPath -replace "\\", "/")"

$EnvVars = @{
    APP_INTERNAL_TOKEN = $AppInternalToken
    AI_GRADING_MODE    = "bedrock"
    BEDROCK_MODEL_ID   = $BedrockModelId
}
if ($CropsBucket) {
    $EnvVars["CHECKMATE_CROPS_BUCKET"] = $CropsBucket
}
if ($CropsPrefix) {
    $EnvVars["CHECKMATE_CROPS_PREFIX"] = $CropsPrefix
}
$EnvironmentJson = @{ Variables = $EnvVars } | ConvertTo-Json -Compress
$EnvironmentFile = Join-Path $env:TEMP "checkmate-lambda-bedrock-environment.json"
$Utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($EnvironmentFile, $EnvironmentJson, $Utf8NoBom)
$EnvironmentFileUri = "file://$($EnvironmentFile -replace "\\", "/")"

$GetFunctionErrorFile = Join-Path $env:TEMP "checkmate-bedrock-get-function-error.txt"

if (Test-Path $GetFunctionErrorFile) {
    Remove-Item $GetFunctionErrorFile -Force
}

$PreviousErrorActionPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = "Continue"
    $GetFunctionOutput = & aws @GlobalAwsArgs lambda get-function --function-name $FunctionName 2> $GetFunctionErrorFile
    $GetFunctionExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $PreviousErrorActionPreference
}
$FunctionExists = $GetFunctionExitCode -eq 0

$GetFunctionErrorText = ""
if (Test-Path $GetFunctionErrorFile) {
    $GetFunctionErrorText = Get-Content $GetFunctionErrorFile -Raw
}

if (-not $FunctionExists -and ($GetFunctionErrorText -notmatch "ResourceNotFoundException")) {
    Write-Error $GetFunctionErrorText
    throw "AWS CLI failed while checking Lambda function: $FunctionName"
}

if ($FunctionExists) {
    Write-Host "Updating Lambda function code: $FunctionName"
    Invoke-AwsCli @(
        "lambda", "update-function-code",
        "--function-name", $FunctionName,
        "--zip-file", $ZipFileUri
    )
    Invoke-AwsCli @("lambda", "wait", "function-updated", "--function-name", $FunctionName)

    Write-Host "Updating Lambda function configuration: $FunctionName"
    Invoke-AwsCli @(
        "lambda", "update-function-configuration",
        "--function-name", $FunctionName,
        "--runtime", $Runtime,
        "--handler", $Handler,
        "--timeout", "$Timeout",
        "--memory-size", "$MemorySize",
        "--environment", $EnvironmentFileUri
    )
    Invoke-AwsCli @("lambda", "wait", "function-updated", "--function-name", $FunctionName)
} else {
    if (-not $RoleArn) {
        throw "Function does not exist. Pass -RoleArn with an IAM role that has AWSLambdaBasicExecutionRole + S3 + bedrock:InvokeModel (see scripts/iam_lambda_bedrock_policy.json)."
    }

    Write-Host "Creating Lambda function: $FunctionName"
    Invoke-AwsCli @(
        "lambda", "create-function",
        "--function-name", $FunctionName,
        "--runtime", $Runtime,
        "--role", $RoleArn,
        "--handler", $Handler,
        "--zip-file", $ZipFileUri,
        "--timeout", "$Timeout",
        "--memory-size", "$MemorySize",
        "--architectures", $Architecture,
        "--environment", $EnvironmentFileUri
    )
    Invoke-AwsCli @("lambda", "wait", "function-active", "--function-name", $FunctionName)
}

Write-Host ""
Write-Host "Bedrock Lambda is ready."
Write-Host "Function name: $FunctionName"
Write-Host "Handler: $Handler"
Write-Host "Region: $Region"
Write-Host "Env: AI_GRADING_MODE=bedrock, BEDROCK_MODEL_ID=$BedrockModelId"
if ($CropsBucket) {
    Write-Host "Crops upload bucket: $CropsBucket (prefix=$CropsPrefix)"
} else {
    Write-Host "Crops upload bucket: (none — will reuse submission bucket)"
}
Write-Host ""
Write-Host "Reminders:"
Write-Host "  1) Execution role needs s3:GetObject (+ s3:PutObject if uploading crops) and bedrock:InvokeModel."
Write-Host "     Attach scripts/iam_lambda_bedrock_policy.json as an inline/managed policy."
Write-Host "  2) Bedrock console: enable model access for $BedrockModelId in $Region."
Write-Host "  3) BE payload must include submission.image.{bucket,key} and questions[].imageCrop."
