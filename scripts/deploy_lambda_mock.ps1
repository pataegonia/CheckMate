[CmdletBinding()]
param(
    [string]$FunctionName = "checkmate-ai-grading-mock",
    [string]$RoleArn = "",
    [string]$Region = $(if ($env:AWS_REGION) { $env:AWS_REGION } elseif ($env:AWS_DEFAULT_REGION) { $env:AWS_DEFAULT_REGION } else { "us-east-1" }),
    [string]$Profile = $env:AWS_PROFILE,
    [string]$ZipPath = "build/checkmate-ai-grading-mock.zip",
    [string]$Runtime = "python3.12",
    [string]$Handler = "ai_grading.lambda_handler.handler",
    [int]$Timeout = 60,
    [int]$MemorySize = 256,
    [string]$Architecture = "x86_64",
    [string]$AppInternalToken = $(if ($env:APP_INTERNAL_TOKEN) { $env:APP_INTERNAL_TOKEN } else { "change-me-in-prod" }),
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
    & (Join-Path $PSScriptRoot "package_lambda_mock.ps1") `
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
$EnvironmentJson = @{
    Variables = @{
        APP_INTERNAL_TOKEN = $AppInternalToken
        AI_GRADING_MODE = "mock"
    }
} | ConvertTo-Json -Compress
$EnvironmentFile = Join-Path $env:TEMP "checkmate-lambda-environment.json"
$Utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($EnvironmentFile, $EnvironmentJson, $Utf8NoBom)
$EnvironmentFileUri = "file://$($EnvironmentFile -replace "\\", "/")"

$GetFunctionErrorFile = Join-Path $env:TEMP "checkmate-get-function-error.txt"

if (Test-Path $GetFunctionErrorFile) {
    Remove-Item $GetFunctionErrorFile -Force
}

$PreviousErrorActionPreference = $ErrorActionPreference
try {
    # First deploys normally hit ResourceNotFoundException here. Windows
    # PowerShell can turn native stderr into a terminating error when
    # ErrorActionPreference is Stop, so capture this probe gently.
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
        throw "Function does not exist. Pass -RoleArn with an IAM role that has AWSLambdaBasicExecutionRole."
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
Write-Host "Mock Lambda is ready."
Write-Host "Function name: $FunctionName"
Write-Host "Handler: $Handler"
Write-Host "Environment: AI_GRADING_MODE=mock, APP_INTERNAL_TOKEN=$AppInternalToken"
Write-Host ""
Write-Host "Set BE cloud.aws.lambda.grading-function-name to: $FunctionName"
