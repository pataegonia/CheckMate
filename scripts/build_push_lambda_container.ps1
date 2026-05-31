[CmdletBinding()]
param(
    [string]$RepositoryName = "checkmate-ai-grading",
    [string]$ImageTag = "latest",
    [string]$AccountId = $(if ($env:AWS_ACCOUNT_ID) { $env:AWS_ACCOUNT_ID } else { "259808926999" }),
    [string]$Region = $(if ($env:AWS_REGION) { $env:AWS_REGION } elseif ($env:AWS_DEFAULT_REGION) { $env:AWS_DEFAULT_REGION } else { "us-east-1" }),
    [string]$Profile = $env:AWS_PROFILE,
    [string]$Platform = "linux/amd64",
    [switch]$SkipBuild,
    [switch]$SkipPush
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

$GlobalAwsArgs = @()
if ($Region) {
    $GlobalAwsArgs += @("--region", $Region)
}
if ($Profile) {
    $GlobalAwsArgs += @("--profile", $Profile)
}

$RegistryHost = "$AccountId.dkr.ecr.$Region.amazonaws.com"
$ImageUri = "$RegistryHost/${RepositoryName}:$ImageTag"

# 1) Ensure ECR repository exists.
$DescribeErrorFile = Join-Path $env:TEMP "checkmate-ecr-describe-error.txt"
if (Test-Path $DescribeErrorFile) {
    Remove-Item $DescribeErrorFile -Force
}

$PreviousErrorActionPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = "Continue"
    & aws @GlobalAwsArgs ecr describe-repositories --repository-names $RepositoryName 1>$null 2>$DescribeErrorFile
    $DescribeExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $PreviousErrorActionPreference
}

if ($DescribeExitCode -ne 0) {
    $DescribeErrorText = ""
    if (Test-Path $DescribeErrorFile) {
        $DescribeErrorText = Get-Content $DescribeErrorFile -Raw
    }
    if ($DescribeErrorText -match "RepositoryNotFoundException") {
        Write-Host "Creating ECR repository: $RepositoryName"
        Invoke-AwsCli @("ecr", "create-repository", "--repository-name", $RepositoryName, "--image-scanning-configuration", "scanOnPush=true")
    } else {
        Write-Error $DescribeErrorText
        throw "Failed to describe ECR repository: $RepositoryName"
    }
}

# 2) Docker login to ECR.
Write-Host "Logging Docker in to $RegistryHost"
$LoginPassword = & aws @GlobalAwsArgs ecr get-login-password
if ($LASTEXITCODE -ne 0) {
    throw "aws ecr get-login-password failed"
}
$LoginPassword | docker login --username AWS --password-stdin $RegistryHost
if ($LASTEXITCODE -ne 0) {
    throw "docker login failed"
}

# 3) Build (linux/amd64 for Lambda x86_64 runtime).
if (-not $SkipBuild) {
    Write-Host "Building image: $ImageUri (platform=$Platform)"
    docker buildx build `
        --platform $Platform `
        --provenance=false `
        --tag $ImageUri `
        --load `
        $RepoRoot
    if ($LASTEXITCODE -ne 0) {
        throw "docker buildx build failed"
    }
}

# 4) Push.
if (-not $SkipPush) {
    Write-Host "Pushing image: $ImageUri"
    docker push $ImageUri
    if ($LASTEXITCODE -ne 0) {
        throw "docker push failed"
    }
}

Write-Host ""
Write-Host "Pushed image URI:"
Write-Host "  $ImageUri"
Write-Host ""
Write-Host "Pass this URI to scripts/deploy_lambda_container.ps1 with -ImageUri,"
Write-Host "or rely on its default which uses the same naming convention."
