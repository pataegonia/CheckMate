[CmdletBinding()]
param(
    [string]$OutputPath = "build/checkmate-ai-grading-mock.zip",
    [string]$RequirementsPath = "requirements-lambda-mock.txt",
    [string]$PythonExecutable = "python",
    [string]$PythonVersion = "3.12",
    [string]$PythonAbi = "cp312",
    [string]$Platform = "manylinux2014_x86_64",
    [switch]$SkipDependencyInstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BuildRoot = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot "build\lambda-mock"))
$PackageDir = Join-Path $BuildRoot "package"
$RepoRootWithSep = $RepoRoot.TrimEnd("\", "/") + [System.IO.Path]::DirectorySeparatorChar

if (-not (($BuildRoot + [System.IO.Path]::DirectorySeparatorChar).StartsWith($RepoRootWithSep, [System.StringComparison]::OrdinalIgnoreCase))) {
    throw "Refusing to remove build path outside repository: $BuildRoot"
}

if ([System.IO.Path]::IsPathRooted($OutputPath)) {
    $ResolvedOutputPath = [System.IO.Path]::GetFullPath($OutputPath)
} else {
    $ResolvedOutputPath = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $OutputPath))
}

if ([System.IO.Path]::IsPathRooted($RequirementsPath)) {
    $ResolvedRequirementsPath = [System.IO.Path]::GetFullPath($RequirementsPath)
} else {
    $ResolvedRequirementsPath = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $RequirementsPath))
}

if (Test-Path -LiteralPath $BuildRoot) {
    Remove-Item -LiteralPath $BuildRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $PackageDir -Force | Out-Null

if (-not $SkipDependencyInstall) {
    if (-not (Test-Path -LiteralPath $ResolvedRequirementsPath)) {
        throw "Requirements file not found: $ResolvedRequirementsPath"
    }

    & $PythonExecutable -m pip install --upgrade `
        --requirement $ResolvedRequirementsPath `
        --target $PackageDir `
        --platform $Platform `
        --implementation cp `
        --python-version $PythonVersion `
        --abi $PythonAbi `
        --only-binary=:all: `
        --no-cache-dir
    if ($LASTEXITCODE -ne 0) {
        throw "pip install failed"
    }
}

Copy-Item -LiteralPath (Join-Path $RepoRoot "ai_grading") -Destination $PackageDir -Recurse

Get-ChildItem -LiteralPath $PackageDir -Recurse -Force -Directory -Filter "__pycache__" |
    Remove-Item -Recurse -Force
Get-ChildItem -LiteralPath $PackageDir -Recurse -Force -File |
    Where-Object { $_.Extension -in @(".pyc", ".pyo") } |
    Remove-Item -Force

$OutputParent = Split-Path -Parent $ResolvedOutputPath
if ($OutputParent) {
    New-Item -ItemType Directory -Path $OutputParent -Force | Out-Null
}
if (Test-Path -LiteralPath $ResolvedOutputPath) {
    Remove-Item -LiteralPath $ResolvedOutputPath -Force
}

$PackageItems = Get-ChildItem -LiteralPath $PackageDir -Force
if (-not $PackageItems) {
    throw "Package directory is empty: $PackageDir"
}

Compress-Archive -Path $PackageItems.FullName -DestinationPath $ResolvedOutputPath -Force
Write-Host "Created Lambda zip: $ResolvedOutputPath"
