param(
    [switch]$WithModel,
    [switch]$Development,
    [string]$ModelDirectory
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvDirectory = Join-Path $projectRoot '.venv'
$pythonExe = Join-Path $venvDirectory 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    python -m venv $venvDirectory
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
Push-Location $projectRoot
try {
    $coreInstall = if ($Development) { '.[dev]' } else { '.' }
    & $pythonExe -m pip install -e $coreInstall
    if ($LASTEXITCODE -ne 0) { throw 'Core installation failed.' }
    $env:ARCH_REVIEW_HOME = Join-Path $projectRoot '.runtime'
    if ($WithModel) {
        & $pythonExe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
        if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed.' }
        & $pythonExe -m pip install -e '.[model]'
        if ($LASTEXITCODE -ne 0) { throw 'Laya installation failed.' }
        if (-not $ModelDirectory) { $ModelDirectory = Join-Path $projectRoot 'models\laya-multilingual' }
        & $pythonExe -m arch_review download-model --destination $ModelDirectory
        if ($LASTEXITCODE -ne 0) { throw 'Checkpoint download failed.' }
        & $pythonExe -m arch_review configure --checkpoint $ModelDirectory
        if ($LASTEXITCODE -ne 0) { throw 'Checkpoint configuration failed. Stop the local daemon first.' }
    }
    & $pythonExe -m arch_review mcp-config
    Write-Host 'Before connecting a Windows MCP assistant, start the shared service from this terminal:'
    Write-Host ('& "' + $pythonExe + '" -m arch_review start')
} finally {
    Pop-Location
}
