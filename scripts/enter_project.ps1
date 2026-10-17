# Dot source once in PowerShell: . .\scripts\enter_project.ps1
$taskProject = Split-Path -Parent $PSScriptRoot
$taskCacheMap = @{
    TEMP = 'build\tmp'; TMP = 'build\tmp'; TMPDIR = 'build\tmp'
    CUDA_CACHE_PATH = '.cache\cuda'; TORCH_EXTENSIONS_DIR = 'build\torch_extensions'
    PIP_CACHE_DIR = '.cache\pip'; HF_HOME = '.cache\huggingface'
    HF_HUB_CACHE = '.cache\huggingface\hub'; HF_XET_CACHE = '.cache\huggingface\xet'
    TORCH_HOME = '.cache\torch'; TORCHINDUCTOR_CACHE_DIR = '.cache\inductor'
    XDG_CACHE_HOME = '.cache'
}
foreach ($taskEntry in $taskCacheMap.GetEnumerator()) {
    $taskPath = Join-Path $taskProject $taskEntry.Value
    New-Item -ItemType Directory -Force -Path $taskPath | Out-Null
    Set-Item -LiteralPath "Env:$($taskEntry.Key)" -Value $taskPath
}
$env:USE_TF = '0'
$env:VSLANG = '1033'
$env:PATH = (Join-Path $taskProject '.venv\Scripts') + ';' + $env:PATH
Write-Output "Project caches: $taskProject\.cache"
