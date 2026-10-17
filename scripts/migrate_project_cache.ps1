# Only move verified wheel caches of packages installed for this project.
# Uses native PowerShell operations throughout; never moves shared cache roots.
$ErrorActionPreference = 'Stop'
$taskProject = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$taskPipSource = [IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA 'pip\Cache'))
$taskDestination = [IO.Path]::GetFullPath((Join-Path $taskProject '.cache\pip'))
if (-not $taskDestination.StartsWith($taskProject + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Destination must stay inside the project'
}
$taskStarted = (Get-Item -LiteralPath (Join-Path $taskProject '.venv')).CreationTime
$taskPackages = @('transformers', 'huggingface_hub', 'huggingface-hub', 'tokenizers', 'ninja')
$taskAudit = [Collections.Generic.List[object]]::new()
Add-Type -AssemblyName System.IO.Compression.FileSystem
if (Test-Path -LiteralPath $taskPipSource) {
    $taskBodies = Get-ChildItem -LiteralPath $taskPipSource -Recurse -File -Filter '*.body' |
        Where-Object { $_.LastWriteTime -ge $taskStarted }
    foreach ($taskBody in $taskBodies) {
        $taskPackage = $null
        try {
            $taskArchive = [IO.Compression.ZipFile]::OpenRead($taskBody.FullName)
            try {
                $taskMetadata = $taskArchive.Entries | Where-Object { $_.FullName -match '\.dist-info/METADATA$' } | Select-Object -First 1
                if ($taskMetadata) {
                    $taskReader = [IO.StreamReader]::new($taskMetadata.Open())
                    try { $taskContent = $taskReader.ReadToEnd() } finally { $taskReader.Dispose() }
                    $taskMatch = [regex]::Match($taskContent, '(?m)^Name: ([^\r\n]+)')
                    if ($taskMatch.Success -and $taskPackages -contains $taskMatch.Groups[1].Value) {
                        $taskPackage = $taskMatch.Groups[1].Value
                    }
                }
            } finally { $taskArchive.Dispose() }
        } catch [IO.InvalidDataException] { continue }
        if (-not $taskPackage) { continue }
        $taskPair = @($taskBody.FullName, $taskBody.FullName.Substring(0, $taskBody.FullName.Length - 5))
        foreach ($taskFile in $taskPair) {
            if (-not (Test-Path -LiteralPath $taskFile -PathType Leaf)) { continue }
            $taskResolved = [IO.Path]::GetFullPath($taskFile)
            if (-not $taskResolved.StartsWith($taskPipSource + '\', [StringComparison]::OrdinalIgnoreCase)) {
                throw 'Source escaped pip cache'
            }
            $taskRelative = $taskResolved.Substring($taskPipSource.Length + 1)
            $taskTarget = [IO.Path]::GetFullPath((Join-Path $taskDestination $taskRelative))
            if (-not $taskTarget.StartsWith($taskDestination + '\', [StringComparison]::OrdinalIgnoreCase)) {
                throw 'Target escaped project pip cache'
            }
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $taskTarget) | Out-Null
            if (Test-Path -LiteralPath $taskTarget) { throw "Target already exists: $taskTarget" }
            $taskHash = (Get-FileHash -LiteralPath $taskResolved -Algorithm SHA256).Hash
            $taskBytes = (Get-Item -LiteralPath $taskResolved).Length
            Move-Item -LiteralPath $taskResolved -Destination $taskTarget
            if ((Get-FileHash -LiteralPath $taskTarget -Algorithm SHA256).Hash -ne $taskHash) { throw 'Hash mismatch' }
            $taskAudit.Add([pscustomobject]@{package=$taskPackage; source=$taskResolved; destination=$taskTarget; bytes=$taskBytes; sha256=$taskHash})
        }
    }
}
$taskAuditPath = Join-Path $taskProject 'reports\cache_migration.json'
ConvertTo-Json -InputObject @($taskAudit.ToArray()) -Depth 4 | Set-Content -LiteralPath $taskAuditPath -Encoding utf8
$taskAudit | Select-Object package,bytes,destination
Write-Output "Migration audit: $taskAuditPath"
foreach ($taskShared in @((Join-Path $env:APPDATA 'NVIDIA\ComputeCache'), (Join-Path $env:LOCALAPPDATA 'NVIDIA\ComputeCache'))) {
    if (Test-Path -LiteralPath $taskShared) {
        $taskRecent = @(Get-ChildItem -LiteralPath $taskShared -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $_.LastWriteTime -ge $taskStarted })
        Write-Output "Shared GPU cache (not attributable, retained): $taskShared; recent files=$($taskRecent.Count)"
    }
}
