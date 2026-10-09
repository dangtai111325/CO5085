param([switch]$Preview)

$ErrorActionPreference = 'Stop'
function Invoke-Git {
    & git @args
    if ($LASTEXITCODE -ne 0) { throw "Git failed: $($args -join ' ')" }
}

$project = (Invoke-Git -C $PSScriptRoot rev-parse --show-toplevel).Trim()
if ((Invoke-Git -C $project branch --show-current).Trim() -ne 'main') {
    throw 'Publish from the main branch.'
}
if (Invoke-Git -C $project status --porcelain --untracked-files=no) {
    throw 'Commit project changes before publishing.'
}
$revision = (Invoke-Git -C $project rev-parse HEAD).Trim()
$remote = (Invoke-Git -C $project remote get-url origin).Trim()
$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$work = Join-Path $tempRoot ('co5085-pages-' + [guid]::NewGuid().ToString('N'))
$site = Join-Path $work 'site'
New-Item -ItemType Directory -Path $work | Out-Null

try {
    # Export committed files only: raw datasets and local caches are excluded.
    $archive = Join-Path $work 'source.zip'
    Invoke-Git -C $project -c core.autocrlf=false archive --format=zip "--output=$archive" HEAD -- `
        public assignment source_code report/CO5085_Report.pdf datasets/README.md README.md
    Expand-Archive -LiteralPath $archive -DestinationPath $site
    Remove-Item -LiteralPath (Join-Path $site 'public/publish.ps1')
    Copy-Item -LiteralPath (Join-Path $site 'public/.nojekyll') -Destination $site
    $html = [IO.File]::ReadAllText((Join-Path $site 'public/index.html'))
    [IO.File]::WriteAllText((Join-Path $site 'index.html'),
        $html.Replace('href="../', 'href="./'), [Text.UTF8Encoding]::new($false))

    if ($Preview) {
        Write-Output "Preview: $site"
        return
    }

    # Build a separate deployment branch without checking out or changing main.
    Invoke-Git -C $site init -b gh-pages
    Invoke-Git -C $site config user.name (Invoke-Git -C $project config user.name)
    Invoke-Git -C $site config user.email (Invoke-Git -C $project config user.email)
    Invoke-Git -C $site config core.autocrlf false
    Invoke-Git -C $site remote add origin $remote
    $hasRemote = [bool](Invoke-Git -C $site ls-remote --heads origin gh-pages)
    if ($hasRemote) {
        Invoke-Git -C $site fetch --depth=1 origin gh-pages
        Invoke-Git -C $site update-ref refs/heads/gh-pages FETCH_HEAD
    }
    Invoke-Git -C $site add --all
    $hasChanges = $true
    if ($hasRemote) {
        & git -C $site diff --cached --quiet HEAD
        if ($LASTEXITCODE -gt 1) { throw 'Could not compare the deployment snapshot.' }
        $hasChanges = $LASTEXITCODE -eq 1
    }
    if ($hasChanges) {
        Invoke-Git -C $site commit -m "Publish coursework site from $revision"
    }
    Invoke-Git -C $site push origin gh-pages
    Write-Output 'Published: https://dangtai111325.github.io/CO5085/'
} finally {
    if (-not $Preview) {
        $resolved = (Resolve-Path -LiteralPath $work).Path
        if ($resolved -ne $work -or -not $resolved.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Unexpected temporary directory; cleanup stopped.'
        }
        Remove-Item -LiteralPath $resolved -Recurse -Force
    }
}
