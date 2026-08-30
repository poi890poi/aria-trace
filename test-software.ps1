[CmdletBinding()]
param(
    [ValidateSet('Foundation', 'Full')]
    [string]$Scope = 'Foundation',

    [string]$Python
)

$ErrorActionPreference = 'Stop'
$repository = $PSScriptRoot

if (-not $Python) {
    $pinned = Join-Path $repository '.tools\standalone-release-py31210\Scripts\python.exe'
    if (Test-Path -LiteralPath $pinned) {
        $Python = $pinned
    }
    else {
        $command = Get-Command python -ErrorAction Stop
        $Python = $command.Source
    }
}

& $Python -c "import sys, cv2, numpy; print('Python', sys.version.split()[0]); print('OpenCV', cv2.__version__); print('NumPy', numpy.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw 'The selected Python environment lacks the required software dependencies'
}

if ($Scope -eq 'Full') {
    & $Python -m unittest discover -s (Join-Path $repository 'tests') -v
}
else {
    & $Python -m unittest `
        tests.test_component_contracts `
        tests.test_architecture_boundaries `
        tests.test_legacy_component_adapters `
        tests.test_frame_pump `
        tests.test_invocation_trace `
        tests.test_restore_workbench_backup `
        -v
}

exit $LASTEXITCODE
