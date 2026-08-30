param(
    [ValidateRange(1, 100)]
    [int]$Runs = 3,
    [string]$GameId = "genshin-impact",
    [string]$CameraId,
    [string]$PhoneSerial,
    [string]$OutputRoot,
    [ValidateRange(0, 3600)]
    [int]$PauseSeconds = 5,
    [ValidateRange(0, 300)]
    [int]$GamePreparationSeconds = 20,
    [ValidateRange(1, 500)]
    [int]$MaximumEvidenceImagesPerRun = 80,
    [switch]$StopOnFailure,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$pipeline = Join-Path $root "calibrate-hik-game-camera.ps1"
$sessionRoot = Join-Path $root "sessions\calibration"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $root "artifacts\hik-calibration-stress-$timestamp"
}
$OutputRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputRoot)

function Write-JsonFile {
    param([string]$Path, $Value)
    $temporary = "$Path.tmp"
    $Value | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Read-JsonFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    try {
        return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    }
    catch {
        return $null
    }
}

function Add-ScalarMetric {
    param([System.Collections.IDictionary]$Metrics, [string]$Name, $Value)
    if ($null -eq $Value) { return }
    try {
        $number = [double]$Value
        if (-not [double]::IsNaN($number) -and -not [double]::IsInfinity($number)) {
            $Metrics[$Name] = $number
        }
    }
    catch {}
}

function Extract-Metrics {
    param($Rig, $Minimap)
    $metrics = [ordered]@{}
    if ($null -ne $Rig) {
        $screen = $Rig.geometry.camera_visible_screen_region.xywh
        $roi = $Rig.camera.hardware_roi_xywh
        if ($screen.Count -eq 4) {
            Add-ScalarMetric $metrics "rig.screen_region.x_px" $screen[0]
            Add-ScalarMetric $metrics "rig.screen_region.y_px" $screen[1]
            Add-ScalarMetric $metrics "rig.screen_region.width_px" $screen[2]
            Add-ScalarMetric $metrics "rig.screen_region.height_px" $screen[3]
        }
        if ($roi.Count -eq 4) {
            Add-ScalarMetric $metrics "rig.hardware_roi.x_px" $roi[0]
            Add-ScalarMetric $metrics "rig.hardware_roi.y_px" $roi[1]
            Add-ScalarMetric $metrics "rig.hardware_roi.width_px" $roi[2]
            Add-ScalarMetric $metrics "rig.hardware_roi.height_px" $roi[3]
        }
        Add-ScalarMetric $metrics "rig.screen_iou" $Rig.geometry.camera_visible_screen_region.screen_view_iou
        Add-ScalarMetric $metrics "rig.exposure_us" $Rig.imaging.exposure_us
        Add-ScalarMetric $metrics "rig.gain_db" $Rig.imaging.gain
        Add-ScalarMetric $metrics "rig.wb.red" $Rig.imaging.white_balance.ratio_red
        Add-ScalarMetric $metrics "rig.wb.green" $Rig.imaging.white_balance.ratio_green
        Add-ScalarMetric $metrics "rig.wb.blue" $Rig.imaging.white_balance.ratio_blue
        Add-ScalarMetric $metrics "rig.reprojection_p95_px" $Rig.results.cv_verification.reprojection_error_camera_px_p95
        Add-ScalarMetric $metrics "rig.cross_source.confidence" $Rig.results.cross_source_check.confidence
        Add-ScalarMetric $metrics "rig.cross_source.grayscale_correlation" $Rig.results.cross_source_check.grayscale_correlation
        Add-ScalarMetric $metrics "rig.display_response_p50_ms" $Rig.results.latency_benchmark.request_to_first_stable_ms.p50
    }
    if ($null -ne $Minimap) {
        $boundary = $Minimap.android.outer_boundary
        $hik = $Minimap.hik_session_observation
        Add-ScalarMetric $metrics "minimap.android.center_x_px" $boundary.center_x
        Add-ScalarMetric $metrics "minimap.android.center_y_px" $boundary.center_y
        Add-ScalarMetric $metrics "minimap.android.radius_px" $boundary.radius
        Add-ScalarMetric $metrics "minimap.android.confidence" $boundary.confidence
        Add-ScalarMetric $metrics "minimap.android.radial_rmse_px" $boundary.radial_rmse_px
        Add-ScalarMetric $metrics "minimap.hik.center_x_px" $hik.center_xy[0]
        Add-ScalarMetric $metrics "minimap.hik.center_y_px" $hik.center_xy[1]
        Add-ScalarMetric $metrics "minimap.hik.visible_circumference" $hik.visible_circumference_fraction
        Add-ScalarMetric $metrics "minimap.color.gamma" $Minimap.hik_bayer_conversion.gamma
        Add-ScalarMetric $metrics "minimap.color.baseline_rgb_mae_dn" $Minimap.hik_bayer_conversion.fit.baseline_validation.rgb_mae_dn
        Add-ScalarMetric $metrics "minimap.color.adjusted_rgb_mae_dn" $Minimap.hik_bayer_conversion.fit.selected_validation.rgb_mae_dn
    }
    return $metrics
}

function Get-SessionSnapshot {
    $snapshot = @{}
    if (Test-Path -LiteralPath $sessionRoot -PathType Container) {
        Get-ChildItem -LiteralPath $sessionRoot -Directory | ForEach-Object {
            $snapshot[$_.FullName] = $true
        }
    }
    return $snapshot
}

function Invoke-ObservedPipeline {
    param(
        [string]$Executable,
        [string[]]$Arguments,
        [string]$LogPath,
        [int]$PreparationSeconds
    )
    $quotedArguments = @($Arguments | ForEach-Object {
        '"{0}"' -f ([string]$_).Replace('"', '\"')
    }) -join " "
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $Executable
    $startInfo.Arguments = $quotedArguments
    $startInfo.WorkingDirectory = $root
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardInput = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
        throw "Failed to start stress pipeline: $Executable"
    }
    $process.StandardInput.AutoFlush = $true
    $channels = @(
        [pscustomobject]@{ name = "stdout"; reader = $process.StandardOutput; task = $process.StandardOutput.ReadLineAsync(); closed = $false },
        [pscustomobject]@{ name = "stderr"; reader = $process.StandardError; task = $process.StandardError.ReadLineAsync(); closed = $false }
    )
    $lines = New-Object System.Collections.Generic.List[string]
    $preparationDeadline = $null
    $confirmationSent = $false
    $exitCode = $null

    function Add-ObservedLine {
        param([string]$Message)
        $timestamped = "{0}`t{1}" -f (Get-Date).ToUniversalTime().ToString("o"), $Message
        $lines.Add($timestamped)
        Write-Host $Message
    }

    try {
        while (-not $process.HasExited -or @($channels | Where-Object { -not $_.closed }).Count -gt 0) {
            $madeProgress = $false
            foreach ($channel in $channels) {
                if ($channel.closed -or -not $channel.task.IsCompleted) { continue }
                $madeProgress = $true
                $line = $channel.task.GetAwaiter().GetResult()
                if ($null -eq $line) {
                    $channel.closed = $true
                    continue
                }
                Add-ObservedLine ([string]$line)
                if ($null -eq $preparationDeadline -and $line -match '^This command records data only') {
                    $preparationDeadline = (Get-Date).AddSeconds($PreparationSeconds)
                    Add-ObservedLine (
                        "[stress] Existing wake/launch preparation completed; " +
                        "waiting $PreparationSeconds seconds for the game to settle/reconnect."
                    )
                }
                $channel.task = $channel.reader.ReadLineAsync()
            }
            if (
                -not $confirmationSent -and
                $null -ne $preparationDeadline -and
                (Get-Date) -ge $preparationDeadline
            ) {
                $process.StandardInput.WriteLine("")
                $confirmationSent = $true
                Add-ObservedLine "[stress] Confirmed the existing game-preparation checkpoint."
            }
            if (-not $madeProgress) { Start-Sleep -Milliseconds 50 }
        }
        $process.WaitForExit()
        $exitCode = $process.ExitCode
    }
    finally {
        if (-not $process.HasExited) { $process.Kill() }
        $process.Dispose()
    }
    $lines | Set-Content -LiteralPath $LogPath -Encoding UTF8
    return [pscustomobject]@{
        exit_code = $exitCode
        lines = @($lines)
        preparation_checkpoint_observed = ($null -ne $preparationDeadline)
        preparation_confirmation_sent = $confirmationSent
    }
}

function Read-TimestampedLogEntries {
    param([string[]]$Lines)
    $entries = @()
    foreach ($line in $Lines) {
        if ($line -match '^(?<time>[^\t]+)\t(?<message>.*)$') {
            try {
                $entries += [pscustomobject]@{
                    time = [datetime]::Parse($Matches.time).ToUniversalTime()
                    message = $Matches.message
                }
            }
            catch {}
        }
    }
    return $entries
}

function Get-PipelineTimingBreakdown {
    param([string[]]$LogLines, [datetime]$PipelineStarted, [datetime]$PipelineFinished, [string]$RunStatus)
    $entries = @(Read-TimestampedLogEntries $LogLines)
    $stageDefinitions = @(
        [ordered]@{ name = "rig_precheck"; pattern = '^\[0/5\]' },
        [ordered]@{ name = "rig_calibration"; pattern = '^\[1-2/5\]' },
        [ordered]@{ name = "dual_source_capture"; pattern = '^\[3-4/5\]' },
        [ordered]@{ name = "minimap_localization"; pattern = '^\[5/5\]' }
    )
    $observedStages = @()
    foreach ($definition in $stageDefinitions) {
        $entry = @($entries | Where-Object { $_.message -match $definition.pattern } | Select-Object -First 1)
        if ($entry.Count -gt 0) {
            $observedStages += [pscustomobject]@{ name = $definition.name; time = $entry[0].time; message = $entry[0].message }
        }
    }
    $observedStages = @($observedStages | Sort-Object time)
    $stages = @()
    for ($index = 0; $index -lt $observedStages.Count; $index++) {
        $current = $observedStages[$index]
        $end = if ($index + 1 -lt $observedStages.Count) { $observedStages[$index + 1].time } else { $PipelineFinished.ToUniversalTime() }
        $stageStatus = if ($index + 1 -lt $observedStages.Count -or $RunStatus -eq "passed") { "completed" } else { "failed_or_incomplete" }
        $stages += [ordered]@{
            name = $current.name
            status = $stageStatus
            started_utc = $current.time.ToString("o")
            finished_utc = $end.ToString("o")
            duration_seconds = ($end - $current.time).TotalSeconds
            marker = $current.message
        }
    }

    $milestoneDefinitions = @(
        [ordered]@{ name = "rig_read_specs"; pattern = '^Reading phone and HIK camera specifications' },
        [ordered]@{ name = "rig_charuco_geometry"; pattern = '^Showing ChArUco atlas' },
        [ordered]@{ name = "rig_black_level"; pattern = '^Checking .*black-level' },
        [ordered]@{ name = "rig_one_shot_auto"; pattern = '^Running HIK one-shot auto exposure' },
        [ordered]@{ name = "rig_exposure_lock"; pattern = '^Calibrating refresh-quantized exposure' },
        [ordered]@{ name = "rig_white_balance"; pattern = '^(Checking one residual white-balance|Calibrating white balance)' },
        [ordered]@{ name = "rig_final_cv_verification"; pattern = '^Final imaging verification' },
        [ordered]@{ name = "rig_latency_benchmark"; pattern = '^Benchmarking final' },
        [ordered]@{ name = "rig_cross_source_check"; pattern = '^Cross-source alignment check' },
        [ordered]@{ name = "game_preparation_wait_started"; pattern = '^\[stress\] Existing wake/launch preparation completed' },
        [ordered]@{ name = "game_preparation_confirmed"; pattern = '^\[stress\] Confirmed the existing game-preparation checkpoint' },
        [ordered]@{ name = "zigzag_capture_complete"; pattern = '^Captured .*zigzag' },
        [ordered]@{ name = "minimap_localization_complete"; pattern = '^Mini-map localization:' }
    )
    $milestones = @()
    foreach ($definition in $milestoneDefinitions) {
        $entry = @($entries | Where-Object { $_.message -match $definition.pattern } | Select-Object -First 1)
        if ($entry.Count -gt 0) {
            $milestones += [pscustomobject]@{ name = $definition.name; time = $entry[0].time; message = $entry[0].message }
        }
    }
    $milestones = @($milestones | Sort-Object time)
    $milestoneIntervals = @()
    for ($index = 0; $index -lt $milestones.Count; $index++) {
        $current = $milestones[$index]
        $end = if ($index + 1 -lt $milestones.Count) { $milestones[$index + 1].time } else { $PipelineFinished.ToUniversalTime() }
        $milestoneIntervals += [ordered]@{
            name = $current.name
            observed_utc = $current.time.ToString("o")
            seconds_until_next_observed_milestone_or_pipeline_end = ($end - $current.time).TotalSeconds
            marker = $current.message
        }
    }
    $preparationStart = @($milestones | Where-Object { $_.name -eq "game_preparation_wait_started" } | Select-Object -First 1)
    $preparationEnd = @($milestones | Where-Object { $_.name -eq "game_preparation_confirmed" } | Select-Object -First 1)
    return [ordered]@{
        pipeline_total_seconds = ($PipelineFinished - $PipelineStarted).TotalSeconds
        game_preparation_seconds = if ($preparationStart.Count -gt 0 -and $preparationEnd.Count -gt 0) {
            ($preparationEnd[0].time - $preparationStart[0].time).TotalSeconds
        } else { $null }
        stages = $stages
        observed_milestone_intervals = $milestoneIntervals
        note = "Milestones are passive timestamps from existing output; missing markers are reported by omission and are never synthesized."
    }
}

function Copy-ReviewEvidence {
    param([string[]]$SourceRoots, [string]$Destination, [int]$Maximum)
    $destinationFullPath = [System.IO.Path]::GetFullPath($Destination)
    $destinationPrefix = $destinationFullPath.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    $candidates = @()
    foreach ($source in $SourceRoots) {
        if (-not $source -or -not (Test-Path -LiteralPath $source -PathType Container)) { continue }
        $candidates += Get-ChildItem -LiteralPath $source -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object {
                $_.Extension.ToLowerInvariant() -in @(".png", ".jpg", ".jpeg") -and
                -not $_.FullName.StartsWith($destinationPrefix, [System.StringComparison]::OrdinalIgnoreCase)
            } |
            ForEach-Object {
                $name = $_.FullName.ToLowerInvariant()
                $priority = 10
                if ($name -match "failure") { $priority = 0 }
                elseif ($name -match "cross.source|side.by.side|triptych") { $priority = 1 }
                elseif ($name -match "overlay|heatmap|mask|review") { $priority = 2 }
                elseif ($name -match "charuco|camera|android|adb|hik") { $priority = 3 }
                [pscustomobject]@{ File = $_; Priority = $priority }
            }
    }
    $selected = @($candidates | Sort-Object Priority, @{Expression={$_.File.LastWriteTimeUtc}; Descending=$true} | Select-Object -First $Maximum)
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    $records = @()
    $index = 0
    foreach ($candidate in $selected) {
        $index += 1
        $safeName = $candidate.File.Name -replace '[^A-Za-z0-9._-]', '_'
        $destinationName = "{0:D3}-{1}" -f $index, $safeName
        $destinationPath = Join-Path $Destination $destinationName
        Copy-Item -LiteralPath $candidate.File.FullName -Destination $destinationPath -Force
        $records += [ordered]@{
            copy = $destinationPath
            source = $candidate.File.FullName
            priority = $candidate.Priority
            bytes = $candidate.File.Length
        }
    }
    return $records
}

function Build-Repeatability {
    param($RunResults, [bool]$SuccessfulOnly = $true)
    $population = if ($SuccessfulOnly) {
        @($RunResults | Where-Object { $_.status -eq "passed" })
    } else {
        @($RunResults)
    }
    $names = @($population | ForEach-Object { $_.metrics.Keys } | Sort-Object -Unique)
    $result = @()
    foreach ($name in $names) {
        $values = @($population | ForEach-Object {
            if ($_.metrics.Contains($name)) { [double]$_.metrics[$name] }
        })
        if ($values.Count -eq 0) { continue }
        $mean = ($values | Measure-Object -Average).Average
        $minimum = ($values | Measure-Object -Minimum).Minimum
        $maximum = ($values | Measure-Object -Maximum).Maximum
        $sumSquares = 0.0
        foreach ($value in $values) { $sumSquares += [math]::Pow($value - $mean, 2) }
        $stddev = if ($values.Count -gt 1) { [math]::Sqrt($sumSquares / ($values.Count - 1)) } else { 0.0 }
        $result += [ordered]@{
            metric = $name
            count = $values.Count
            mean = [double]$mean
            sample_stddev = [double]$stddev
            minimum = [double]$minimum
            maximum = [double]$maximum
            range = [double]($maximum - $minimum)
            coefficient_of_variation_percent = if ([math]::Abs($mean) -gt 1.0e-12) { [double](100.0 * $stddev / [math]::Abs($mean)) } else { $null }
            values = $values
        }
    }
    return $result
}

function Write-MarkdownReport {
    param([string]$Path, $Summary)
    $lines = @(
        "# HIK headless calibration stress report",
        "",
        "Generated: $($Summary.finished_utc)",
        "",
        "Requested runs: $($Summary.requested_runs)  ",
        "Passed: $($Summary.passed_runs)  ",
        "Review required: $($Summary.review_required_runs)  ",
        "Failed: $($Summary.failed_runs)  ",
        "",
        "The harness did not retry, repair, or modify any failed calibration stage.",
        "",
        "## Runs",
        "",
        "| Run | Status | Duration (s) | Log | Result |",
        "|---:|---|---:|---|---|"
    )
    foreach ($run in $Summary.runs) {
        $runDirectoryName = Split-Path -Leaf $run.run_directory
        $lines += "| $($run.run_index) | $($run.status) | $([math]::Round($run.duration_seconds, 1)) | [$runDirectoryName/run.log]($runDirectoryName/run.log) | [$runDirectoryName/run-result.json]($runDirectoryName/run-result.json) |"
        if ($run.status -ne "passed") {
            $lines += ""
            $lines += "Run $($run.run_index) issue: $($run.failure_summary)"
        }
    }
    $lines += @(
        "",
        "## Time breakdown",
        "",
        "| Run | Pipeline total (s) | Rig calibration (s) | Dual capture (s) | Mini-map localization (s) | Game preparation (s) | Evidence collection (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|"
    )
    foreach ($run in $Summary.runs) {
        $stageValues = @{}
        foreach ($stage in $run.timing.stages) { $stageValues[$stage.name] = $stage.duration_seconds }
        $rigTime = if ($stageValues.ContainsKey("rig_calibration")) { "{0:N3}" -f $stageValues["rig_calibration"] } else { "" }
        $captureTime = if ($stageValues.ContainsKey("dual_source_capture")) { "{0:N3}" -f $stageValues["dual_source_capture"] } else { "" }
        $minimapTime = if ($stageValues.ContainsKey("minimap_localization")) { "{0:N3}" -f $stageValues["minimap_localization"] } else { "" }
        $preparationTime = if ($null -eq $run.timing.game_preparation_seconds) { "" } else { "{0:N3}" -f $run.timing.game_preparation_seconds }
        $lines += "| $($run.run_index) | $("{0:N3}" -f $run.timing.pipeline_total_seconds) | $rigTime | $captureTime | $minimapTime | $preparationTime | $("{0:N3}" -f $run.timing.evidence_collection_seconds) |"
    }
    $lines += @("", "## Repeatability", "", "| Metric | Count | Mean | Stddev | Min | Max | Range | CV % |", "|---|---:|---:|---:|---:|---:|---:|---:|")
    foreach ($metric in $Summary.repeatability) {
        $cv = if ($null -eq $metric.coefficient_of_variation_percent) { "" } else { "{0:N4}" -f $metric.coefficient_of_variation_percent }
        $lines += "| $($metric.metric) | $($metric.count) | $("{0:N6}" -f $metric.mean) | $("{0:N6}" -f $metric.sample_stddev) | $("{0:N6}" -f $metric.minimum) | $("{0:N6}" -f $metric.maximum) | $("{0:N6}" -f $metric.range) | $cv |"
    }
    if ($Summary.partial_stage_repeatability.Count -gt 0) {
        $lines += @(
            "",
            "## Partial-stage repeatability",
            "",
            "These observations include metrics from failed end-to-end runs and must not be read as successful full-pipeline repeatability.",
            "",
            "| Metric | Count | Mean | Stddev | Min | Max | Range | CV % |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"
        )
        foreach ($metric in $Summary.partial_stage_repeatability) {
            $cv = if ($null -eq $metric.coefficient_of_variation_percent) { "" } else { "{0:N4}" -f $metric.coefficient_of_variation_percent }
            $lines += "| $($metric.metric) | $($metric.count) | $("{0:N6}" -f $metric.mean) | $("{0:N6}" -f $metric.sample_stddev) | $("{0:N6}" -f $metric.minimum) | $("{0:N6}" -f $metric.maximum) | $("{0:N6}" -f $metric.range) | $cv |"
        }
    }
    $lines | Set-Content -LiteralPath $Path -Encoding UTF8
}

if (-not (Test-Path -LiteralPath $pipeline -PathType Leaf)) {
    throw "Headless pipeline does not exist: $pipeline"
}
if (Test-Path -LiteralPath $OutputRoot) {
    throw "Stress output already exists: $OutputRoot"
}
New-Item -ItemType Directory -Path $OutputRoot | Out-Null

$startedUtc = (Get-Date).ToUniversalTime().ToString("o")
$plan = [ordered]@{
    schema_version = "1.0"
    purpose = "Stress and repeatability test of the existing complete headless HIK calibration pipeline"
    pipeline = $pipeline
    requested_runs = $Runs
    game_id = $GameId
    camera_id = $CameraId
    phone_serial = $PhoneSerial
    pause_seconds = $PauseSeconds
    game_preparation_seconds = $GamePreparationSeconds
    stop_on_failure = [bool]$StopOnFailure
    repair_or_retry = $false
    output_root = $OutputRoot
    started_utc = $startedUtc
}
Write-JsonFile (Join-Path $OutputRoot "plan.json") $plan

Write-Host "HIK headless calibration stress test" -ForegroundColor Cyan
Write-Host "Runs:   $Runs"
Write-Host "Output: $OutputRoot"
Write-Host "Policy: record and continue; no repair and no retry"
if ($PlanOnly) {
    Write-Host "Plan-only validation completed; pipeline was not started." -ForegroundColor Yellow
    exit 0
}

$runResults = @()
for ($runIndex = 1; $runIndex -le $Runs; $runIndex++) {
    $runName = "run-{0:D3}" -f $runIndex
    $runDirectory = Join-Path $OutputRoot $runName
    $rigOutput = Join-Path $runDirectory "rig"
    $minimapOutput = Join-Path $runDirectory "minimap"
    $logPath = Join-Path $runDirectory "run.log"
    New-Item -ItemType Directory -Path $runDirectory | Out-Null
    $sessionsBefore = Get-SessionSnapshot
    $runStarted = Get-Date

    Write-Host ""
    Write-Host "[$runIndex/$Runs] Starting isolated complete calibration" -ForegroundColor Cyan
    $arguments = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $pipeline,
        "-GameId", $GameId,
        "-RigOutput", $rigOutput,
        "-MinimapOutput", $minimapOutput
    )
    if ($CameraId) { $arguments += @("-CameraId", $CameraId) }
    if ($PhoneSerial) { $arguments += @("-PhoneSerial", $PhoneSerial) }

    # The existing capture stage owns wake-up, game launch, and Game Booster
    # dismissal, then deliberately waits at a human checkpoint. The stress
    # harness observes that checkpoint, allows a bounded reconnect/settle
    # period, and confirms it through stdin without altering pipeline code.
    $observed = Invoke-ObservedPipeline `
        -Executable "powershell.exe" `
        -Arguments $arguments `
        -LogPath $logPath `
        -PreparationSeconds $GamePreparationSeconds
    $exitCode = $observed.exit_code
    $pipelineFinished = Get-Date
    $newSessions = @()
    if (Test-Path -LiteralPath $sessionRoot -PathType Container) {
        $newSessions = @(Get-ChildItem -LiteralPath $sessionRoot -Directory | Where-Object { -not $sessionsBefore.ContainsKey($_.FullName) } | Sort-Object LastWriteTimeUtc)
    }

    $rigPath = Join-Path $rigOutput "hik_camera_calibration.json"
    $rigReusePath = Join-Path $rigOutput "reused_calibration.json"
    $rigReuse = Read-JsonFile $rigReusePath
    if (-not (Test-Path -LiteralPath $rigPath -PathType Leaf) -and $null -ne $rigReuse) {
        $rigPath = [string]$rigReuse.calibration
    }
    $minimapPath = Join-Path $minimapOutput "localization_summary.json"
    $rig = Read-JsonFile $rigPath
    $minimap = Read-JsonFile $minimapPath
    $executionStatus = if ($exitCode -eq 0 -and $null -ne $rig -and $null -ne $minimap -and $newSessions.Count -eq 1) { "completed" } else { "failed" }
    $qualityStatus = if ($null -ne $minimap -and $minimap.status) { [string]$minimap.status } else { "unavailable" }
    $status = if ($executionStatus -eq "failed") {
        "failed"
    } elseif ($qualityStatus -eq "review_required") {
        "review_required"
    } else {
        "passed"
    }
    $logLines = if (Test-Path -LiteralPath $logPath) { @(Get-Content -LiteralPath $logPath) } else { @() }
    $failureSummary = $null
    if ($status -eq "review_required") {
        $failureSummary = "Mini-map localization completed with status review_required"
    } elseif ($status -eq "failed") {
        $failureLine = @($logLines | Where-Object { $_ -match "failed|failure|traceback|stopped|error" } | Select-Object -Last 1)
        if ($failureLine.Count -gt 0) {
            $failureSummary = ([string]$failureLine[0]) -replace '^[^\t]+\t', ''
        }
        elseif ($exitCode -ne 0) { $failureSummary = "Pipeline exited with code $exitCode" }
        elseif ($null -eq $rig) { $failureSummary = "Rig calibration JSON is missing or unreadable" }
        elseif ($newSessions.Count -ne 1) { $failureSummary = "Expected one new session, observed $($newSessions.Count)" }
        else { $failureSummary = "Mini-map localization JSON is missing or unreadable" }
    }

    $timing = Get-PipelineTimingBreakdown $logLines $runStarted $pipelineFinished $(if ($executionStatus -eq "completed") { "passed" } else { "failed" })
    $sourceRoots = @($runDirectory) + @($newSessions | ForEach-Object { $_.FullName })
    $evidenceDirectory = Join-Path $runDirectory "review-evidence"
    $evidenceStarted = Get-Date
    $evidence = @(Copy-ReviewEvidence `
        -SourceRoots $sourceRoots `
        -Destination $evidenceDirectory `
        -Maximum $MaximumEvidenceImagesPerRun)
    $evidenceFinished = Get-Date
    $timing["evidence_collection_seconds"] = ($evidenceFinished - $evidenceStarted).TotalSeconds
    $metrics = Extract-Metrics $rig $minimap
    Add-ScalarMetric $metrics "time.pipeline_total_seconds" $timing.pipeline_total_seconds
    Add-ScalarMetric $metrics "time.game_preparation_seconds" $timing.game_preparation_seconds
    Add-ScalarMetric $metrics "time.evidence_collection_seconds" $timing.evidence_collection_seconds
    foreach ($stage in $timing.stages) {
        Add-ScalarMetric $metrics ("time.stage.{0}_seconds" -f $stage.name) $stage.duration_seconds
    }
    $runFinished = Get-Date
    $timing["harness_total_seconds"] = ($runFinished - $runStarted).TotalSeconds
    $result = [ordered]@{
        schema_version = "1.0"
        run_index = $runIndex
        status = $status
        pipeline_execution_status = $executionStatus
        calibration_quality_status = $qualityStatus
        exit_code = $exitCode
        started_utc = $runStarted.ToUniversalTime().ToString("o")
        finished_utc = $runFinished.ToUniversalTime().ToString("o")
        duration_seconds = ($runFinished - $runStarted).TotalSeconds
        run_directory = $runDirectory
        log = $logPath
        rig_calibration = if ($null -ne $rig) { $rigPath } else { $null }
        rig_calibration_reuse = if ($null -ne $rigReuse) { $rigReusePath } else { $null }
        minimap_localization = if ($null -ne $minimap) { $minimapPath } else { $null }
        new_sessions = @($newSessions | ForEach-Object { $_.FullName })
        failure_summary = $failureSummary
        failure_log_tail = if ($status -eq "failed") { @($logLines | Select-Object -Last 80) } else { @() }
        timing = $timing
        metrics = $metrics
        evidence = $evidence
        game_preparation = [ordered]@{
            settle_seconds = $GamePreparationSeconds
            checkpoint_observed = $observed.preparation_checkpoint_observed
            confirmation_sent = $observed.preparation_confirmation_sent
        }
        repair_attempted = $false
        retry_attempted = $false
    }
    Write-JsonFile (Join-Path $runDirectory "run-result.json") $result
    $runResults += $result
    $color = if ($status -eq "passed") { "Green" } elseif ($status -eq "review_required") { "Yellow" } else { "Red" }
    Write-Host "[$runIndex/$Runs] $status; evidence: $runDirectory" -ForegroundColor $color

    if ($status -ne "passed" -and $StopOnFailure) { break }
    if ($runIndex -lt $Runs -and $PauseSeconds -gt 0) {
        Write-Host "Cooling/settling pause: $PauseSeconds seconds"
        Start-Sleep -Seconds $PauseSeconds
    }
}

$passed = @($runResults | Where-Object { $_.status -eq "passed" }).Count
$reviewRequired = @($runResults | Where-Object { $_.status -eq "review_required" }).Count
$failed = @($runResults | Where-Object { $_.status -eq "failed" }).Count
$failureSignatures = @(
    $runResults |
    Where-Object { $_.status -ne "passed" } |
    Group-Object status, failure_summary |
    ForEach-Object {
        [ordered]@{
            issue = $_.Name
            count = $_.Count
            runs = @($_.Group | ForEach-Object { $_.run_index })
        }
    }
)
$summary = [ordered]@{
    schema_version = "1.0"
    purpose = $plan.purpose
    pipeline = $pipeline
    requested_runs = $Runs
    completed_runs = $runResults.Count
    passed_runs = $passed
    review_required_runs = $reviewRequired
    failed_runs = $failed
    started_utc = $startedUtc
    finished_utc = (Get-Date).ToUniversalTime().ToString("o")
    output_root = $OutputRoot
    policy = [ordered]@{
        repair_attempted = $false
        retry_attempted = $false
        failures_are_retained = $true
    }
    runs = $runResults
    failure_signatures = $failureSignatures
    repeatability = @(Build-Repeatability -RunResults $runResults -SuccessfulOnly $true)
    partial_stage_repeatability = @(Build-Repeatability -RunResults $runResults -SuccessfulOnly $false)
}
Write-JsonFile (Join-Path $OutputRoot "summary.json") $summary
Write-MarkdownReport (Join-Path $OutputRoot "report.md") $summary

Write-Host ""
Write-Host "Stress test complete: $passed passed, $reviewRequired review required, $failed failed" -ForegroundColor $(if ($reviewRequired -eq 0 -and $failed -eq 0) { "Green" } else { "Yellow" })
Write-Host "Report: $(Join-Path $OutputRoot 'report.md')"
Write-Host "Summary: $(Join-Path $OutputRoot 'summary.json')"
exit $(if ($reviewRequired -eq 0 -and $failed -eq 0) { 0 } else { 1 })
