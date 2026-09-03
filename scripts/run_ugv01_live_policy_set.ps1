param(
    [string]$RoverUrl = "http://10.0.0.171/js",
    [string]$PhysicalCondition = "turning_intensive",
    [string]$WirelessCondition = "wifi_baseline",
    [int]$Trial = 1,
    [int]$DurationSeconds = 120,
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8765,
    [string]$Python = "python",
    [string]$MotionProfile = "turning_intensive",
    [string]$MotionSpeed = "slow",
    [int]$RestSeconds = 15,
    [switch]$Open
)

$ErrorActionPreference = "Stop"

$policies = @("static-low", "static-high", "aoi-only", "contract-aware")
$openArg = @()
if ($Open) {
    $openArg = @("--open")
}

for ($policyIndex = 0; $policyIndex -lt $policies.Count; $policyIndex++) {
    $policy = $policies[$policyIndex]
    $safePolicy = $policy.Replace("-", "_")
    $runLabel = "${PhysicalCondition}_${WirelessCondition}_${safePolicy}_trial${Trial}"
    Write-Host ""
    Write-Host "==> Starting $policy | $PhysicalCondition | $WirelessCondition | trial $Trial" -ForegroundColor Cyan
    Write-Host "Automated motion: $MotionProfile at $MotionSpeed speed for $DurationSeconds seconds." -ForegroundColor Yellow

    $dashboardArgs = @(
        "-m", "DigitalTwin.dashboard.server",
        "--mode", "live",
        "--rover-url", $RoverUrl,
        "--host", $HostAddress,
        "--port", "$Port",
        "--policy", $policy,
        "--run-label", $runLabel,
        "--physical-condition", $PhysicalCondition,
        "--wireless-condition", $WirelessCondition,
        "--trial", "$Trial",
        "--duration-s", "$DurationSeconds",
        "--notes", "compact_live_contract_policy_set"
    ) + $openArg

    $serverProcess = Start-Process -FilePath $Python -ArgumentList $dashboardArgs -PassThru -WindowStyle Hidden

    # Wait for the dashboard server to become available
    $dashboardReady = $false
    $maxWaitSeconds = 30

    for ($i = 0; $i -lt ($maxWaitSeconds * 2); $i++) {
        if ($serverProcess.HasExited) {
            throw "Dashboard server exited before becoming ready for policy $policy"
        }

        try {
            $response = Invoke-WebRequest `
                -Uri "http://${HostAddress}:$Port" `
                -UseBasicParsing `
                -TimeoutSec 2 `
                -ErrorAction Stop

            $dashboardReady = $true
            break
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }

    if (-not $dashboardReady) {
        throw "Dashboard server did not become ready within $maxWaitSeconds seconds for policy $policy"
    }

    Write-Host "Dashboard ready. Starting automated motion..." -ForegroundColor Green

    try {
        & $Python -m DigitalTwin.dashboard.automated_motion `
            --dashboard-url "http://${HostAddress}:$Port" `
            --profile $MotionProfile `
            --speed $MotionSpeed `
            --duration-s ([Math]::Max(1, $DurationSeconds - 5))
        if ($LASTEXITCODE -ne 0) {
            throw "Automated motion failed for policy $policy"
        }
        Wait-Process -Id $serverProcess.Id -Timeout ([Math]::Max(5, $DurationSeconds + 10)) -ErrorAction SilentlyContinue
    }
    finally {
        try {
            & $Python -m DigitalTwin.dashboard.automated_motion `
                --dashboard-url "http://${HostAddress}:$Port" `
                --profile stop_only `
                --speed slow `
                --duration-s 1 | Out-Null
        } catch {
            Write-Host "Final stop request could not be confirmed; checking server process." -ForegroundColor Yellow
        }
        if (-not $serverProcess.HasExited) {
            Stop-Process -Id $serverProcess.Id -Force
        }
    }

    if ($RestSeconds -gt 0 -and $policyIndex -lt ($policies.Count - 1)) {
        Write-Host "Resting stopped rover for $RestSeconds seconds before next policy arm." -ForegroundColor Yellow
        Start-Sleep -Seconds $RestSeconds
    }
}

Write-Host ""
Write-Host "Policy set complete. Analyze with:" -ForegroundColor Green
Write-Host "python -m DigitalTwin.analysis.analyze_live_contract_logs"
