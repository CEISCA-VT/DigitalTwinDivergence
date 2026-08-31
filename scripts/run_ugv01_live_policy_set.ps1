param(
    [string]$RoverUrl = "http://192.168.4.1/js",
    [string]$PhysicalCondition = "turning_intensive",
    [string]$WirelessCondition = "wifi_baseline",
    [int]$Trial = 1,
    [int]$DurationSeconds = 120,
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8765,
    [string]$Python = "python",
    [switch]$Open
)

$ErrorActionPreference = "Stop"

$policies = @("static-low", "static-high", "aoi-only", "contract-aware")
$openArg = @()
if ($Open) {
    $openArg = @("--open")
}

foreach ($policy in $policies) {
    $safePolicy = $policy.Replace("-", "_")
    $runLabel = "${PhysicalCondition}_${WirelessCondition}_${safePolicy}_trial${Trial}"
    Write-Host ""
    Write-Host "==> Starting $policy | $PhysicalCondition | $WirelessCondition | trial $Trial" -ForegroundColor Cyan
    Write-Host "Use the same motion script for this policy arm. The server stops after $DurationSeconds seconds." -ForegroundColor Yellow
    & $Python -m DigitalTwin.dashboard.server `
        --mode live `
        --rover-url $RoverUrl `
        --host $HostAddress `
        --port $Port `
        --policy $policy `
        --run-label $runLabel `
        --physical-condition $PhysicalCondition `
        --wireless-condition $WirelessCondition `
        --trial $Trial `
        --duration-s $DurationSeconds `
        --notes "compact_live_contract_policy_set" `
        @openArg
    if ($LASTEXITCODE -ne 0) {
        throw "Dashboard run failed for policy $policy"
    }
}

Write-Host ""
Write-Host "Policy set complete. Analyze with:" -ForegroundColor Green
Write-Host "python -m DigitalTwin.analysis.analyze_live_contract_logs"
