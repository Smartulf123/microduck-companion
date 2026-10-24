# Opens everything: the 3D duck (Windows), Pollen's robot software (WSL), then voice control here.
#   powershell -ExecutionPolicy Bypass -File start.ps1
# Close this window (or Ctrl-C) and the rest shuts down with it.
$root = $PSScriptRoot
$ProgressPreference = "SilentlyContinue"
$py = "$root\.venv\Scripts\python.exe"

function Stop-Duck {
    Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
        Where-Object { $_.CommandLine -match 'sim_world' } | ForEach-Object { Stop-Process $_.ProcessId -Force -ErrorAction SilentlyContinue }
    wsl -d Ubuntu-24.04 -- pkill -x robotd
    wsl -d Ubuntu-24.04 -- pkill -x tofd
    wsl -d Ubuntu-24.04 -- pkill -x socat
}
Stop-Duck  # leftovers from a window that was closed instead of Ctrl-C'd
# The Windows address WSL can reach; the body listens only there, not on the LAN.
$hostIp = (wsl -d Ubuntu-24.04 -- bash -c "ip route show default | cut -d' ' -f3").Trim()

$env:PYTHONPATH = "$root\ref\microduck_rl\src;$root"
$body = Start-Process "$root\.venv\Scripts\pythonw.exe" -PassThru -ArgumentList "-m", "sim_world",
    "--host", $hostIp, "--port", "7801", "--keyframe", "SIT"
$simSh = (wsl -d Ubuntu-24.04 -- wslpath -a ($root -replace '\\', '/')).Trim() + "/sim.sh"
$daemons = Start-Process wsl -PassThru -WindowStyle Minimized -ArgumentList "-d", "Ubuntu-24.04", "--",
    "bash", $simSh

Write-Host "starting the duck..."
while (-not (Test-NetConnection 127.0.0.1 -Port 7810 -InformationLevel Quiet -WarningAction SilentlyContinue)) {
    if ($body.HasExited -or $daemons.HasExited) { Write-Host "the sim did not start"; exit 1 }
    Start-Sleep 1
}
try {
    & $py "$root\main.py" @args
} finally {
    Stop-Process $daemons.Id -ErrorAction SilentlyContinue
    Stop-Duck
}
