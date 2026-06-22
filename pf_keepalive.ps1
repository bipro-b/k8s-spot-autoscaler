# Keeps kubectl port-forward svc/workload alive on a fixed port.
# Usage: powershell -ExecutionPolicy Bypass -File pf_keepalive.ps1
param([int]$LocalPort = 8080, [string]$Namespace = "bench", [string]$Svc = "workload", [string]$SvcPort = "80")
while ($true) {
    Write-Host "$(Get-Date -Format HH:mm:ss) Starting port-forward $LocalPort -> svc/$Svc`:$SvcPort"
    $proc = Start-Process kubectl -ArgumentList "port-forward","svc/$Svc","-n",$Namespace,"${LocalPort}:${SvcPort}" -PassThru -NoNewWindow
    $proc.WaitForExit()
    Write-Host "$(Get-Date -Format HH:mm:ss) port-forward exited (code $($proc.ExitCode)). Restarting in 1s..."
    Start-Sleep -Seconds 1
}
