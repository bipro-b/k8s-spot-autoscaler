# Keeps the minikube service tunnel alive for the workload on a fixed port.
# Run in a separate terminal: powershell -ExecutionPolicy Bypass -File tunnel_keepalive.ps1
param([int]$Port = 63735)
$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH","User")
while ($true) {
    Write-Host "$(Get-Date -Format HH:mm:ss) Starting tunnel on port $Port..."
    $proc = Start-Process minikube -ArgumentList "service","workload","-n","bench","--url","--https=false" -PassThru -RedirectStandardOutput "$PSScriptRoot\tunnel_url.txt" -NoNewWindow
    Start-Sleep -Seconds 3
    Write-Host "$(Get-Date -Format HH:mm:ss) Tunnel up (pid $($proc.Id)). Monitoring..."
    $proc.WaitForExit()
    Write-Host "$(Get-Date -Format HH:mm:ss) Tunnel exited. Restarting in 2s..."
    Start-Sleep -Seconds 2
}
