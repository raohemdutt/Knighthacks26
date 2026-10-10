# Open the Jetson Orin Nano serial console over the USB-C cable (Windows).
# Usage: .\connect-serial.ps1            (auto-detect port)
#        .\connect-serial.ps1 -Port COM7 (force a port)
param(
    [string]$Port,
    [int]$Baud = 115200
)

if (-not $Port) {
    # The Jetson enumerates as a USB CDC-ACM serial device with NVIDIA's vendor ID 0955.
    $dev = Get-PnpDevice -PresentOnly -Class Ports -ErrorAction SilentlyContinue |
        Where-Object { $_.InstanceId -match 'VID_0955' } |
        Select-Object -First 1
    if ($dev -and $dev.FriendlyName -match '\((COM\d+)\)') {
        $Port = $Matches[1]
    }
}

if (-not $Port) {
    Write-Host "No Jetson serial port found." -ForegroundColor Red
    Write-Host "Check: USB-C cable is data-capable, board is powered from the barrel jack, waited ~1 min after power-on."
    Write-Host "If nothing ever appears, the board likely needs a firmware update (boot JetPack 5.1.3 once, then JetPack 6)."
    Write-Host "Available ports:"; [System.IO.Ports.SerialPort]::GetPortNames()
    exit 1
}

Write-Host "Connecting to $Port at $Baud baud..."
if (Get-Command plink -ErrorAction SilentlyContinue) {
    plink -serial $Port -sercfg "$Baud,8,n,1,N"
} elseif (Get-Command putty -ErrorAction SilentlyContinue) {
    putty -serial $Port -sercfg "$Baud,8,n,1,N"
} else {
    Write-Host "PuTTY not found. Install it with: winget install PuTTY.PuTTY" -ForegroundColor Yellow
    Write-Host "Or open PuTTY manually: Connection type Serial, line $Port, speed $Baud."
    exit 1
}
