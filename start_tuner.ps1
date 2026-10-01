param(
    [string]$Port = "COM3"
)

$ErrorActionPreference = "Stop"
$projectDir = $PSScriptRoot
$pythonExe = Join-Path $projectDir ".venv\Scripts\python.exe"
$tunerScript = Join-Path $projectDir "tuner.py"

function Wait-BeforeExit {
    Write-Host ""
    Read-Host "Press Enter to close this window"
}

function Read-TuningAxes {
    Write-Host ""
    Write-Host "请选择要调参的轴：X、Y、YAW。将按输入顺序依次调试。"
    Write-Host "示例：X    X Y    X/Y/YAW"
    do {
        $axisInput = Read-Host "输入一个、两个或三个轴"
        $axisTokens = @($axisInput.Trim().ToUpperInvariant() -split '[\s,/;，、；]+' | Where-Object { $_ })
        $invalidAxes = @($axisTokens | Where-Object { $_ -notin @("X", "Y", "YAW") })
        $validSelection = $axisTokens.Count -gt 0 -and $invalidAxes.Count -eq 0
        if (-not $validSelection) {
            Write-Host "输入无效。只能输入 X、Y、YAW，多个轴用空格、逗号或 / 分隔。" -ForegroundColor Yellow
        }
    } while (-not $validSelection)
    return @($axisTokens | Select-Object -Unique)
}

try {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Mecanum Chassis PID Tuner Launcher" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    if (-not (Test-Path -LiteralPath $pythonExe)) {
        throw "Virtual environment not found: $pythonExe`nCreate .venv and install requirements first."
    }
    if (-not (Test-Path -LiteralPath $tunerScript)) {
        throw "tuner.py not found: $tunerScript"
    }

    $selectedAxes = @(Read-TuningAxes)
    Write-Host "本次调参顺序：$($selectedAxes -join ' -> ')" -ForegroundColor Cyan

    $availablePorts = [System.IO.Ports.SerialPort]::GetPortNames()
    if ($Port -notin $availablePorts) {
        $portText = if ($availablePorts.Count -gt 0) { $availablePorts -join ", " } else { "none" }
        throw "$Port was not detected. Available ports: $portText"
    }

    if (-not $env:LLM_API_KEY) {
        $secureKey = Read-Host "Enter DeepSeek API Key (input is hidden)" -AsSecureString
        $keyPtr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
        try {
            $env:LLM_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPtr)
        }
        finally {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPtr)
        }
    }

    if ([string]::IsNullOrWhiteSpace($env:LLM_API_KEY)) {
        throw "API Key is empty. Launch cancelled."
    }

    Write-Host ""
    Write-Host "Safety checklist:" -ForegroundColor Yellow
    Write-Host "1. Close serial assistants and the CubeIDE serial terminal."
    Write-Host "2. Check STM32, OPS9 and motor power."
    Write-Host "3. Lift the wheels for the first test and keep power cutoff accessible."
    Write-Host "4. Tuning serial port: $Port"
    Write-Host "5. Selected axes: $($selectedAxes -join ' -> ')"
    Write-Host ""
    Read-Host "Press Enter to start, or close this window to cancel"

    Set-Location -LiteralPath $projectDir
    $tunerArgs = @($tunerScript, $Port, "--plain", "--axes") + $selectedAxes
    & $pythonExe @tunerArgs

    if ($LASTEXITCODE -ne 0) {
        throw "Tuner exited with code $LASTEXITCODE"
    }
}
catch {
    Write-Host ""
    Write-Host "Launch failed: $($_.Exception.Message)" -ForegroundColor Red
}
finally {
    Remove-Item Env:LLM_API_KEY -ErrorAction SilentlyContinue
    Wait-BeforeExit
}
