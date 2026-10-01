$ErrorActionPreference = "Stop"
$launcherPath = Join-Path (Split-Path $PSScriptRoot -Parent) "start_tuner.ps1"
$parseErrors = $null
$parseTokens = $null
$launcherAst = [System.Management.Automation.Language.Parser]::ParseFile(
    $launcherPath, [ref]$parseTokens, [ref]$parseErrors
)
if ($parseErrors.Count -gt 0) {
    throw "Launcher syntax errors: $parseErrors"
}

# Load only the input function, without running serial checks or moving hardware.
$axesFunction = $launcherAst.Find({
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $node.Name -eq "Read-TuningAxes"
}, $true)
if ($null -eq $axesFunction) { throw "Read-TuningAxes not found" }
. ([scriptblock]::Create($axesFunction.Extent.Text))

function Read-Host {
    param([string]$Prompt)
    if ($script:answers.Count -eq 0) { throw "Unexpected extra input prompt" }
    return $script:answers.Dequeue()
}

function Write-Host {
    param([string]$Message, [ConsoleColor]$ForegroundColor)
}

$testCases = @(
    @{ Inputs = @("x"); Expected = "X" },
    @{ Inputs = @("Y"); Expected = "Y" },
    @{ Inputs = @("yaw"); Expected = "YAW" },
    @{ Inputs = @("X Y"); Expected = "X|Y" },
    @{ Inputs = @("yaw/x/y"); Expected = "YAW|X|Y" },
    @{ Inputs = @("Y，X、YAW；x"); Expected = "Y|X|YAW" },
    @{ Inputs = @("X,X;YAW"); Expected = "X|YAW" },
    @{ Inputs = @("", " , / ", "Z", "X BAD", "XY", "X/Y/YAW"); Expected = "X|Y|YAW" }
)

foreach ($testCase in $testCases) {
    $script:answers = New-Object 'System.Collections.Generic.Queue[string]'
    foreach ($answer in $testCase.Inputs) { $script:answers.Enqueue($answer) }
    $actual = @(Read-TuningAxes) -join "|"
    if ($actual -ne $testCase.Expected -or $script:answers.Count -ne 0) {
        throw "Axis input failed: expected $($testCase.Expected), got $actual"
    }
}
[Console]::WriteLine("PASS: launcher parses and all {0} axis-input cases passed.", $testCases.Count)
