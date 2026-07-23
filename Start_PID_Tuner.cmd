@echo off
chcp 65001 >nul
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%USERPROFILE%\STM32CubeIDE\workspace_1.19.0\serialPlotTest\llm-pid-tuner-main\start_tuner.ps1"
