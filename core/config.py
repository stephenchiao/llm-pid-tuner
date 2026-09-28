#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Global runtime configuration helpers."""

import io
import json
import os
import sys
from pathlib import Path
from typing import Any


def ensure_utf8_console() -> None:
    """Force UTF-8 console IO on Windows when possible."""
    if sys.platform != "win32":
        return

    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass

    def _retarget_stream_if_needed(stream: Any) -> Any:
        encoding = getattr(stream, "encoding", None)
        if not encoding or encoding.lower() == "utf-8":
            return None

        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", line_buffering=True)
                return None
            except Exception:
                pass

        is_tty = getattr(stream, "isatty", lambda: False)
        try:
            if is_tty():
                return None
        except Exception:
            pass

        buffer = getattr(stream, "buffer", None)
        if buffer is None:
            return None
        try:
            return io.TextIOWrapper(buffer, encoding="utf-8", line_buffering=True)
        except Exception:
            return None

    wrapped_stdout = _retarget_stream_if_needed(sys.stdout)
    if wrapped_stdout is not None:
        sys.stdout = wrapped_stdout

    wrapped_stderr = _retarget_stream_if_needed(sys.stderr)
    if wrapped_stderr is not None:
        sys.stderr = wrapped_stderr


# ============================================================================
# Default Configuration
# ============================================================================

DEFAULT_CONFIG: dict[str, Any] = {
    "SERIAL_PORT"                   : "AUTO",
    "BAUD_RATE"                     : 115200,
    "LLM_API_KEY"                   : "your-deepseek-api-key-here",
    "LLM_API_BASE_URL"              : "https://api.deepseek.com",
    "LLM_MODEL_NAME"                : "deepseek-flash",
    "LLM_PROVIDER"                  : "openai",
    "HTTP_PROXY"                    : "",
    "HTTPS_PROXY"                   : "",
    "ALL_PROXY"                     : "",
    "NO_PROXY"                      : "",
    "BUFFER_SIZE"                   : 100,
    "MIN_ERROR_THRESHOLD"           : 5.0,
    "MAX_TUNING_ROUNDS"             : 20,
    "LLM_REQUEST_TIMEOUT"           : 60,
    "LLM_DEBUG_OUTPUT"              : False,
    "GOOD_ENOUGH_AVG_ERROR"         : 80.0,
    "GOOD_ENOUGH_STEADY_STATE_ERROR": 10.0,
    "GOOD_ENOUGH_OVERSHOOT"         : 3.0,
    "REQUIRED_STABLE_ROUNDS"        : 3,
    "HARDWARE_STAGE_MAX_ROUNDS"     : 5,
    "HARDWARE_VERIFY_ROUNDS"        : 3,
    "HARDWARE_RESUME_LAST_PID"      : True,
    "HARDWARE_TUNE_AXIS"            : "X",
    "HARDWARE_INITIAL_PID_X"        : {"p": 0.00495, "i": 0.0, "d": 0.0},
    "HARDWARE_INITIAL_PID_Y"        : {"p": 0.0018, "i": 0.0, "d": 0.0},
    "HARDWARE_INITIAL_PID_YAW"      : {"p": 0.02, "i": 0.000015, "d": 0.0},
    "HARDWARE_OUTPUT_LIMIT_MPS"     : 0.20,
    "HARDWARE_YAW_OUTPUT_LIMIT_RADPS": 0.25,
    # X/Y 调参时的航向协同（方案 A+B）。关闭后退回旧行为：只调主轴、不修正 YAW。
    "HARDWARE_YAW_COPILOT"          : True,
    "HARDWARE_YAW_SOFT_BUDGET_DEG"  : 5.0,
    "HARDWARE_YAW_VERIFY_LIMIT_DEG" : 8.0,
    "HARDWARE_YAW_HOLD_LIMIT_RADPS" : 0.15,
    "HARDWARE_YAW_HOLD_SAT_RATIO_MAX": 0.5,
    "HARDWARE_YAW_ADJUST_SAT_RATIO" : 0.25,
    "HARDWARE_POST_MOTION_TESTS"    : True,
    "HARDWARE_TEST_LINEAR_MPS"      : 0.03,
    "HARDWARE_TEST_TURN_RADPS"      : 0.15,
    "HARDWARE_TEST_DURATION_MS"     : 600,
    "PID_RESULT_LOG"                : "logs/pid_results.jsonl",
    "MATLAB_MODEL_PATH"             : "",
    "MATLAB_PID_BLOCK_PATH"         : "",
    "MATLAB_ROOT"                   : "",
    "MATLAB_OUTPUT_SIGNAL"          : "y_out",
    "MATLAB_SIM_STEP_TIME"          : 15.0,
    "MATLAB_SETPOINT"               : 200.0,
}

CONFIG: dict[str, Any] = dict(DEFAULT_CONFIG)
PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_DIR / "config.json"
PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")


def resolve_project_path(value: str | os.PathLike[str]) -> Path:
    """Resolve tool-owned relative paths against llm-pid-tuner-main, never cwd."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


def _parse_env_value(default_value: Any, raw_value: str) -> Any:
    if isinstance(default_value, bool):
        return raw_value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(raw_value)
    if isinstance(default_value, float):
        return float(raw_value)
    return raw_value


def load_config(create_if_missing: bool = True, verbose: bool = True) -> None:
    """Load config from disk and environment variables."""
    CONFIG.clear()
    CONFIG.update(DEFAULT_CONFIG)

    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as handle:
                user_config = json.load(handle)
            CONFIG.update(user_config)
            if verbose:
                print(f"[INFO] 已加载配置文件: {CONFIG_PATH}")
        except Exception as exc:
            if verbose:
                print(f"[WARN] 配置文件加载失败: {exc}，将使用默认值。")
    elif create_if_missing:
        try:
            with CONFIG_PATH.open("w", encoding="utf-8") as handle:
                json.dump(CONFIG, handle, indent=4, ensure_ascii=False)
            if verbose:
                print(f"[INFO] 未找到配置文件，已生成默认配置: {CONFIG_PATH}")
                print(f"[HINT] 请打开 {CONFIG_PATH} 修改您的 API Key 和串口设置。")
        except Exception as exc:
            if verbose:
                print(f"[WARN] 无法创建配置文件: {exc}")

    for key in list(CONFIG):
        env_val = os.getenv(key)
        if not env_val:
            continue
        try:
            CONFIG[key] = _parse_env_value(CONFIG[key], env_val)
        except Exception:
            if verbose:
                print(f"[WARN] 环境变量 {key} 值无效，已忽略。")


def _apply_proxy_env_from_config() -> None:
    """Populate proxy env vars from config when the environment is unset."""
    for key in PROXY_KEYS:
        raw_value = CONFIG.get(key)
        if raw_value is None:
            continue
        if not isinstance(raw_value, str):
            if CONFIG.get("LLM_DEBUG_OUTPUT"):
                print(
                    f"[WARN] 代理配置 {key} 应为字符串，当前类型为 "
                    f"{type(raw_value).__name__}，已忽略。"
                )
            continue
        value = raw_value.strip()
        if not value:
            continue
        if not os.getenv(key):
            os.environ[key] = value
        lower_key = key.lower()
        if not os.getenv(lower_key):
            os.environ[lower_key] = value


def initialize_runtime_config(
    create_if_missing: bool = True, verbose: bool = True
) -> None:
    """Initialize runtime config and proxy environment variables."""
    ensure_utf8_console()
    load_config(create_if_missing=create_if_missing, verbose=verbose)
    _apply_proxy_env_from_config()
