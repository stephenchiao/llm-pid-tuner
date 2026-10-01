#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
tuner.py - LLM PID 自动调参系统 (History-Aware + Chain-of-Thought)
===============================================================================

作者: KINGSTON-115, ApexGP

依赖：pyserial, openai (或 requests), numpy (可选，用于高级计算)
"""

from __future__ import annotations

import argparse
import math
from queue import Queue
import re
import sys
import time
import traceback
from typing import Any, Callable

from core.buffer import AdvancedDataBuffer
from core.config import CONFIG, initialize_runtime_config, resolve_project_path
from core.pid_results import (
    append_pid_result,
    build_local_tuning_summary,
    load_last_usable_pid,
)
from core.round_csv import HardwareRoundCsvRecorder
from core.tuning_session import (
    apply_rollback,
    build_tuning_result,
    create_tuning_session,
    evaluate_completed_round,
    finalize_decision,
    record_rollback_round,
)
from hw.bridge import SerialBridge, safe_pause, select_serial_port, _is_demo_port
from hw.session import can_start_round
from hw.diagnostics import SerialTranscript, capture_failure, fields
from llm.client import LLMTuner
from pid_safety import build_fallback_suggestion, get_pid_limits, pid_equals
from sim.runtime import (
    EVENT_DECISION,
    EVENT_LIFECYCLE,
    EVENT_LOG,
    EVENT_ROLLBACK,
    EVENT_ROUND_METRICS,
    EVENT_SAMPLE,
    QueueEventSink,
    SimulationController,
    now_elapsed,
    publish_event,
)

HARDWARE_WRONG_DIRECTION_MM = 20.0
HARDWARE_MAX_YAW_ERROR_DEG = 15.0
HARDWARE_TUNING_STAGES = ("P", "I", "D", "VERIFY")
OPS_CENTER_OFFSET_X_MM = 0.0
OPS_CENTER_OFFSET_Y_MM = 25.0
_PID_NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_PID_ALL_PATTERN = re.compile(
    rf"^# PID ALL X=(?P<xp>{_PID_NUMBER_PATTERN}),(?P<xi>{_PID_NUMBER_PATTERN}),(?P<xd>{_PID_NUMBER_PATTERN}) "
    rf"Y=(?P<yp>{_PID_NUMBER_PATTERN}),(?P<yi>{_PID_NUMBER_PATTERN}),(?P<yd>{_PID_NUMBER_PATTERN}) "
    rf"YAW=(?P<yawp>{_PID_NUMBER_PATTERN}),(?P<yawi>{_PID_NUMBER_PATTERN}),(?P<yawd>{_PID_NUMBER_PATTERN})$"
)


def _normalize_hardware_axis(value: Any) -> str:
    axis = str(value or "Y").strip().upper()
    return axis if axis in {"X", "Y", "YAW"} else "Y"


def _configured_initial_pid(tune_axis: str) -> dict[str, float]:
    defaults = {
        "X": {"p": 0.00495, "i": 0.0, "d": 0.0},
        "Y": {"p": 0.0018, "i": 0.0, "d": 0.0},
        "YAW": {"p": 0.02, "i": 0.000015, "d": 0.0},
    }
    configured = CONFIG.get(f"HARDWARE_INITIAL_PID_{tune_axis}", defaults[tune_axis])
    if not isinstance(configured, dict):
        configured = defaults[tune_axis]
    return {
        key: float(configured.get(key, defaults[tune_axis][key]))
        for key in ("p", "i", "d")
    }


def _parse_pid_snapshot(line: str) -> dict[str, dict[str, float]] | None:
    match = _PID_ALL_PATTERN.match(str(line or "").strip())
    if match is None:
        return None
    return {
        "X": {
            "p": float(match.group("xp")),
            "i": float(match.group("xi")),
            "d": float(match.group("xd")),
        },
        "Y": {
            "p": float(match.group("yp")),
            "i": float(match.group("yi")),
            "d": float(match.group("yd")),
        },
        "YAW": {
            "p": float(match.group("yawp")),
            "i": float(match.group("yawi")),
            "d": float(match.group("yawd")),
        },
    }


def _read_pid_snapshot(bridge: Any, timeout_sec: float = 0.5) -> dict[str, dict[str, float]]:
    """停车后读取STM32实际保存的三轴参数；失败时由调用方补当前调试轴。"""
    bridge.send_command("PID STATUS ALL")
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        line = bridge.read_line()
        snapshot = _parse_pid_snapshot(line or "")
        if snapshot is not None:
            return snapshot
        if not line:
            time.sleep(0.01)
    return {}


def _angle_error_deg(current_deg: float, reference_deg: float) -> float:
    return (current_deg - reference_deg + 180.0) % 360.0 - 180.0


def _ops_to_chassis_center(
    ops_x_mm: float, ops_y_mm: float, yaw_deg: float
) -> tuple[float, float]:
    """把 OPS9 安装点坐标换算成小车几何中心坐标。"""
    yaw_rad = math.radians(yaw_deg)
    center_x = ops_x_mm - (
        math.cos(yaw_rad) * OPS_CENTER_OFFSET_X_MM
        - math.sin(yaw_rad) * OPS_CENTER_OFFSET_Y_MM
    )
    center_y = ops_y_mm - (
        math.sin(yaw_rad) * OPS_CENTER_OFFSET_X_MM
        + math.cos(yaw_rad) * OPS_CENTER_OFFSET_Y_MM
    )
    return center_x, center_y


def _hardware_sample_safety_reason(
    data: dict[str, Any], round_start_yaw: float | None, tune_axis: str = "Y"
) -> str | None:
    """PC 侧冗余急停；STM32 仍是最终安全边界。"""
    normalized_input = float(data.get("input", 0.0))
    wrong_direction_limit = 3.0 if tune_axis == "YAW" else HARDWARE_WRONG_DIRECTION_MM
    if normalized_input < -wrong_direction_limit:
        unit = "deg" if tune_axis == "YAW" else "mm"
        return f"WRONG DIR input={normalized_input:.1f}{unit}"

    yaw = data.get("yaw")
    if tune_axis != "YAW" and yaw is not None and round_start_yaw is not None:
        yaw_error = _angle_error_deg(float(yaw), round_start_yaw)
        if abs(yaw_error) > HARDWARE_MAX_YAW_ERROR_DEG:
            return f"YAW LIMIT error={yaw_error:.1f}deg"
    return None


def _augment_hardware_round_metrics(
    metrics: dict[str, Any],
    samples: list[dict[str, Any]],
    *,
    output_limit: float,
    stop_reason: str,
) -> None:
    """补充底盘专用指标；旧固件没有扩展列时保持向后兼容。"""
    if not samples:
        return

    outputs = [abs(float(sample.get("pwm", 0.0))) for sample in samples]
    cross_values = [
        float(sample["cross_track"])
        for sample in samples
        if sample.get("cross_track") is not None
    ]
    yaw_values = [
        float(sample["yaw_delta"])
        for sample in samples
        if sample.get("yaw_delta") is not None
    ]
    metrics.update(
        {
            "main_axis_final_error": float(metrics.get("current_error", 0.0)),
            "main_axis_overshoot": float(metrics.get("overshoot", 0.0)),
            "speed_saturation_ratio": (
                sum(value >= output_limit * 0.99 for value in outputs) / len(outputs)
            ),
            "round_stop_reason": stop_reason,
            # 单轮调参结束只代表到达测试目标，回到统一起点需要由独立 POSE SET 流程确认。
            "return_to_start_checked": False,
        }
    )
    if cross_values:
        metrics["cross_track_final_mm"] = cross_values[-1]
        metrics["cross_track_peak_mm"] = max(abs(value) for value in cross_values)
    if yaw_values:
        metrics["yaw_delta_final_deg"] = yaw_values[-1]
        metrics["yaw_delta_peak_deg"] = max(abs(value) for value in yaw_values)
    hold_values = [
        abs(float(sample["hold_yaw_output"]))
        for sample in samples
        if sample.get("hold_yaw_output") is not None
    ]
    hold_limit = float(CONFIG.get("HARDWARE_YAW_HOLD_LIMIT_RADPS", 0.15) or 0.15)
    if hold_values and hold_limit > 0.0:
        metrics["hold_yaw_saturated_ratio"] = sum(
            value >= hold_limit * 0.99 for value in hold_values
        ) / len(hold_values)


def _build_hardware_prompt_context(
    serial_port: str,
    tuning_stage: str = "P",
    output_limit: float | None = None,
    tune_axis: str | None = None,
) -> dict[str, Any]:
    axis = _normalize_hardware_axis(tune_axis or CONFIG.get("HARDWARE_TUNE_AXIS", "Y"))
    is_yaw = axis == "YAW"
    limits = get_pid_limits("hardware_yaw" if is_yaw else "hardware")
    stage_terms = {
        "P": ("P", "I,D"),
        "I": ("I", "P,D"),
        "D": ("D", "P,I"),
        "VERIFY": ("none", "P,I,D"),
    }
    adjustable, frozen = stage_terms.get(tuning_stage, stage_terms["P"])
    return {
        "source": "serial_hardware",
        "serial_port": serial_port,
        "tune_axis": axis,
        "controller_input_unit": "degree" if is_yaw else "mm",
        "controller_output_signal": "yaw_rate_radps" if is_yaw else "chassis_velocity_mps",
        "controller_output_unit": "rad/s" if is_yaw else "m/s",
        "controller_output_limit": (
            float(CONFIG["HARDWARE_YAW_OUTPUT_LIMIT_RADPS"] if is_yaw else CONFIG["HARDWARE_OUTPUT_LIMIT_MPS"])
            if output_limit is None else output_limit
        ),
        "target_value": 30.0 if is_yaw else 200.0,
        "target_tolerance": 1.0 if is_yaw else 5.0,
        "pwm_signal_available": False,
        "tuning_style": "conservative_hardware_safe",
        "tuning_stage": tuning_stage,
        "adjustable_terms": adjustable,
        "frozen_terms": frozen,
        "done_meaning": "Current stage is complete; the host advances to the next stage.",
        "pid_limits": limits,
        "per_round_guardrail_hint": "P/I/D increases are limited to 1.5x per round and the absolute pid_limits. Do not repeat a clipped proposal.",
        "acceptance_goal": "Reach TARGET within tolerance with acceptable overshoot and yaw. Once accepted, hold gains for verification; do not optimize speed indefinitely.",
        "yaw_hold_limit_radps": float(CONFIG.get("HARDWARE_YAW_HOLD_LIMIT_RADPS", 0.15) or 0.15),
        "yaw_peak_budget_deg": float(CONFIG.get("HARDWARE_YAW_SOFT_BUDGET_DEG", 5.0) or 5.0),
        "yaw_verify_limit_deg": float(CONFIG.get("HARDWARE_YAW_VERIFY_LIMIT_DEG", 8.0) or 8.0),
        "yaw_copilot_enabled": bool(CONFIG.get("HARDWARE_YAW_COPILOT", True)) and not is_yaw,
        "yaw_adjustable_hint": (
            "Optional yaw_p/yaw_i/yaw_d may refine the independent YAW hold loop; "
            "change yaw_p by at most ~1.2x per round and never raise main-axis P to fight yaw."
            if (bool(CONFIG.get("HARDWARE_YAW_COPILOT", True)) and not is_yaw)
            else "YAW hold parameters are frozen for this session."
        ),
    }


def _freeze_pid_terms_for_stage(
    result: dict[str, Any], current_pid: dict[str, float], tuning_stage: str
) -> dict[str, Any]:
    """每个阶段只开放一个参数，避免后续阶段破坏已经确定的参数。"""
    staged = dict(result)
    if tuning_stage == "P":
        staged["i"] = current_pid["i"]
        staged["d"] = current_pid["d"]
    elif tuning_stage == "I":
        staged["p"] = current_pid["p"]
        staged["d"] = current_pid["d"]
    elif tuning_stage == "D":
        staged["p"] = current_pid["p"]
        staged["i"] = current_pid["i"]
    return staged


def _hardware_validation_passed(
    metrics: dict[str, Any], stop_reason: str, tune_axis: str = "Y"
) -> bool:
    """最终验证使用确定性门槛，不让 LLM 单独决定整个流程结束。"""
    axis = str(tune_axis or "Y").upper()
    base_ok = (
        stop_reason == "TARGET"
        and float(metrics.get("overshoot", float("inf")))
        <= float(CONFIG["GOOD_ENOUGH_OVERSHOOT"])
        # TARGET 已表示固件连续10个周期进入 +/-5 mm；最后20%平均值包含减速段，
        # 不再用它否决已经到位的轮次，避免在 I/D/VERIFY 之间无限循环。
        and float(metrics.get("current_error", float("inf")))
        <= (1.0 if axis == "YAW" else 5.0)
        and int(metrics.get("zero_crossings", 0)) <= 6
    )
    if not base_ok or axis == "YAW" or not bool(CONFIG.get("HARDWARE_YAW_COPILOT", True)):
        return base_ok

    yaw_peak = abs(float(metrics.get("yaw_delta_peak_deg", 0.0) or 0.0))
    yaw_limit = float(CONFIG.get("HARDWARE_YAW_VERIFY_LIMIT_DEG", 8.0) or 8.0)
    if yaw_peak > yaw_limit:
        return False
    sat_ratio = float(metrics.get("hold_yaw_saturated_ratio", 0.0) or 0.0)
    sat_max = float(CONFIG.get("HARDWARE_YAW_HOLD_SAT_RATIO_MAX", 0.5) or 0.5)
    return sat_ratio <= sat_max


def _yaw_adjust_allowed(metrics: dict[str, Any], tune_axis: str, tuning_stage: str) -> bool:
    """策略 b：仅当偏航超预算或 hold 明显饱和时，才允许 LLM 微调 YAW hold。"""
    if not bool(CONFIG.get("HARDWARE_YAW_COPILOT", True)):
        return False
    if str(tune_axis or "").upper() not in {"X", "Y"}:
        return False
    if tuning_stage == "VERIFY":
        return False
    yaw_peak = abs(float(metrics.get("yaw_delta_peak_deg", 0.0) or 0.0))
    budget = float(CONFIG.get("HARDWARE_YAW_SOFT_BUDGET_DEG", 5.0) or 5.0)
    sat_ratio = float(metrics.get("hold_yaw_saturated_ratio", 0.0) or 0.0)
    sat_trigger = float(CONFIG.get("HARDWARE_YAW_ADJUST_SAT_RATIO", 0.25) or 0.25)
    return yaw_peak > budget or sat_ratio >= sat_trigger


def _evaluate_hardware_metrics(
    buffer: AdvancedDataBuffer, tune_axis: str, output_limit: float, stop_reason: str,
) -> dict[str, Any]:
    """Apply the same acceptance rules before scoring, rollback and verification."""
    metrics = buffer.calculate_advanced_metrics(tune_axis=tune_axis)
    samples = list(buffer.buffer)
    _augment_hardware_round_metrics(
        metrics, samples, output_limit=output_limit, stop_reason=stop_reason,
    )
    tolerance = 1.0 if tune_axis == "YAW" else 5.0
    metrics["first_in_tolerance_ms"] = next(
        (sample.get("timestamp") for sample in samples
         if abs(float(sample["setpoint"]) - float(sample["input"])) <= tolerance),
        None,
    )
    metrics["last_sample_ms"] = samples[-1].get("timestamp") if samples else None
    metrics["response_status"] = metrics.get("status", "UNKNOWN")
    metrics["hardware_accepted"] = _hardware_validation_passed(metrics, stop_reason, tune_axis)
    if metrics["hardware_accepted"]:
        metrics["status"] = "STABLE"
    elif metrics.get("status") == "STABLE":
        metrics["status"] = "CONSTRAINT_VIOLATION"
    return metrics


class RoundAdmissionClosed(Exception):
    pass


def _confirm_hardware_stop(bridge):
    if not getattr(bridge, "requires_hardware_preflight", False) or getattr(bridge, "is_demo", False):
        bridge.send_command("STOP")
        return "command_sent"
    # Read first: an already sent stop does not need another CAN abort batch.
    try:
        line = bridge.request("MOTOR STOP STATUS", lambda text: text.startswith("# MOTOR STOP STATE="), timeout=0.3)
        status = fields(line)
    except (ConnectionError, RuntimeError, TimeoutError):
        # A failed diagnostic must never prevent the actual stop command.
        status = {}
    if status.get("STATE") == "SENT" and status.get("EVIDENCE") == "CAN_TX_ONLY":
        return "can_stop_sent"
    bridge.request("STOP", lambda text: text.startswith(("# STOP MODE=", "# ROUND STOP HOST")))
    bridge.wait_stopped()
    return "can_stop_sent"


def _pause_hardware(bridge, controller):
    _confirm_hardware_stop(bridge)
    # STOP transmission closes the old round; account for buffered events before
    # requesting a fresh round. No reset_input_buffer and no blind wire discard.
    while getattr(bridge, "pending_lines", None):
        line = bridge.read_line()
        if line and line.startswith(("# ERROR", "# CAN SAFETY", "# MOTION STOP SAFETY")):
            raise RuntimeError(line)
    while controller.is_paused and not controller.should_stop:
        line = bridge.read_line()
        if line and line.startswith(("# ERROR", "# CAN SAFETY", "# MOTION STOP SAFETY")):
            raise RuntimeError(line)
        if not line:
            time.sleep(0.01)


def _send_next_round_commands(
    bridge: Any,
    safe_pid: dict[str, float],
    safe_yaw_pid: dict[str, float] | None,
    previous_yaw_pid: dict[str, float] | None,
) -> list[str]:
    """先装载 YAW hold（若有变化），再用 SET P 启动下一轮主轴测试。"""
    sent: list[str] = []
    if safe_yaw_pid is not None:
        changed = True
        if previous_yaw_pid is not None:
            changed = any(
                abs(float(safe_yaw_pid.get(key, 0.0)) - float(previous_yaw_pid.get(key, 0.0))) > 1e-12
                for key in ("p", "i", "d")
            )
        if changed:
            yaw_cmd = (
                f"PID SET YAW {safe_yaw_pid['p']} {safe_yaw_pid['i']} {safe_yaw_pid['d']}"
            )
            bridge.send_command(yaw_cmd)
            sent.append(yaw_cmd)
    cmd = (
        f"SET P:{safe_pid['p']} "
        f"I:{safe_pid['i']} D:{safe_pid['d']}"
    )
    bridge.send_command(cmd)
    sent.append(cmd)
    return sent


_OPS_STATUS_RE = re.compile(
    r"# OPS LINK=(?P<link>\S+)\s+X=(?P<x>-?\d+(?:\.\d+)?)\s+"
    r"Y=(?P<y>-?\d+(?:\.\d+)?)\s+YAW=(?P<yaw>-?\d+(?:\.\d+)?)"
    r"(?:\s+CENTER_X=(?P<center_x>-?\d+(?:\.\d+)?)"
    r"\s+CENTER_Y=(?P<center_y>-?\d+(?:\.\d+)?))?"
)


def _read_ops_pose(bridge: Any, timeout_sec: float = 1.5) -> dict[str, float] | None:
    """请求一帧OPS状态；只读位姿，不占用第二个串口连接。"""
    bridge.send_command("OPS STATUS")
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        line = bridge.read_line()
        if not line:
            continue
        if line.startswith(("# ERROR", "# CAN SAFETY", "# MOTION STOP SAFETY")):
            raise RuntimeError(line)
        match = _OPS_STATUS_RE.search(line)
        if match and match.group("link") == "OK":
            x = float(match.group("x"))
            y = float(match.group("y"))
            yaw = float(match.group("yaw"))
            if match.group("center_x") is not None:
                center_x = float(match.group("center_x"))
                center_y = float(match.group("center_y"))
            else:
                center_x, center_y = _ops_to_chassis_center(x, y, yaw)
            return {
                "x": x,
                "y": y,
                "yaw": yaw,
                "center_x": center_x,
                "center_y": center_y,
            }
    return None


def _wait_for_ops_ready(bridge: Any, timeout_sec: float = 3.0) -> dict[str, float] | None:
    """Wait through serial-open/reset transients before any tuning command."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        pose = _read_ops_pose(bridge, timeout_sec=min(0.75, max(0.05, remaining)))
        if pose is not None:
            return pose
        time.sleep(0.05)
    return None


def _wait_for_chassis_auto_stop(bridge: Any, timeout_sec: float, controller=None) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if controller is not None and (controller.should_stop or controller.is_paused):
            bridge.send_command("STOP")
            return False
        line = bridge.read_line()
        if line and line.startswith(("# ERROR", "# MOTION STOP SAFETY", "# CAN SAFETY")):
            bridge.send_command("STOP")
            return False
        if line and line.startswith("# MOVE AUTO STOP"):
            return True
    bridge.send_command("MOVE STOP")
    return False


def _run_post_tune_motion_tests(
    bridge: Any, *, emit_console: bool = True, controller=None
) -> list[dict[str, Any]]:
    """用低速短动作验收横移和旋转；这里只检查运动学，不修改PID。"""
    linear_speed = min(0.08, max(0.005, float(CONFIG["HARDWARE_TEST_LINEAR_MPS"])))
    turn_speed = min(0.30, max(0.03, float(CONFIG["HARDWARE_TEST_TURN_RADPS"])))
    duration_ms = min(3000, max(200, int(CONFIG["HARDWARE_TEST_DURATION_MS"])))
    actions = [
        ("LEFT", f"MOVE LEFT {linear_speed:.3f} {duration_ms}", "linear", -1.0),
        ("RIGHT", f"MOVE RIGHT {linear_speed:.3f} {duration_ms}", "linear", 1.0),
        ("CCW", f"TURN CCW {turn_speed:.3f} {duration_ms}", "turn", 1.0),
        ("CW", f"TURN CW {turn_speed:.3f} {duration_ms}", "turn", -1.0),
    ]
    results: list[dict[str, Any]] = []
    _console(emit_console, "[MotionTest] 开始低速验收：LEFT、RIGHT、CCW、CW")

    for name, command, kind, expected_sign in actions:
        if controller is not None and (controller.should_stop or controller.is_paused):
            _confirm_hardware_stop(bridge)
            results.append({"action": name, "passed": False, "reason": "USER_INTERRUPTED"})
            break
        before = _read_ops_pose(bridge)
        if before is None:
            results.append({"action": name, "passed": False, "reason": "OPS_NOT_READY"})
            _console(emit_console, f"[MotionTest] {name} FAIL: OPS_NOT_READY")
            break

        if controller is not None and (controller.should_stop or controller.is_paused):
            _confirm_hardware_stop(bridge)
            results.append({"action": name, "passed": False, "reason": "USER_INTERRUPTED"})
            break
        bridge.send_command(command)
        stopped = _wait_for_chassis_auto_stop(
            bridge, duration_ms / 1000.0 + 1.5, controller
        )
        if not stopped:
            results.append({"action": name, "passed": False, "reason": "STOP_NOT_CONFIRMED"})
            break
        _confirm_hardware_stop(bridge)
        after = _read_ops_pose(bridge)
        if after is None:
            results.append({"action": name, "passed": False, "reason": "OPS_NOT_READY_AFTER"})
            _console(emit_console, f"[MotionTest] {name} FAIL: OPS_NOT_READY_AFTER")
            break

        raw_dx = after["x"] - before["x"]
        raw_dy = after["y"] - before["y"]
        dx = after["center_x"] - before["center_x"]
        dy = after["center_y"] - before["center_y"]
        dyaw = _angle_error_deg(after["yaw"], before["yaw"])
        heading = math.radians(before["yaw"])
        body_right_mm = math.cos(heading) * dx + math.sin(heading) * dy
        translation_mm = math.hypot(dx, dy)

        if kind == "linear":
            passed = stopped and expected_sign * body_right_mm >= 5.0 and abs(dyaw) <= 5.0
            reason = "OK" if passed else "DIRECTION_DISTANCE_OR_YAW"
        else:
            passed = stopped and expected_sign * dyaw >= 1.5 and translation_mm <= 25.0
            reason = "OK" if passed else "DIRECTION_ANGLE_OR_TRANSLATION"

        record = {
            "action": name,
            "passed": passed,
            "reason": reason,
            "dx_mm": round(dx, 2),
            "dy_mm": round(dy, 2),
            "raw_ops_dx_mm": round(raw_dx, 2),
            "raw_ops_dy_mm": round(raw_dy, 2),
            "body_right_mm": round(body_right_mm, 2),
            "dyaw_deg": round(dyaw, 2),
            "translation_mm": round(translation_mm, 2),
        }
        results.append(record)
        _console(
            emit_console,
            f"[MotionTest] {name} {'PASS' if passed else 'FAIL'} "
            f"dX={dx:.1f} dY={dy:.1f} BodyRight={body_right_mm:.1f} dYaw={dyaw:.1f}",
        )
        if not passed:
            break

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the hardware PID tuner against a serial device."
    )
    parser.add_argument(
        "serial_port",
        nargs="?",
        help="Serial port to use, for example COM5.",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Disable the Textual dashboard and use plain console logs.",
    )
    parser.add_argument(
        "--axes",
        nargs="+",
        help="Tune X, Y or YAW in the supplied order, e.g. --axes X Y YAW.",
    )
    return parser


def parse_hardware_axes(value: str) -> list[str]:
    """Validate the entire selection before opening hardware; deduplicate in order."""
    tokens = re.split(r"[\s,/;，、；]+", value.strip().upper())
    axes: list[str] = []
    for axis in tokens:
        if not axis:
            continue
        if axis not in {"X", "Y", "YAW"}:
            raise ValueError(f"无效轴 {axis!r}，只能输入 X、Y、YAW。")
        if axis not in axes:
            axes.append(axis)
    if not axes:
        raise ValueError("请至少输入一个轴：X、Y 或 YAW。")
    return axes


def resolve_serial_port(serial_port_arg: str | None) -> str | None:
    if serial_port_arg:
        return serial_port_arg

    serial_port = CONFIG["SERIAL_PORT"]
    if serial_port and serial_port.upper() != "AUTO":
        print(f"[INFO] 使用配置端口: {serial_port}")
        use_env = input("是否使用该端口? (Y/n): ").strip().lower()
        if use_env != "n":
            return serial_port

    return select_serial_port()


def choose_tui_language(default: str = "zh") -> str:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return default

    print("Choose interface language / 选择界面语言")
    print("[1] 中文")
    print("[2] English")
    choice = input("Press Enter for 中文 / 回车默认中文: ").strip().lower()
    if choice in {"2", "en", "english"}:
        return "en"
    return default


def _console(enabled: bool, message: str, *, end: str = "\n") -> None:
    if enabled:
        print(message, end=end, flush=True)


def _emit_lifecycle(
    event_sink: QueueEventSink | None,
    start_time: float,
    phase: str,
    message: str,
) -> None:
    publish_event(
        event_sink,
        EVENT_LIFECYCLE,
        phase=phase,
        message=message,
        elapsed_sec=now_elapsed(start_time),
    )


def _emit_log(
    event_sink: QueueEventSink | None,
    start_time: float,
    label: str,
    message: str,
    *,
    replace_last: bool = False,
    stream_id: int | None = None,
) -> None:
    publish_event(
        event_sink,
        EVENT_LOG,
        label=label,
        message=message,
        replace_last=replace_last,
        stream_id=stream_id,
        elapsed_sec=now_elapsed(start_time),
    )


def _run_hardware_tuning_loop(
    serial_port: str,
    event_sink: QueueEventSink | None = None,
    controller: SimulationController | None = None,
    emit_console: bool = True,
    initial_pid: dict[str, float] | None = None,
    preserved_pids: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    tune_axis = _normalize_hardware_axis(CONFIG.get("HARDWARE_TUNE_AXIS", "Y"))
    is_yaw_axis = tune_axis == "YAW"
    if initial_pid is None:
        initial_pid = _configured_initial_pid(tune_axis)

    hardware_pid_limits = get_pid_limits("hardware_yaw" if is_yaw_axis else "hardware")
    hardware_output_limit = (
        min(0.80, max(0.02, float(CONFIG["HARDWARE_YAW_OUTPUT_LIMIT_RADPS"])))
        if is_yaw_axis
        else min(0.30, max(0.02, float(CONFIG["HARDWARE_OUTPUT_LIMIT_MPS"])))
    )
    hardware_steady_error_limit = (
        2.0 if is_yaw_axis else float(CONFIG["GOOD_ENOUGH_STEADY_STATE_ERROR"])
    )
    bridge = SerialBridge(serial_port, CONFIG["BAUD_RATE"], emit_console=False)
    is_demo_hardware = _is_demo_port(serial_port)
    # Five seconds at 20 ms is about 250 samples; keep the whole real round.
    hardware_buffer_size = (
        int(CONFIG["BUFFER_SIZE"])
        if is_demo_hardware
        else max(int(CONFIG["BUFFER_SIZE"]), 300)
    )
    session = create_tuning_session(
        initial_pid=initial_pid,
        buffer_size=hardware_buffer_size,
        initial_yaw_pid=_configured_initial_pid("YAW") if not is_yaw_axis else None,
    )
    start_time = time.time()
    round_csv = (
        None
        if is_demo_hardware
        else HardwareRoundCsvRecorder(
            resolve_project_path("logs/round_csv"), tune_axis
        )
    )
    current_stream_round = [0]

    def llm_log_callback(label: str, message: str) -> None:
        _emit_log(
            event_sink,
            start_time,
            label,
            message,
            stream_id=current_stream_round[0] or None,
        )

    def llm_stream_callback(text: str, done: bool) -> None:
        _emit_log(
            event_sink,
            start_time,
            "llm_stream",
            text,
            replace_last=True,
            stream_id=current_stream_round[0] or None,
        )

    tuner = LLMTuner(
        CONFIG["LLM_API_KEY"],
        CONFIG["LLM_API_BASE_URL"],
        CONFIG["LLM_MODEL_NAME"],
        CONFIG["LLM_PROVIDER"],
        stream_callback=llm_stream_callback,
        log_callback=llm_log_callback,
        emit_console=emit_console,
    )

    _emit_lifecycle(
        event_sink,
        start_time,
        "starting",
        f"Opening {serial_port} at {CONFIG['BAUD_RATE']} baud.",
    )

    transcript = None
    if not is_demo_hardware and hasattr(bridge, "on_io"):
        transcript = SerialTranscript(resolve_project_path("logs/can_sessions"))
        bridge.on_io = transcript.write
        bridge.on_line = lambda line: _emit_log(event_sink, start_time, "serial", line) if line.startswith("#") else None
        _console(emit_console, f"[SerialLog] {transcript.path}")
        _emit_log(event_sink, start_time, "serial_log", str(transcript.path))

    if not bridge.connect():
        message = f"串口连接或 CAN 准备失败 {serial_port}: {bridge.last_error or 'unknown error'}"
        if transcript is not None:
            transcript.close()
        session.completed_reason = "error"
        _console(emit_console, f"[ERROR] {message}")
        _emit_lifecycle(event_sink, start_time, "error", message)
        return {
            "elapsed_sec": now_elapsed(start_time),
            "tune_axis": tune_axis,
            "failure_detail": message,
            "serial_log_path": str(transcript.path) if transcript else None,
            **build_tuning_result(
                session,
                final_pid=dict(session.buffer.current_pid),
                stopped=False,
            ),
        }

    _console(emit_console, f"[INFO] 已连接到串口: {serial_port}")
    _emit_lifecycle(
        event_sink,
        start_time,
        "connected",
        f"Connected to {serial_port}.",
    )

    if hasattr(bridge, "on_line"):
        bridge.on_line = lambda line: _emit_log(event_sink, start_time, "serial", line) if line.startswith("#") else None

    motion_test_results: list[dict[str, Any]] = []
    pid_snapshot: dict[str, dict[str, float]] = {}
    tested_result = None
    loaded_pid = None
    stop_confirmation = "not_requested"
    failure_detail = ""
    failed_round = None
    unchanged_proposals = 0

    def start_next(pid, yaw=None, previous_yaw=None):
        if controller is not None and controller.is_paused:
            _pause_hardware(bridge, controller)
        if not can_start_round(session.round_num, CONFIG["MAX_TUNING_ROUNDS"], controller):
            raise RoundAdmissionClosed("stopped_by_user" if controller is not None and controller.should_stop else "max_rounds_reached")
        checkpoint = getattr(bridge, "checkpoint", None)
        if callable(checkpoint):
            checkpoint("before_round")
        if not can_start_round(session.round_num, CONFIG["MAX_TUNING_ROUNDS"], controller):
            raise RoundAdmissionClosed("stopped_by_user")
        return _send_next_round_commands(bridge, pid, yaw, previous_yaw)

    try:
        if (
            not is_demo_hardware
            and getattr(bridge, "requires_hardware_preflight", False)
        ):
            _console(emit_console, "[INFO] 等待 OPS-9 定位链路就绪...")
            ops_pose = _wait_for_ops_ready(bridge, timeout_sec=3.0)
            if ops_pose is None:
                session.completed_reason = "ops_not_ready"
                message = "OPS-9 在串口连接后 3 秒内未返回 LINK=OK，未启动调参"
                _console(emit_console, f"[ERROR] {message}")
                _emit_lifecycle(event_sink, start_time, "error", message)
                return {
                    "elapsed_sec": now_elapsed(start_time),
                    "tune_axis": tune_axis,
                    **build_tuning_result(
                        session,
                        final_pid=dict(session.buffer.current_pid),
                        stopped=False,
                    ),
                }
            _console(
                emit_console,
                f"[INFO] OPS-9 就绪: X={ops_pose['x']:.2f} "
                f"Y={ops_pose['y']:.2f} YAW={ops_pose['yaw']:.2f}",
            )
        bridge.send_command("PROTO VERSION")
        _emit_log(event_sink, start_time, "cmd", "PROTO VERSION")
        time.sleep(0.05)
        bridge.send_command("MODE TUNE")
        checkpoint = getattr(bridge, "checkpoint", None)
        if callable(checkpoint):
            checkpoint("after_mode_tune")
        _emit_log(event_sink, start_time, "cmd", "MODE TUNE")
        time.sleep(0.05)
        linear_limit = min(0.30, max(0.02, float(CONFIG["HARDWARE_OUTPUT_LIMIT_MPS"])))
        yaw_limit = min(
            0.80, max(0.02, float(CONFIG["HARDWARE_YAW_OUTPUT_LIMIT_RADPS"]))
        )
        for limit_axis, limit_value in (
            ("X", linear_limit),
            ("Y", linear_limit),
            ("YAW", yaw_limit),
        ):
            axis_limit_cmd = f"PID LIMIT {limit_axis} {limit_value:.3f}"
            bridge.send_command(axis_limit_cmd)
            _emit_log(event_sink, start_time, "cmd", axis_limit_cmd)
            time.sleep(0.02)

        # 三轴统一按续调策略装载；关闭续调时禁止任何旧日志参数静默回灌。
        result_log = str(
            resolve_project_path(CONFIG.get("PID_RESULT_LOG", "logs/pid_results.jsonl"))
        )
        for saved_axis in ("X", "Y", "YAW"):
            if saved_axis == tune_axis:
                continue
            # Earlier axes in this launch take precedence over history/config,
            # including when resume is disabled or saving the result log failed.
            saved_pid = (preserved_pids or {}).get(saved_axis)
            if saved_pid is None and bool(CONFIG.get("HARDWARE_RESUME_LAST_PID", False)):
                saved_mode = "hardware_yaw" if saved_axis == "YAW" else "hardware"
                saved_pid = load_last_usable_pid(
                    result_log, get_pid_limits(saved_mode), saved_axis
                )
            if saved_pid is None:
                saved_pid = _configured_initial_pid(saved_axis)
            load_cmd = (
                f"PID SET {saved_axis} {saved_pid['p']} "
                f"{saved_pid['i']} {saved_pid['d']}"
            )
            bridge.send_command(load_cmd)
            if saved_axis == "YAW":
                session.current_yaw_pid = dict(saved_pid)
            _emit_log(event_sink, start_time, "cmd", load_cmd)
            _console(emit_console, f"[CMD] Loaded {saved_axis}: {load_cmd}")
            time.sleep(0.03)

        axis_cmd = f"TUNE AXIS {tune_axis}"
        bridge.send_command(axis_cmd)
        _emit_log(event_sink, start_time, "cmd", axis_cmd)
        _console(emit_console, f"[CMD] Sent: {axis_cmd}")
        time.sleep(0.05)
        bridge.send_command("STATUS")
        _emit_log(event_sink, start_time, "cmd", "STATUS")
        _console(emit_console, "[CMD] Sent: STATUS")
        time.sleep(0.1)
        limit_cmd = f"TUNE LIMIT {hardware_output_limit:.3f}"
        bridge.send_command(limit_cmd)
        _emit_log(event_sink, start_time, "cmd", limit_cmd)
        _console(emit_console, f"[CMD] Sent: {limit_cmd}")
        time.sleep(0.05)
        if not is_demo_hardware and getattr(bridge, "requires_hardware_preflight", False):
            configured_snapshot = _read_pid_snapshot(bridge)
            if set(configured_snapshot) != {"X", "Y", "YAW"}:
                raise RuntimeError("missing PID configuration snapshot")
            if not is_yaw_axis:
                session.current_yaw_pid = dict(configured_snapshot["YAW"])
        if initial_pid:
            cmd = f"SET P:{initial_pid['p']} I:{initial_pid['i']} D:{initial_pid['d']}"
            start_next(initial_pid)
            _emit_log(event_sink, start_time, "cmd", cmd)
            _console(emit_console, f"[CMD] Initial PID: {cmd}")

        _console(emit_console, "[INFO] 开始采集数据...")
        _emit_lifecycle(
            event_sink,
            start_time,
            "collecting",
            f"Collecting data from {serial_port}.",
        )

        # Do not wait forever when the MCU rejects a round or stops producing CSV.
        data_timeout_sec = 4.0
        last_sample_at = time.monotonic()
        last_device_message = "no MCU status received"
        minimum_round_samples = min(20, max(3, int(CONFIG["BUFFER_SIZE"])))
        maximum_round_duration_sec = 7.0
        round_active = is_demo_hardware
        round_requested = not is_demo_hardware
        round_complete = False
        round_sample_count = 0
        round_pid = None
        round_started_at = time.monotonic()
        round_start_yaw: float | None = None
        round_stop_reason = "UNKNOWN"
        last_timestamp = None
        tuning_stage = "P"
        stage_rounds = 0
        verification_passes = 0
        required_verification_rounds = max(
            1, int(CONFIG.get("HARDWARE_VERIFY_ROUNDS", 3))
        )
        max_stage_rounds = max(1, int(CONFIG.get("HARDWARE_STAGE_MAX_ROUNDS", 5)))
        _console(emit_console, "[Stage] 进入 P 阶段：只调整 P，冻结 I/D")

        while session.round_num < CONFIG["MAX_TUNING_ROUNDS"]:
            if controller is not None and controller.should_stop:
                session.completed_reason = "stopped_by_user"
                _console(emit_console, "\n[INFO] 用户停止")
                _emit_lifecycle(event_sink, start_time, "stopped", "Hardware tuning stopped by user.")
                break

            if controller is not None and controller.is_paused:
                if round_csv is not None:
                    round_csv.close()
                _pause_hardware(bridge, controller)
                session.buffer.reset()
                start_next(session.buffer.current_pid, session.current_yaw_pid, session.current_yaw_pid)
                round_active = is_demo_hardware
                round_requested = not is_demo_hardware
                round_complete = False
                round_sample_count = 0
                round_pid = None
                round_started_at = last_sample_at = time.monotonic()
                last_timestamp = None
                continue

            line = bridge.read_line()
            if line:
                if line.startswith("#"):
                    last_device_message = line
                    # "# CAN" 覆盖 # CAN SAFETY（CAN 故障的具体判据）和
                    # # CAN RECOVERED（自愈频率），两者是诊断 ROUND STOP
                    # CAN FAULT 的唯一线索，不能过滤掉。
                    important_status = line.startswith(
                        ("# STATUS", "# PID", "# ROUND", "# ERROR", "# OPS", "# CAN")
                    )
                    if important_status:
                        _console(emit_console, f"[MCU] {line}")
                        _emit_log(event_sink, start_time, "mcu", line)

                    if line.startswith("# ROUND START"):
                        if not round_requested or round_active:
                            raise RuntimeError("unexpected or duplicate ROUND START")
                        axis_match = re.search(r"AXIS=(\w+)", line)
                        if axis_match and axis_match.group(1) != tune_axis:
                            raise RuntimeError("ROUND START axis mismatch")
                        session.buffer.reset()
                        last_timestamp = None
                        if round_csv is not None:
                            csv_path = round_csv.start()
                            _console(emit_console, f"[CSV] 本轮原始数据: {csv_path}")
                            _emit_log(event_sink, start_time, "csv", str(csv_path))
                        round_requested = False
                        round_active = True
                        round_complete = False
                        round_sample_count = 0
                        round_pid = None
                        round_started_at = time.monotonic()
                        last_sample_at = round_started_at
                        round_start_yaw = None
                        round_stop_reason = "UNKNOWN"
                    elif line.startswith("# ROUND STOP") and (
                        round_active or round_requested
                    ):
                        stop_payload = line.removeprefix("# ROUND STOP").strip()
                        stop_reason = stop_payload.split(maxsplit=1)[0] if stop_payload else "UNKNOWN"
                        round_stop_reason = stop_reason
                        if round_csv is not None:
                            round_csv.close()
                        round_requested = False
                        round_active = False
                        if stop_reason in {"TARGET", "TIMEOUT"}:
                            if round_sample_count >= minimum_round_samples:
                                round_complete = True
                                _console(
                                    emit_console,
                                    f"[INFO] 本轮结束：{stop_reason}，"
                                    f"共 {round_sample_count} 个样本",
                                )
                            else:
                                session.completed_reason = "insufficient_data"
                                _console(
                                    emit_console,
                                    f"[ERROR] 本轮仅收到 {round_sample_count} 个样本，"
                                    "不足以分析",
                                )
                                break
                        else:
                            session.completed_reason = "hardware_stopped"
                            failure_detail = line
                            failed_round = {
                                "round": session.round_num + 1,
                                "stop_reason": stop_payload,
                                "pid": dict(session.buffer.current_pid),
                                "metrics": _evaluate_hardware_metrics(
                                    session.buffer, tune_axis, hardware_output_limit, stop_reason),
                            }
                            _console(
                                emit_console,
                                f"[ERROR] 主控异常停止本轮：{line}",
                            )
                            _emit_lifecycle(
                                event_sink,
                                start_time,
                                "error",
                                f"MCU stopped the tuning round: {line}",
                            )
                            break
                    elif line.startswith("# ERROR"):
                        session.completed_reason = "hardware_stopped"
                        failure_detail = line
                        _console(
                            emit_console,
                            f"[ERROR] 主控拒绝本轮：{line}",
                        )
                        _emit_lifecycle(
                            event_sink,
                            start_time,
                            "error",
                            f"MCU stopped the tuning round: {line}",
                        )
                        break

                data = None if line.startswith("#") else bridge.parse_data(line)
                if data:
                    if round_active:
                        last_sample_at = time.monotonic()
                        timestamp = float(data["timestamp"])
                        if last_timestamp is not None and timestamp <= last_timestamp:
                            raise RuntimeError("non-increasing round sample timestamp")
                        last_timestamp = timestamp
                        loaded_pid = {key: data[key] for key in ("p", "i", "d")}
                        if round_pid is not None and loaded_pid != round_pid:
                            raise RuntimeError("PID changed within the active round")
                        round_pid = dict(loaded_pid)
                        if round_csv is not None:
                            round_csv.append(line)
                        if round_start_yaw is None and data.get("yaw") is not None:
                            round_start_yaw = float(data["yaw"])
                        safety_reason = _hardware_sample_safety_reason(
                            data, round_start_yaw, tune_axis
                        )
                        if safety_reason:
                            bridge.send_command("STOP")
                            session.completed_reason = "hardware_diverging"
                            round_active = False
                            message = f"检测到运动发散，已急停：{safety_reason}"
                            _console(emit_console, f"[ERROR] {message}")
                            _emit_lifecycle(event_sink, start_time, "error", message)
                            break
                        session.buffer.add(data)
                        round_sample_count += 1
                    publish_event(
                        event_sink,
                        EVENT_SAMPLE,
                        timestamp=float(data.get("timestamp", 0.0)),
                        setpoint=float(data.get("setpoint", 0.0)),
                        input=float(data.get("input", 0.0)),
                        pwm=float(data.get("pwm", 0.0)),
                        error=float(data.get("error", 0.0)),
                        p=float(data.get("p", session.buffer.current_pid["p"])),
                        i=float(data.get("i", session.buffer.current_pid["i"])),
                        d=float(data.get("d", session.buffer.current_pid["d"])),
                    )
                    _console(
                        emit_console,
                        f"\r[DATA] T={data['input']:.1f} Err={data['error']:.1f} "
                        f"Out={data['pwm']:.3f}{'rad/s' if is_yaw_axis else 'm/s'}",
                        end="",
                    )

            if is_demo_hardware and session.buffer.is_full():
                round_active = False
                round_complete = True

            if round_requested and time.monotonic() - round_started_at >= 3.0:
                session.completed_reason = "start_timeout"
                break

            if (
                round_active
                and time.monotonic() - round_started_at >= maximum_round_duration_sec
            ):
                session.completed_reason = "round_timeout"
                bridge.send_command("STOP")
                message = "硬件一轮运行超过 7 秒但没有收到 ROUND STOP"
                _console(emit_console, f"[ERROR] {message}")
                _emit_lifecycle(event_sink, start_time, "error", message)
                break

            if (
                not round_complete
                and time.monotonic() - last_sample_at >= data_timeout_sec
            ):
                session.completed_reason = "data_timeout"
                message = (
                    f"连续 {data_timeout_sec:.0f} 秒未收到有效 CSV；"
                    f"主控最后消息：{last_device_message}"
                )
                _console(emit_console, f"[ERROR] {message}")
                _emit_lifecycle(event_sink, start_time, "error", message)
                break

            if not round_complete:
                continue

            if emit_console:
                print("\n\n" + "-" * 60)

            evaluation = evaluate_completed_round(
                session,
                dict(session.buffer.current_pid),
                tune_axis=tune_axis,
                current_yaw_pid=session.current_yaw_pid,
                round_metrics=_evaluate_hardware_metrics(
                    session.buffer, tune_axis, hardware_output_limit, round_stop_reason),
            )
            session.last_metrics.update(evaluation.metrics)
            tested_result = {"round": evaluation.round_index, "axis": tune_axis,
                             "pid": dict(evaluation.current_pid), "yaw_pid": dict(session.current_yaw_pid or {}),
                             "metrics": dict(evaluation.metrics),
                             "stop_reason": round_stop_reason, "verified": False}
            publish_event(
                event_sink,
                EVENT_ROUND_METRICS,
                round=evaluation.round_index,
                avg_error=float(evaluation.metrics["avg_error"]),
                max_error=float(evaluation.metrics["max_error"]),
                steady_state_error=float(evaluation.metrics["steady_state_error"]),
                overshoot=float(evaluation.metrics["overshoot"]),
                zero_crossings=int(evaluation.metrics["zero_crossings"]),
                status=str(evaluation.metrics["status"]),
                stable_rounds=evaluation.stable_rounds,
            )
            _console(
                emit_console,
                f"[第 {evaluation.round_index} 轮] 分析中... AvgErr={evaluation.metrics['avg_error']:.2f}, Status={evaluation.metrics['status']}",
            )

            if evaluation.best_result_updated and evaluation.best_result is not None:
                best_message = (
                    f"Round {evaluation.round_index} captured a new best PID: "
                    f"P={evaluation.best_result['pid']['p']}, I={evaluation.best_result['pid']['i']}, D={evaluation.best_result['pid']['d']}"
                )
                _console(
                    emit_console,
                    f"[Best] 更新最佳参数 -> "
                    f"P={evaluation.best_result['pid']['p']}, I={evaluation.best_result['pid']['i']}, D={evaluation.best_result['pid']['d']}",
                )
                _emit_log(event_sink, start_time, "best", best_message)

            if tuning_stage != "VERIFY" and evaluation.metrics["hardware_accepted"]:
                tuning_stage = "VERIFY"
                verification_passes = 0
                unchanged_proposals = 0
                _console(emit_console, "[Stage] 当前参数已满足到位约束，固定 PID 进入重复验证")

            if tuning_stage == "VERIFY":
                passed = _hardware_validation_passed(
                    evaluation.metrics, round_stop_reason, tune_axis
                )
                validation_text = (
                    f"最终验证 {'PASS' if passed else 'FAIL'} "
                    f"({verification_passes + 1}/{required_verification_rounds})"
                )
                session.history.add_record(
                    evaluation.round_index,
                    evaluation.current_pid,
                    evaluation.metrics,
                    validation_text,
                    "固定 P/I/D 进行重复性验证，不调用 LLM 修改参数。",
                )
                session.round_num += 1
                session.buffer.reset()
                publish_event(
                    event_sink,
                    EVENT_DECISION,
                    round=evaluation.round_index,
                    action="VERIFY_PASS" if passed else "VERIFY_FAIL",
                    analysis_summary=validation_text,
                    fallback_used=False,
                    guardrail_notes=[],
                )

                if passed:
                    verification_passes += 1
                    _console(
                        emit_console,
                        f"[Verify] PASS {verification_passes}/{required_verification_rounds}",
                    )
                    if verification_passes >= required_verification_rounds:
                        session.completed_reason = "staged_validation_passed"
                        tested_result["verified"] = True
                        _console(emit_console, "\n[SUCCESS] 当前 PID 已通过连续到位验证！")
                        _emit_lifecycle(
                            event_sink,
                            start_time,
                            "completed",
                            "PID passed consecutive hardware verification rounds.",
                        )
                        break
                else:
                    verification_passes = 0
                    if _yaw_adjust_allowed(evaluation.metrics, tune_axis, "P"):
                        tuning_stage = "P"
                        _console(emit_console, "[Stage] 验证未通过：返回 P 阶段处理航向约束")
                    elif float(evaluation.metrics.get("current_error", 0.0)) > (1.0 if is_yaw_axis else 5.0):
                        tuning_stage = "I"
                        _console(emit_console, "[Stage] 验证未通过：返回 I 阶段处理稳态误差")
                    else:
                        tuning_stage = "D"
                        _console(emit_console, "[Stage] 验证未通过：返回 D 阶段处理超调/振荡")
                    stage_rounds = 0

                if session.round_num >= int(CONFIG["MAX_TUNING_ROUNDS"]):
                    session.completed_reason = "max_rounds_reached"
                    break

                cmd_list = start_next(
                    evaluation.current_pid,
                    session.current_yaw_pid if not is_yaw_axis else None,
                    session.current_yaw_pid if not is_yaw_axis else None,
                )
                last_sample_at = time.monotonic()
                last_device_message = "waiting for verification round"
                round_active = is_demo_hardware
                round_requested = not is_demo_hardware
                round_complete = False
                round_sample_count = 0
                round_pid = None
                round_started_at = last_sample_at
                for cmd in cmd_list:
                    _emit_log(event_sink, start_time, "cmd", cmd)
                    _console(emit_console, f"[CMD] Sent: {cmd}")
                continue

            if evaluation.rollback_pid:
                rollback_message = (
                    f"当前表现劣于第 {evaluation.best_result['round']} 轮最佳结果，恢复到 "
                    f"P={evaluation.rollback_pid['p']}, I={evaluation.rollback_pid['i']}, D={evaluation.rollback_pid['d']}"
                )
                rollback_message = record_rollback_round(
                    session,
                    evaluation,
                    evaluation.rollback_pid,
                    target_round=int(evaluation.best_result["round"]) if evaluation.best_result else None,
                    rollback_yaw_pid=evaluation.rollback_yaw_pid,
                )
                _console(emit_console, f"[Rollback] {rollback_message}")
                publish_event(
                    event_sink,
                    EVENT_ROLLBACK,
                    round=evaluation.round_index,
                    target_round=int(evaluation.best_result["round"]) if evaluation.best_result else evaluation.round_index,
                    pid=dict(evaluation.rollback_pid),
                    reason=rollback_message,
                )

                previous_yaw = dict(session.current_yaw_pid or {})
                apply_rollback(session, evaluation.rollback_pid, evaluation.rollback_yaw_pid)
                time.sleep(0.05)
                cmd_list = start_next(
                    evaluation.rollback_pid,
                    evaluation.rollback_yaw_pid if not is_yaw_axis else None,
                    previous_yaw if not is_yaw_axis else None,
                )
                last_sample_at = time.monotonic()
                last_device_message = "waiting for rollback round"
                round_active = is_demo_hardware
                round_requested = not is_demo_hardware
                round_complete = False
                round_sample_count = 0
                round_pid = None
                round_started_at = last_sample_at
                for cmd in cmd_list:
                    _emit_log(event_sink, start_time, "cmd", cmd)
                    _console(emit_console, f"[CMD] Sent: {cmd}")
                stage_rounds += 1
                continue

            prompt_data = session.buffer.to_prompt_data(
                tune_axis=tune_axis,
                current_yaw_pid=session.current_yaw_pid,
                round_metrics=evaluation.metrics,
            )
            history_text = session.history.to_prompt_text()
            current_stream_round[0] = evaluation.round_index
            _emit_lifecycle(
                event_sink,
                start_time,
                "llm_request",
                f"Requesting PID suggestion for round {evaluation.round_index}.",
            )
            result = tuner.analyze(
                prompt_data,
                history_text,
                tuning_mode="hardware",
                prompt_context={
                    **_build_hardware_prompt_context(
                        serial_port, tuning_stage, hardware_output_limit, tune_axis),
                    "yaw_adjustment_allowed": _yaw_adjust_allowed(
                        evaluation.metrics, tune_axis, tuning_stage),
                },
            )

            if not result:
                _console(emit_console, "[WARN] LLM 本轮不可用，启用保守兜底策略。")
                _emit_lifecycle(
                    event_sink,
                    start_time,
                    "fallback",
                    f"LLM unavailable at round {evaluation.round_index}; using fallback rules.",
                )
                result = build_fallback_suggestion(
                    evaluation.current_pid,
                    evaluation.metrics,
                    limits=hardware_pid_limits,
                )

            result = _freeze_pid_terms_for_stage(
                result, evaluation.current_pid, tuning_stage
            )

            saturation_ratio = float(
                evaluation.metrics.get("speed_saturation_ratio", 0.0)
            )

            allow_yaw_edit = _yaw_adjust_allowed(
                evaluation.metrics, tune_axis, tuning_stage
            )
            yaw_blocks_increase = (
                allow_yaw_edit
                and float(result.get("p", evaluation.current_pid["p"])) > evaluation.current_pid["p"]
            )
            if yaw_blocks_increase:
                result["p"] = evaluation.current_pid["p"]
            previous_yaw = dict(session.current_yaw_pid or {})
            decision = finalize_decision(
                session,
                evaluation,
                result,
                limits=hardware_pid_limits,
                freeze_integral=(
                    tuning_stage == "I"
                    and saturation_ratio >= 0.5
                    and float(evaluation.metrics.get("steady_state_error", 0.0))
                    <= hardware_steady_error_limit
                ),
                current_yaw_pid=session.current_yaw_pid,
                allow_yaw_adjust=allow_yaw_edit,
            )
            if yaw_blocks_increase:
                if not decision.guardrail_notes:
                    session.guardrail_count += 1
                decision.guardrail_notes.append("偏航超预算或 hold 饱和，禁止提高主轴 P")
            publish_event(
                event_sink,
                EVENT_DECISION,
                round=evaluation.round_index,
                action=decision.action,
                analysis_summary=decision.analysis,
                fallback_used=decision.fallback_used,
                guardrail_notes=list(decision.guardrail_notes),
            )

            _console(
                emit_console,
                f"\n[Action] {decision.action} -> P={decision.safe_pid['p']}, I={decision.safe_pid['i']}, D={decision.safe_pid['d']}",
            )
            if decision.safe_yaw_pid and not is_yaw_axis:
                _console(
                    emit_console,
                    f"[YawHold] -> P={decision.safe_yaw_pid['p']}, "
                    f"I={decision.safe_yaw_pid['i']}, D={decision.safe_yaw_pid['d']}"
                    f" ({'allow' if allow_yaw_edit else 'frozen'})",
                )
            if decision.guardrail_notes:
                _console(emit_console, f"[Guardrail] {'; '.join(decision.guardrail_notes)}")
            if decision.fallback_used:
                _console(emit_console, "[Fallback] 本轮使用规则策略替代 LLM 建议。")

            unchanged = (pid_equals(decision.safe_pid, evaluation.current_pid)
                         and pid_equals(decision.safe_yaw_pid or {}, previous_yaw))
            unchanged_proposals = unchanged_proposals + 1 if unchanged and decision.status != "DONE" else 0
            if session.history.history:
                session.history.history[-1].update(
                    applied_pid=dict(decision.safe_pid),
                    applied_yaw_pid=dict(decision.safe_yaw_pid or {}),
                    guardrail_notes=list(decision.guardrail_notes),
                )
            if unchanged_proposals >= 2:
                session.completed_reason = "no_parameter_progress"
                _console(emit_console, "[STOP] 连续两轮建议经护栏处理后参数不变且未达标，停止无效重复调参")
                break

            stage_rounds += 1
            stage_finished = (
                decision.status == "DONE" or stage_rounds >= max_stage_rounds
            )
            if stage_finished:
                old_stage = tuning_stage
                next_index = HARDWARE_TUNING_STAGES.index(tuning_stage) + 1
                tuning_stage = HARDWARE_TUNING_STAGES[next_index]
                stage_rounds = 0
                reason = "LLM确认完成" if decision.status == "DONE" else "达到阶段轮数上限"
                _console(
                    emit_console,
                    f"[Stage] {old_stage} 阶段结束（{reason}），进入 {tuning_stage} 阶段",
                )

            safe_yaw = decision.safe_yaw_pid if not is_yaw_axis else None
            cmd_list = start_next(
                decision.safe_pid,
                safe_yaw,
                previous_yaw if not is_yaw_axis else None,
            )
            last_sample_at = time.monotonic()
            last_device_message = "waiting for next tuning round"
            round_active = is_demo_hardware
            round_requested = not is_demo_hardware
            round_complete = False
            round_sample_count = 0
            round_pid = None
            round_started_at = last_sample_at
            for cmd in cmd_list:
                _emit_log(event_sink, start_time, "cmd", cmd)
                _console(emit_console, f"[CMD] Sent: {cmd}")


        if (
            session.completed_reason == "staged_validation_passed"
            and tune_axis == "Y"
            and bool(CONFIG.get("HARDWARE_POST_MOTION_TESTS", True))
            and not is_demo_hardware
        ):
            motion_test_results = _run_post_tune_motion_tests(
                bridge, emit_console=emit_console, controller=controller
            )
            if motion_test_results and not all(
                bool(item.get("passed")) for item in motion_test_results
            ):
                _console(
                    emit_console,
                    "[WARN] PID验证已通过，但横移/旋转动作验收存在失败项，请检查运动学或机械状态。",
                )

    except RoundAdmissionClosed as exc:
        session.completed_reason = str(exc)
    except (ConnectionError, TimeoutError, RuntimeError, ValueError) as exc:
        session.completed_reason = "hardware_error"
        failure_detail = str(exc)
        _emit_lifecycle(event_sink, start_time, "error", str(exc))
        _console(emit_console, f"[ERROR] {exc}")
    except KeyboardInterrupt:
        session.completed_reason = "keyboard_interrupt"
        _console(emit_console, "\n[INFO] 用户停止")
        _emit_lifecycle(
            event_sink,
            start_time,
            "stopped",
            "Hardware tuning interrupted by keyboard.",
        )
    finally:
        try:
            stop_confirmation = _confirm_hardware_stop(bridge)
            if not is_demo_hardware:
                pid_snapshot = _read_pid_snapshot(bridge)
            # Keep the stopped TUNE mode; MODE WORK has motor/actuator side effects.
        except Exception as exc:
            stop_confirmation = "unconfirmed" if stop_confirmation == "not_requested" else stop_confirmation
            _emit_log(event_sink, start_time, "error", f"Cleanup: {exc}")
            failure_detail = failure_detail or str(exc)
        finally:
            if transcript is not None:
                try:
                    capture_failure(bridge)
                except Exception as exc:
                    bridge.trace("ERROR", f"exit capture: {exc}")
            if round_csv is not None:
                round_csv.close()
            try:
                bridge.disconnect()
            finally:
                if transcript is not None:
                    transcript.close()
        _emit_lifecycle(
            event_sink,
            start_time,
            "finished",
            f"Hardware tuning finished in {now_elapsed(start_time):.1f}s.",
        )

    final_pid = dict(tested_result["pid"] if tested_result else session.buffer.current_pid)
    result = {
        "elapsed_sec": now_elapsed(start_time),
        "failure_detail": failure_detail,
        "failed_round": failed_round,
        "serial_log_path": str(transcript.path) if transcript else None,
        "tune_axis": tune_axis,
        "output_limit": hardware_output_limit,
        "output_unit": "rad/s" if is_yaw_axis else "m/s",
        "pid_snapshot": pid_snapshot,
        "suggested_pid": dict(session.buffer.current_pid),
        "loaded_pid": pid_snapshot.get(tune_axis, loaded_pid),
        "tested_result": tested_result,
        "verified_pid": dict(tested_result["pid"]) if tested_result and tested_result["verified"] else None,
        "stop_confirmation": stop_confirmation,
        "motion_tests": motion_test_results,
        **build_tuning_result(
            session,
            final_pid=final_pid,
            stopped=bool(controller.should_stop) if controller is not None else False,
        ),
    }
    if tested_result:
        result["final_metrics"] = dict(tested_result["metrics"])
        result["final_yaw_pid"] = dict(tested_result["yaw_pid"])
    round_history = [
        {
            "round": item.get("round"),
            "pid": item.get("pid", {}),
            "metrics": item.get("metrics", {}),
            "analysis": item.get("analysis", ""),
        }
        for item in session.history.history
    ]
    summary_payload = {
        "tune_axis": tune_axis,
        "rounds_completed": result["rounds_completed"],
        "completed_reason": result["completed_reason"],
        "failure_detail": failure_detail,
        "failed_round": failed_round,
        "final_pid": final_pid,
        "pid_snapshot": pid_snapshot,
        "output_limit": hardware_output_limit,
        "output_unit": result["output_unit"],
        "final_metrics": result["final_metrics"],
        "motion_tests": motion_test_results,
        "round_history": round_history,
    }
    summarize = getattr(tuner, "summarize_tuning_session", None)
    ai_summary = (
        summarize(summary_payload)
        if callable(summarize) and int(result["rounds_completed"] or 0) > 0
        else None
    )
    result["ai_summary"] = (
        ai_summary
        if isinstance(ai_summary, dict)
        else build_local_tuning_summary(result, round_history)
    )
    return result


def _run_hardware_tuning_with_tui(
    serial_port: str,
    initial_pid: dict[str, float] | None = None,
    preserved_pids: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    from sim.tui import SimulationTUIApp

    event_queue: Queue[dict[str, Any]] = Queue()
    controller = SimulationController()
    event_sink = QueueEventSink(event_queue)
    result_box: dict[str, Any] = {}
    language = choose_tui_language()

    def make_worker(pid: dict[str, float] | None) -> Callable[[], None]:
        def worker() -> None:
            result = _run_hardware_tuning_loop(
                serial_port,
                event_sink=event_sink,
                controller=app.controller,
                emit_console=False,
                initial_pid=pid,
                preserved_pids=preserved_pids,
            )
            result_box["result"] = result
            app._last_result = result

        return worker

    def next_round_factory(last_result: dict[str, Any]) -> Callable[[], None]:
        pid = last_result.get("final_pid")
        return make_worker(pid if isinstance(pid, dict) else None)

    app = SimulationTUIApp(
        event_queue=event_queue,
        controller=controller,
        worker_target=make_worker(initial_pid),
        event_sink=event_sink,
        mode_label="Hardware",
        language=language,
        next_round_factory=next_round_factory,
    )
    app.run()
    return result_box.get("result", {})


def _run_hardware_tuning_plain(
    serial_port: str,
    initial_pid: dict[str, float] | None = None,
    preserved_pids: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    print("=" * 60)
    print("  LLM PID Tuner PRO - 增强版自动调参系统")
    print("=" * 60)
    print(f"Serial Port: {serial_port}, Model: {CONFIG['LLM_MODEL_NAME']}")
    return _run_hardware_tuning_loop(
        serial_port,
        emit_console=True,
        initial_pid=initial_pid,
        preserved_pids=preserved_pids,
    )


def run_hardware_tuner(
    serial_port_arg: str | None = None,
    force_plain: bool = False,
    initial_pid: dict[str, float] | None = None,
    *,
    tune_axis: str | None = None,
    preserved_pids: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    if tune_axis is not None:
        selected = parse_hardware_axes(tune_axis)
        if len(selected) != 1:
            raise ValueError("单次调参只能指定一个轴。")
        tune_axis = selected[0]
    initialize_runtime_config(create_if_missing=True, verbose=True)
    if tune_axis is not None:
        CONFIG["HARDWARE_TUNE_AXIS"] = tune_axis
        # An explicit axis selection adjusts only that axis; yaw hold stays active
        # with fixed gains during X/Y tuning.
        CONFIG["HARDWARE_YAW_COPILOT"] = False
    serial_port = resolve_serial_port(serial_port_arg)
    if not serial_port:
        print("[ERROR] 未指定串口，程序退出。")
        safe_pause()
        return {"completed_reason": "no_serial_port"}

    tune_axis = _normalize_hardware_axis(CONFIG.get("HARDWARE_TUNE_AXIS", "Y"))
    limit_mode = "hardware_yaw" if tune_axis == "YAW" else "hardware"
    if initial_pid is None and bool(CONFIG.get("HARDWARE_RESUME_LAST_PID", False)):
        result_log = str(
            resolve_project_path(CONFIG.get("PID_RESULT_LOG", "logs/pid_results.jsonl"))
        )
        initial_pid = load_last_usable_pid(
            result_log, get_pid_limits(limit_mode), tune_axis
        )
        if initial_pid is not None:
            print(
                f"[INFO] 从最近一次可靠 {tune_axis} 轴记录续调: "
                f"P={initial_pid['p']} I={initial_pid['i']} D={initial_pid['d']}"
            )
        else:
            initial_pid = _configured_initial_pid(tune_axis)
            print(
                f"[INFO] 未找到可靠 {tune_axis} 轴历史 PID，"
                f"使用显式配置初值 P={initial_pid['p']} "
                f"I={initial_pid['i']} D={initial_pid['d']}"
            )

    result: dict[str, Any]
    runner_kwargs: dict[str, Any] = {"initial_pid": initial_pid}
    if preserved_pids is not None:
        runner_kwargs["preserved_pids"] = preserved_pids
    if not force_plain:
        try:
            result = _run_hardware_tuning_with_tui(serial_port, **runner_kwargs)
        except Exception as exc:
            print(f"[WARN] Failed to start the TUI ({exc}); falling back to plain output.")
            if bool(CONFIG.get("LLM_DEBUG_OUTPUT")):
                traceback.print_exc()
            result = _run_hardware_tuning_plain(serial_port, **runner_kwargs)
    else:
        result = _run_hardware_tuning_plain(serial_port, **runner_kwargs)

    try:
        saved_path = append_pid_result(
            result,
            str(
                resolve_project_path(
                    CONFIG.get("PID_RESULT_LOG", "logs/pid_results.jsonl")
                )
            ),
        )
        if saved_path is not None:
            print(f"[INFO] 本次最终 PID 已追加保存到: {saved_path}")
    except Exception as exc:
        # 记录失败不能影响急停和主调参结果。
        print(f"[WARN] 最终 PID 记录保存失败: {exc}")
    return result


def _axis_tuning_succeeded(result: dict[str, Any]) -> bool:
    return (
        result.get("completed_reason") == "staged_validation_passed"
        and isinstance(result.get("verified_pid"), dict)
        and all(key in result["verified_pid"] for key in ("p", "i", "d"))
        and not result.get("failure_detail")
        and not result.get("stopped")
        and result.get("stop_confirmation") in {"command_sent", "can_stop_sent"}
        and all(item.get("passed") for item in result.get("motion_tests", []))
    )


def run_hardware_axis_sequence(
    axes: list[str], serial_port_arg: str | None = None, force_plain: bool = False,
) -> list[dict[str, Any]]:
    axes = parse_hardware_axes(" ".join(axes))
    results: list[dict[str, Any]] = []
    preserved_pids: dict[str, dict[str, float]] = {}
    print(f"[INFO] 本次调参顺序：{' -> '.join(axes)}")
    for index, axis in enumerate(axes, start=1):
        print(f"\n[INFO] 开始调试 {axis} 轴 ({index}/{len(axes)})")
        result = run_hardware_tuner(
            serial_port_arg, force_plain=force_plain,
            tune_axis=axis, preserved_pids=dict(preserved_pids),
        )
        results.append(result)
        if not _axis_tuning_succeeded(result):
            print(
                f"[WARN] {axis} 轴未正常完成："
                f"{result.get('completed_reason', 'unknown')}。停止后续轴调参。"
            )
            break
        preserved_pids[axis] = dict(result["verified_pid"])
        print(f"[INFO] {axis} 轴调参完成。")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.axes is None:
        run_hardware_tuner(args.serial_port, force_plain=args.plain)
        return
    try:
        axes = parse_hardware_axes(" ".join(args.axes))
    except ValueError as exc:
        parser.error(str(exc))
    results = run_hardware_axis_sequence(axes, args.serial_port, force_plain=args.plain)
    if len(results) != len(axes) or not all(_axis_tuning_succeeded(r) for r in results):
        raise SystemExit(1)
    print(f"[INFO] 所选轴全部调参完成：{' -> '.join(axes)}")


if __name__ == "__main__":
    main()
