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

from core.config import CONFIG, initialize_runtime_config
from core.pid_results import append_pid_result, load_last_usable_pid
from core.tuning_session import (
    apply_rollback,
    build_tuning_result,
    create_tuning_session,
    evaluate_completed_round,
    finalize_decision,
    record_rollback_round,
)
from hw.bridge import SerialBridge, safe_pause, select_serial_port
from llm.client import LLMTuner
from pid_safety import build_fallback_suggestion, get_pid_limits
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
    wait_while_paused,
)

HARDWARE_WRONG_DIRECTION_MM = 20.0
HARDWARE_MAX_YAW_ERROR_DEG = 15.0
HARDWARE_TUNING_STAGES = ("P", "I", "D", "VERIFY")
OPS_CENTER_OFFSET_X_MM = 0.0
OPS_CENTER_OFFSET_Y_MM = 25.0


def _normalize_hardware_axis(value: Any) -> str:
    axis = str(value or "Y").strip().upper()
    return axis if axis in {"X", "Y", "YAW"} else "Y"


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


def _build_hardware_prompt_context(
    serial_port: str,
    tuning_stage: str = "P",
    output_limit: float | None = None,
    tune_axis: str | None = None,
) -> dict[str, Any]:
    axis = _normalize_hardware_axis(tune_axis or CONFIG.get("HARDWARE_TUNE_AXIS", "Y"))
    is_yaw = axis == "YAW"
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
        "per_round_guardrail_hint": "Keep P within about 3x the current value, and keep I/D within about 4x. Prefer smaller moves near stability.",
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
    return (
        stop_reason == "TARGET"
        and float(metrics.get("overshoot", float("inf")))
        <= float(CONFIG["GOOD_ENOUGH_OVERSHOOT"])
        # TARGET 已表示固件连续10个周期进入 +/-5 mm；最后20%平均值包含减速段，
        # 不再用它否决已经到位的轮次，避免在 I/D/VERIFY 之间无限循环。
        and float(metrics.get("current_error", float("inf")))
        <= (1.0 if tune_axis == "YAW" else 5.0)
        and int(metrics.get("zero_crossings", 0)) <= 6
    )


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


def _wait_for_chassis_auto_stop(bridge: Any, timeout_sec: float) -> bool:
    deadline = time.monotonic() + timeout_sec
    last_ping = time.monotonic()
    while time.monotonic() < deadline:
        line = bridge.read_line()
        if line and line.startswith("# MOVE AUTO STOP"):
            return True
        now = time.monotonic()
        if now - last_ping >= 0.5:
            sender = getattr(bridge, "send_silent_command", bridge.send_command)
            sender("PING")
            last_ping = now
    bridge.send_command("MOVE STOP")
    return False


def _run_post_tune_motion_tests(
    bridge: Any, *, emit_console: bool = True
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
        before = _read_ops_pose(bridge)
        if before is None:
            results.append({"action": name, "passed": False, "reason": "OPS_NOT_READY"})
            _console(emit_console, f"[MotionTest] {name} FAIL: OPS_NOT_READY")
            continue

        bridge.send_command(command)
        stopped = _wait_for_chassis_auto_stop(
            bridge, duration_ms / 1000.0 + 1.5
        )
        time.sleep(0.05)
        after = _read_ops_pose(bridge)
        if after is None:
            results.append({"action": name, "passed": False, "reason": "OPS_NOT_READY_AFTER"})
            _console(emit_console, f"[MotionTest] {name} FAIL: OPS_NOT_READY_AFTER")
            continue

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
    return parser


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
) -> dict[str, Any]:
    tune_axis = _normalize_hardware_axis(CONFIG.get("HARDWARE_TUNE_AXIS", "Y"))
    is_yaw_axis = tune_axis == "YAW"
    if initial_pid is None:
        initial_pid = {"p": 0.01 if is_yaw_axis else 0.001, "i": 0.0, "d": 0.0}

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
    is_demo_hardware = serial_port.strip().upper() == "COM_FAKE"
    # Five seconds at 20 ms is about 250 samples; keep the whole real round.
    hardware_buffer_size = (
        int(CONFIG["BUFFER_SIZE"])
        if is_demo_hardware
        else max(int(CONFIG["BUFFER_SIZE"]), 300)
    )
    session = create_tuning_session(
        initial_pid=initial_pid,
        buffer_size=hardware_buffer_size,
    )
    start_time = time.time()
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

    if not bridge.connect():
        message = f"无法打开串口 {serial_port}: {bridge.last_error or 'unknown error'}"
        session.completed_reason = "error"
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

    _console(emit_console, f"[INFO] 已连接到串口: {serial_port}")
    _emit_lifecycle(
        event_sink,
        start_time,
        "connected",
        f"Connected to {serial_port}.",
    )

    motion_test_results: list[dict[str, Any]] = []
    try:
        last_heartbeat_at = time.monotonic()
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

        # 先静默恢复其余轴的可靠参数；PID SET只装载参数，不会触发车辆运动。
        result_log = str(CONFIG.get("PID_RESULT_LOG", "logs/pid_results.jsonl"))
        for saved_axis in ("X", "Y", "YAW"):
            if saved_axis == tune_axis:
                continue
            saved_mode = "hardware_yaw" if saved_axis == "YAW" else "hardware"
            saved_pid = load_last_usable_pid(
                result_log, get_pid_limits(saved_mode), saved_axis
            )
            if saved_pid is None:
                continue
            load_cmd = (
                f"PID SET {saved_axis} {saved_pid['p']} "
                f"{saved_pid['i']} {saved_pid['d']}"
            )
            bridge.send_command(load_cmd)
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
        if initial_pid:
            cmd = f"SET P:{initial_pid['p']} I:{initial_pid['i']} D:{initial_pid['d']}"
            bridge.send_command(cmd)
            _emit_log(event_sink, start_time, "cmd", cmd)
            _console(emit_console, f"[CMD] Initial PID: {cmd}")
        time.sleep(1)

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
        round_complete = False
        round_sample_count = 0
        round_started_at = time.monotonic()
        round_start_yaw: float | None = None
        round_stop_reason = "UNKNOWN"
        tuning_stage = "P"
        stage_rounds = 0
        verification_passes = 0
        required_verification_rounds = max(
            1, int(CONFIG.get("HARDWARE_VERIFY_ROUNDS", 3))
        )
        max_stage_rounds = max(1, int(CONFIG.get("HARDWARE_STAGE_MAX_ROUNDS", 5)))
        _console(emit_console, "[Stage] 进入 P 阶段：只调整 P，冻结 I/D")

        while session.round_num < CONFIG["MAX_TUNING_ROUNDS"]:
            # Keep the STM32 host-loss watchdog alive only while this loop is healthy.
            heartbeat_now = time.monotonic()
            if heartbeat_now - last_heartbeat_at >= 0.5:
                heartbeat_sender = getattr(bridge, "send_silent_command", bridge.send_command)
                heartbeat_sender("PING")
                last_heartbeat_at = heartbeat_now

            if controller is not None and controller.should_stop:
                session.completed_reason = "stopped_by_user"
                _console(emit_console, "\n[INFO] 用户停止")
                _emit_lifecycle(event_sink, start_time, "stopped", "Hardware tuning stopped by user.")
                break

            if not wait_while_paused(controller):
                session.completed_reason = "stopped_by_user"
                _console(emit_console, "\n[INFO] 用户停止")
                _emit_lifecycle(event_sink, start_time, "stopped", "Hardware tuning stopped by user.")
                break

            line = bridge.read_line()
            if line:
                if line.startswith("#"):
                    last_device_message = line
                    important_status = line.startswith(
                        ("# STATUS", "# PID", "# ROUND", "# ERROR", "# OPS")
                    )
                    if important_status:
                        _console(emit_console, f"[MCU] {line}")
                        _emit_log(event_sink, start_time, "mcu", line)

                    if line.startswith("# ROUND START"):
                        round_active = True
                        round_complete = False
                        round_sample_count = 0
                        round_started_at = time.monotonic()
                        last_sample_at = round_started_at
                        round_start_yaw = None
                        round_stop_reason = "UNKNOWN"
                    elif line.startswith("# ROUND STOP") and round_active:
                        stop_payload = line.removeprefix("# ROUND STOP").strip()
                        stop_reason = stop_payload.split(maxsplit=1)[0] if stop_payload else "UNKNOWN"
                        round_stop_reason = stop_reason
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
                    last_sample_at = time.monotonic()
                    if round_active:
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
            )
            _augment_hardware_round_metrics(
                evaluation.metrics,
                list(session.buffer.buffer),
                output_limit=hardware_output_limit,
                stop_reason=round_stop_reason,
            )
            session.last_metrics.update(evaluation.metrics)
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
                        _console(emit_console, "\n[SUCCESS] P、I、D 分阶段调参及最终验证完成！")
                        _emit_lifecycle(
                            event_sink,
                            start_time,
                            "completed",
                            "P/I/D staged tuning passed final verification.",
                        )
                        break
                else:
                    verification_passes = 0
                    if float(evaluation.metrics.get("steady_state_error", 0.0)) > hardware_steady_error_limit:
                        tuning_stage = "I"
                        _console(emit_console, "[Stage] 验证未通过：返回 I 阶段处理稳态误差")
                    else:
                        tuning_stage = "D"
                        _console(emit_console, "[Stage] 验证未通过：返回 D 阶段处理超调/振荡")
                    stage_rounds = 0

                if session.round_num >= int(CONFIG["MAX_TUNING_ROUNDS"]):
                    session.completed_reason = "max_rounds_reached"
                    break

                cmd = (
                    f"SET P:{evaluation.current_pid['p']} "
                    f"I:{evaluation.current_pid['i']} D:{evaluation.current_pid['d']}"
                )
                clear_input = getattr(bridge, "clear_input_buffer", None)
                if callable(clear_input):
                    clear_input()
                bridge.send_command(cmd)
                last_sample_at = time.monotonic()
                last_heartbeat_at = last_sample_at
                last_device_message = "waiting for verification round"
                round_active = is_demo_hardware
                round_complete = False
                round_sample_count = 0
                round_started_at = last_sample_at
                _emit_log(event_sink, start_time, "cmd", cmd)
                _console(emit_console, f"[CMD] Sent: {cmd}")
                time.sleep(1)
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

                cmd = (
                    f"SET P:{evaluation.rollback_pid['p']} "
                    f"I:{evaluation.rollback_pid['i']} D:{evaluation.rollback_pid['d']}"
                )
                time.sleep(0.05)
                clear_input = getattr(bridge, "clear_input_buffer", None)
                if callable(clear_input):
                    clear_input()
                bridge.send_command(cmd)
                last_sample_at = time.monotonic()
                last_heartbeat_at = last_sample_at
                last_device_message = "waiting for rollback round"
                round_active = is_demo_hardware
                round_complete = False
                round_sample_count = 0
                round_started_at = last_sample_at
                _emit_log(event_sink, start_time, "cmd", cmd)
                _console(emit_console, f"[CMD] Sent: {cmd}")
                apply_rollback(session, evaluation.rollback_pid)
                stage_rounds += 1
                time.sleep(1)
                continue

            prompt_data = session.buffer.to_prompt_data()
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
                prompt_context=_build_hardware_prompt_context(
                    serial_port, tuning_stage, hardware_output_limit, tune_axis
                ),
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
            )
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
            if decision.guardrail_notes:
                _console(emit_console, f"[Guardrail] {'; '.join(decision.guardrail_notes)}")
            if decision.fallback_used:
                _console(emit_console, "[Fallback] 本轮使用规则策略替代 LLM 建议。")

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

            cmd = (
                f"SET P:{decision.safe_pid['p']} "
                f"I:{decision.safe_pid['i']} D:{decision.safe_pid['d']}"
            )
            clear_input = getattr(bridge, "clear_input_buffer", None)
            if callable(clear_input):
                clear_input()
            bridge.send_command(cmd)
            last_sample_at = time.monotonic()
            last_heartbeat_at = last_sample_at
            last_device_message = "waiting for next tuning round"
            round_active = is_demo_hardware
            round_complete = False
            round_sample_count = 0
            round_started_at = last_sample_at
            _emit_log(event_sink, start_time, "cmd", cmd)
            _console(emit_console, f"[CMD] Sent: {cmd}")

            time.sleep(1)

        if (
            session.completed_reason == "staged_validation_passed"
            and tune_axis == "Y"
            and bool(CONFIG.get("HARDWARE_POST_MOTION_TESTS", True))
            and not is_demo_hardware
        ):
            motion_test_results = _run_post_tune_motion_tests(
                bridge, emit_console=emit_console
            )
            if motion_test_results and not all(
                bool(item.get("passed")) for item in motion_test_results
            ):
                _console(
                    emit_console,
                    "[WARN] PID验证已通过，但横移/旋转动作验收存在失败项，请检查运动学或机械状态。",
                )

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
        bridge.send_command("STOP")
        time.sleep(0.05)
        bridge.disconnect()
        _emit_lifecycle(
            event_sink,
            start_time,
            "finished",
            f"Hardware tuning finished in {now_elapsed(start_time):.1f}s.",
        )

    return {
        "elapsed_sec": now_elapsed(start_time),
        "tune_axis": tune_axis,
        "motion_tests": motion_test_results,
        **build_tuning_result(
            session,
            final_pid=dict(session.buffer.current_pid),
            stopped=bool(controller.should_stop) if controller is not None else False,
        ),
    }


def _run_hardware_tuning_with_tui(
    serial_port: str,
    initial_pid: dict[str, float] | None = None,
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
) -> dict[str, Any]:
    print("=" * 60)
    print("  LLM PID Tuner PRO - 增强版自动调参系统")
    print("=" * 60)
    print(f"Serial Port: {serial_port}, Model: {CONFIG['LLM_MODEL_NAME']}")
    return _run_hardware_tuning_loop(
        serial_port,
        emit_console=True,
        initial_pid=initial_pid,
    )


def run_hardware_tuner(
    serial_port_arg: str | None = None,
    force_plain: bool = False,
    initial_pid: dict[str, float] | None = None,
) -> dict[str, Any]:
    initialize_runtime_config(create_if_missing=True, verbose=True)
    serial_port = resolve_serial_port(serial_port_arg)
    if not serial_port:
        print("[ERROR] 未指定串口，程序退出。")
        safe_pause()
        return {"completed_reason": "no_serial_port"}

    tune_axis = _normalize_hardware_axis(CONFIG.get("HARDWARE_TUNE_AXIS", "Y"))
    limit_mode = "hardware_yaw" if tune_axis == "YAW" else "hardware"
    if initial_pid is None and bool(CONFIG.get("HARDWARE_RESUME_LAST_PID", True)):
        result_log = str(CONFIG.get("PID_RESULT_LOG", "logs/pid_results.jsonl"))
        initial_pid = load_last_usable_pid(
            result_log, get_pid_limits(limit_mode), tune_axis
        )
        if initial_pid is not None:
            print(
                f"[INFO] 从最近一次可靠 {tune_axis} 轴记录续调: "
                f"P={initial_pid['p']} I={initial_pid['i']} D={initial_pid['d']}"
            )
        else:
            initial_pid = {
                "p": 0.01 if tune_axis == "YAW" else 0.001,
                "i": 0.0,
                "d": 0.0,
            }
            print(
                f"[INFO] 未找到可靠 {tune_axis} 轴历史 PID，"
                f"使用安全初值 P={initial_pid['p']} I=0 D=0"
            )

    result: dict[str, Any]
    if not force_plain:
        try:
            result = _run_hardware_tuning_with_tui(serial_port, initial_pid=initial_pid)
        except Exception as exc:
            print(f"[WARN] Failed to start the TUI ({exc}); falling back to plain output.")
            if bool(CONFIG.get("LLM_DEBUG_OUTPUT")):
                traceback.print_exc()
            result = _run_hardware_tuning_plain(serial_port, initial_pid=initial_pid)
    else:
        result = _run_hardware_tuning_plain(serial_port, initial_pid=initial_pid)

    try:
        saved_path = append_pid_result(
            result, str(CONFIG.get("PID_RESULT_LOG", "logs/pid_results.jsonl"))
        )
        if saved_path is not None:
            print(f"[INFO] 本次最终 PID 已追加保存到: {saved_path}")
    except Exception as exc:
        # 记录失败不能影响急停和主调参结果。
        print(f"[WARN] 最终 PID 记录保存失败: {exc}")
    return result


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_hardware_tuner(args.serial_port, force_plain=args.plain)


if __name__ == "__main__":
    main()
