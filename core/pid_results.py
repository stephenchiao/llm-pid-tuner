"""Persist completed hardware tuning results without storing secrets."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def has_verified_result(result: dict[str, Any]) -> bool:
    evidence = result.get("tested_result")
    pid = result.get("final_pid")
    motion_tests = result.get("motion_tests", [])
    return (
        result.get("completed_reason") == "staged_validation_passed"
        and isinstance(evidence, dict)
        and evidence.get("verified") is True
        and evidence.get("axis") == str(result.get("tune_axis", "Y")).upper()
        and evidence.get("pid") == pid
        and evidence.get("metrics") == result.get("final_metrics")
        and result.get("verified_pid") == pid
        and result.get("stop_confirmation") in ("can_stop_sent", "feedback_confirmed")
        and isinstance(motion_tests, list)
        and all(isinstance(item, dict) and item.get("passed") is True for item in motion_tests)
    )


def build_local_tuning_summary(
    result: dict[str, Any],
    round_history: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """LLM不可用时生成可审计的简要总结，保证每次会话日志字段完整。"""
    axis = str(result.get("tune_axis", "Y")).upper()
    rounds = int(result.get("rounds_completed") or 0)
    reason = str(result.get("completed_reason") or "unknown")
    final_pid = result.get("final_pid") or {}
    metrics = result.get("final_metrics") or {}
    history = round_history or []
    analyses = [
        str(item.get("analysis", "")).strip()
        for item in history
        if str(item.get("analysis", "")).strip()
    ]

    process = (
        f"{axis}轴共完成{rounds}轮，结束原因={reason}；"
        f"最终P={float(final_pid.get('p', 0.0)):.7g}，"
        f"I={float(final_pid.get('i', 0.0)):.7g}，"
        f"D={float(final_pid.get('d', 0.0)):.7g}。"
    )
    if analyses:
        process += "最近一轮分析：" + analyses[-1][:240]
    if result.get("failure_detail"):
        process += "异常停止：" + str(result["failure_detail"])

    metric_parts: list[str] = []
    for key, label in (
        ("current_error", "最终误差"),
        ("overshoot", "超调"),
        ("steady_state_error", "稳态误差"),
        ("cross_track_peak_mm", "最大横向漂移"),
        ("yaw_delta_peak_deg", "最大航向变化"),
    ):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            metric_parts.append(f"{label}={float(value):.3g}")

    passed = has_verified_result(result)
    evaluation = "调参和验证已完成" if passed else "本次会话未形成可自动复用的可靠结果"
    if metric_parts:
        evaluation += "；最后完整轮次：" + "，".join(metric_parts)
    if result.get("failed_round"):
        failed = result["failed_round"]
        evaluation += f"；第{failed['round']}轮异常停止={failed['stop_reason']}，未通过验证"
    recommendation = (
        "保存该组参数，并在正反方向、不同载荷和连续运行条件下复验。"
        if passed
        else "检查结束原因和安全日志，排除硬件或数据问题后重新调试；不要自动装载本次结果。"
    )
    return {
        "process_summary": process,
        "evaluation": evaluation + "。",
        "recommendation": recommendation,
        "source": "local_fallback",
    }


def load_last_usable_pid(
    path: str,
    limits: dict[str, dict[str, float]],
    tune_axis: str = "Y",
) -> dict[str, float] | None:
    """倒序读取最近的可靠结果；中止或故障轮次不能作为下次起点。"""
    input_path = Path(path)
    if not input_path.is_file():
        return None

    try:
        lines = input_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    for line in reversed(lines):
        try:
            record = json.loads(line)
            pid = record["final_pid"]
            values = {key: float(pid[key]) for key in ("p", "i", "d")}
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue

        record_axis = str(record.get("tune_axis", "Y")).strip().upper()
        if record_axis != tune_axis.strip().upper():
            continue

        if any(not math.isfinite(value) for value in values.values()):
            continue
        if any(
            values[key] < float(limits[key]["min"])
            or values[key] > float(limits[key]["max"])
            for key in values
        ):
            continue

        usable = record.get("format_version") == 3 and has_verified_result(record)

        if usable:
            return values
    return None


def append_pid_result(result: dict[str, Any], path: str) -> Path | None:
    """追加保存一次调参结果；没有最终 PID 的失败启动不会写入。"""
    final_pid = result.get("final_pid")
    if not isinstance(final_pid, dict):
        return None

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ai_summary = result.get("ai_summary")
    if not isinstance(ai_summary, dict):
        ai_summary = build_local_tuning_summary(result)

    record = {
        "format_version": 3,
        "tested_result": result.get("tested_result"),
        "verified_pid": result.get("verified_pid"),
        "suggested_pid": result.get("suggested_pid"),
        "loaded_pid": result.get("loaded_pid"),
        "stop_confirmation": result.get("stop_confirmation", "unknown"),
        "serial_log_path": result.get("serial_log_path"),
        "failure_detail": result.get("failure_detail", ""),
        "failed_round": result.get("failed_round"),
        "saved_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "provider": result.get("provider"),
        "model": result.get("model"),
        "rounds_completed": result.get("rounds_completed"),
        "completed_reason": result.get("completed_reason"),
        "tune_axis": str(result.get("tune_axis", "Y")).upper(),
        "final_pid": {
            "p": float(final_pid.get("p", 0.0)),
            "i": float(final_pid.get("i", 0.0)),
            "d": float(final_pid.get("d", 0.0)),
        },
        "final_yaw_pid": result.get("final_yaw_pid", {}),
        "pid_snapshot": result.get("pid_snapshot", {}),
        "output_limit": result.get("output_limit"),
        "output_unit": result.get("output_unit"),
        "final_metrics": result.get("final_metrics", {}),
        "ai_summary": {
            "process_summary": str(ai_summary.get("process_summary", "")),
            "evaluation": str(ai_summary.get("evaluation", "")),
            "recommendation": str(ai_summary.get("recommendation", "")),
            "source": str(ai_summary.get("source", "local_fallback")),
        },
        "motion_tests": result.get("motion_tests", []),
        "elapsed_sec": result.get("elapsed_sec"),
        "stopped": bool(result.get("stopped", False)),
        "fallback_count": result.get("fallback_count", 0),
        "guardrail_count": result.get("guardrail_count", 0),
        "rollback_count": result.get("rollback_count", 0),
    }
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path
