"""Persist completed hardware tuning results without storing secrets."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SUCCESS_REASONS = {
    "staged_validation_passed",
    "stable_rounds_reached",
    "low_error_converged",
    "rollback_to_best",
    "llm_marked_done",
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

        reason = str(record.get("completed_reason", ""))
        usable = reason in _SUCCESS_REASONS
        if reason == "max_rounds_reached":
            # 兼容验证误判修复前留下的旧记录：必须确实到位且没有明显超调。
            metrics = record.get("final_metrics", {})
            try:
                tolerance = 1.0 if record_axis == "YAW" else 5.0
                usable = (
                    abs(float(metrics["current_error"])) <= tolerance
                    and float(metrics["overshoot"]) <= 3.0
                    and int(metrics.get("zero_crossings", 0)) <= 6
                )
            except (KeyError, TypeError, ValueError):
                usable = False

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
    record = {
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
        "final_metrics": result.get("final_metrics", {}),
        "motion_tests": result.get("motion_tests", []),
        "fallback_count": result.get("fallback_count", 0),
        "guardrail_count": result.get("guardrail_count", 0),
        "rollback_count": result.get("rollback_count", 0),
    }
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path
