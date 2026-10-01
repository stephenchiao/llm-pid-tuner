#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PID 参数安全护栏。

设计原则：
1. 不干预正常的小步调参；
2. 只限制明显危险的异常跳变；
3. 在 LLM 失败时提供保守、可解释的兜底策略。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from core.config import CONFIG


PID_KEYS = ("p", "i", "d")

DEFAULT_PID_LIMITS: Dict[str, Dict[str, float]] = {
    "p": {"min": 0.0, "max": 100.0, "max_increase_ratio": 3.0},
    "i": {"min": 0.0, "max":  30.0, "max_increase_ratio": 4.0},
    "d": {"min": 0.0, "max":  20.0, "max_increase_ratio": 4.0},
}

HARDWARE_PID_LIMITS: Dict[str, Dict[str, float]] = {
    # STM32F407 麦克纳姆底盘 Y 轴位置环：误差单位 mm，输出单位 m/s。
    # 固件当前采用 20 ms 离散位置式 PID，I/D 参数也是离散域增益。
    "p": {"min": 0.0, "max": 0.005,   "max_increase_ratio": 1.5},
    "i": {"min": 0.0, "max": 0.00005, "max_increase_ratio": 1.5},
    "d": {"min": 0.0, "max": 0.002,   "max_increase_ratio": 1.5},
}

HARDWARE_YAW_PID_LIMITS: Dict[str, Dict[str, float]] = {
    # 航向误差单位 degree，输出单位 rad/s，不能复用毫米位置环的增益范围。
    "p": {"min": 0.0, "max": 0.05,    "max_increase_ratio": 1.5},
    "i": {"min": 0.0, "max": 0.00010, "max_increase_ratio": 1.5},
    "d": {"min": 0.0, "max": 0.02,    "max_increase_ratio": 1.5},
}

PYTHON_SIM_PID_LIMITS: Dict[str, Dict[str, float]] = {
    "p": {"min": 0.0, "max": 5000.0, "max_increase_ratio": 3.0},
    "i": {"min": 0.0, "max":  500.0, "max_increase_ratio": 4.0},
    "d": {"min": 0.0, "max":  500.0, "max_increase_ratio": 4.0},
}

SIMULINK_PID_LIMITS: Dict[str, Dict[str, float]] = {
    "p": {"min": 0.0, "max": 5000.0, "max_increase_ratio": 5.0},
    "i": {"min": 0.0, "max":  500.0, "max_increase_ratio": 6.0},
    "d": {"min": 0.0, "max":  500.0, "max_increase_ratio": 6.0},
}

DEFAULT_CONVERGENCE_RULES: Dict[str, float] = {
    "avg_error_threshold"         : 80.0,
    "steady_state_error_threshold": 10.0,
    "overshoot_threshold"         : 3.0,
}

DEFAULT_ROLLBACK_RULES: Dict[str, float] = {
    "avg_error_ratio"          : 1.5,
    "avg_error_margin"         : 0.5,
    "steady_state_error_ratio" : 1.8,
    "steady_state_error_margin": 0.25,
    "overshoot_margin"         : 1.0,
}


def get_pid_limits(mode: str | None = None) -> Dict[str, Dict[str, float]]:
    normalized = (mode or "").strip().lower()
    if normalized == "python_sim":
        source = PYTHON_SIM_PID_LIMITS
    elif normalized == "simulink":
        source = SIMULINK_PID_LIMITS
    elif normalized == "hardware_yaw":
        source = HARDWARE_YAW_PID_LIMITS
    elif normalized == "hardware":
        source = HARDWARE_PID_LIMITS
    else:
        source = DEFAULT_PID_LIMITS

    return {key: dict(value) for key, value in source.items()}


def _to_float(value: Any, fallback: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(numeric):
        return fallback
    return numeric


def apply_pid_guardrails(
    current_pid  : Dict[str, float],
    candidate_pid: Dict[str, Any],
    limits       : Dict[str, Dict[str, float]] | None = None,
) -> Tuple[Dict[str, float], List[str]]:
    """将候选 PID 参数裁剪到安全范围内。"""
    limits   = limits or DEFAULT_PID_LIMITS
    sanitized: Dict[str, float] = {}
    notes    : List[str]        = []

    for key in PID_KEYS:
        current_value = max(0.0, _to_float(current_pid.get(key, 0.0), 0.0))
        raw_value     = _to_float(candidate_pid.get(key, current_value), current_value)

        cfg           = limits.get(key, DEFAULT_PID_LIMITS[key])
        bounded_value = max(cfg["min"], min(cfg["max"], raw_value))
        if current_value >= cfg["max"] and raw_value > current_value:
            notes.append(f"{key.upper()} 已达上限 {cfg['max']:.4f}，将不会继续升高")

        max_increase_ratio = max(1.0, cfg.get("max_increase_ratio", 1.0))
        if current_value > 0:
            max_step_value = min(cfg["max"], current_value * max_increase_ratio)
            if bounded_value > max_step_value:
                notes.append(
                    f"{key.upper()} 增幅过大，已从 {bounded_value:.4f} 限制到 {max_step_value:.4f}"
                )
                bounded_value = max_step_value
        elif bounded_value > cfg["max"]:
            notes.append(f"{key.upper()} 超出上限，已裁剪到 {cfg['max']:.4f}")

        if bounded_value != raw_value and not any(note.startswith(key.upper()) for note in notes):
            notes.append(f"{key.upper()} 已从 {raw_value:.4f} 调整到 {bounded_value:.4f}")

        sanitized[key] = bounded_value

    return sanitized, notes


def apply_yaw_guardrails(
    current_yaw_pid: Dict[str, float],
    candidate: Dict[str, Any],
    *,
    max_increase_ratio: float = 1.2,
) -> Tuple[Dict[str, float], List[str]]:
    """X/Y 调参时对 YAW hold 环做更严的单轮步长限制。"""
    limits = get_pid_limits("hardware_yaw")
    for key in PID_KEYS:
        limits[key]["max_increase_ratio"] = max(1.0, float(max_increase_ratio))
    return apply_pid_guardrails(current_yaw_pid, candidate, limits=limits)


def extract_yaw_pid(payload: Dict[str, Any], current_yaw_pid: Dict[str, float]) -> Dict[str, float]:
    """从 LLM 结果中取出 yaw_p/yaw_i/yaw_d 或嵌套 yaw_pid；缺省保持当前值。"""
    source: Dict[str, Any] = {}
    nested = payload.get("yaw_pid")
    if isinstance(nested, dict):
        source.update(nested)
    for key, name in (("p", "yaw_p"), ("i", "yaw_i"), ("d", "yaw_d")):
        if name in payload:
            source[key] = payload[name]
    if not source:
        return {
            "p": float(current_yaw_pid.get("p", 0.0)),
            "i": float(current_yaw_pid.get("i", 0.0)),
            "d": float(current_yaw_pid.get("d", 0.0)),
        }
    return {
        "p": _to_float(source.get("p", current_yaw_pid.get("p", 0.0)), float(current_yaw_pid.get("p", 0.0))),
        "i": _to_float(source.get("i", current_yaw_pid.get("i", 0.0)), float(current_yaw_pid.get("i", 0.0))),
        "d": _to_float(source.get("d", current_yaw_pid.get("d", 0.0)), float(current_yaw_pid.get("d", 0.0))),
    }


def build_fallback_suggestion(
    current_pid: Dict[str, float],
    metrics: Dict[str, float],
    limits: Dict[str, Dict[str, float]] | None = None,
) -> Dict[str, Any]:
    """当 LLM 不可用时，用保守规则生成一个兜底建议。"""
    proposal = {
        "p": current_pid.get("p", 1.0),
        "i": current_pid.get("i", 0.1),
        "d": current_pid.get("d", 0.05),
    }

    status             = str(metrics.get("status", "UNKNOWN")).upper()
    overshoot          = float(metrics.get("overshoot", 0.0) or 0.0)
    steady_state_error = float(metrics.get("steady_state_error", 0.0) or 0.0)
    avg_error          = float(metrics.get("avg_error", 0.0) or 0.0)

    if status == "OSCILLATING":
        proposal["p"] *= 0.80
        proposal["i"] *= 0.85
        proposal["d"] *= 1.20
        action         = "DAMP_OSCILLATION"
        summary        = "检测到震荡，保守降低 P/I 并增加 D。"
    elif status == "OVERSHOOTING" or overshoot > 5.0:
        proposal["p"] *= 0.85
        proposal["i"] *= 0.90
        proposal["d"] *= 1.15
        action         = "REDUCE_OVERSHOOT"
        summary        = "检测到超调，优先降低 P/I 并增加阻尼。"
    elif status == "SLOW_RESPONSE":
        proposal["p"] *= 1.25
        if steady_state_error > max(1.0, avg_error * 0.5):
            proposal["i"] *= 1.20
        action         = "BOOST_RESPONSE"
        summary        = "响应偏慢，适度增加 P，并在稳态误差偏大时增加 I。"
    elif steady_state_error > 1.0:
        proposal["i"] *= 1.15
        action         = "REDUCE_STEADY_ERROR"
        summary        = "系统基本稳定但仍有稳态误差，微增 I。"
    else:
        proposal["p"] *= 1.05
        proposal["i"] *= 1.05
        action         = "FINE_TUNE"
        summary        = "进入细调阶段，做小步修正。"

    safe_pid, notes = apply_pid_guardrails(current_pid, proposal, limits=limits)

    return {
        "analysis_summary": f"LLM 不可用，已启用保守兜底：{summary}",
        "thought_process" : "基于控制基础规则生成保守建议，仅用于兜底，不替代正常 LLM 调参。",
        "tuning_action"   : action,
        "p"               : safe_pid["p"],
        "i"               : safe_pid["i"],
        "d"               : safe_pid["d"],
        "status"          : "TUNING",
        "guardrail_notes" : notes,
        "fallback_used"   : True,
    }


def pid_equals(left: Dict[str, float], right: Dict[str, float], tolerance: float = 1e-9) -> bool:
    return all(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) <= tolerance for key in PID_KEYS)


def yaw_coupling_penalty(metrics: Dict[str, float], tune_axis: str | None = None) -> float:
    """X/Y 调参时的航向代价；YAW 自调参或无 yaw 数据时为 0。"""
    axis = str(tune_axis or metrics.get("tune_axis") or "").strip().upper()
    if axis not in {"X", "Y"}:
        return 0.0

    yaw_peak = abs(float(metrics.get("yaw_delta_peak_deg", 0.0) or 0.0))
    sat_ratio = float(metrics.get("hold_yaw_saturated_ratio", 0.0) or 0.0)
    budget = float(CONFIG.get("HARDWARE_YAW_SOFT_BUDGET_DEG", 5.0) or 5.0)
    peak_excess = max(0.0, yaw_peak - budget)
    # 3° 超预算约等价于 9mm 误差；hold 饱和比满量程约等价于 10 分。
    return 3.0 * peak_excess + 10.0 * max(0.0, min(1.0, sat_ratio))


def score_metrics(metrics: Dict[str, float], tune_axis: str | None = None) -> float:
    """将控制表现压缩成一个可比较的分数，越低越好。"""
    avg_error          = _to_float(metrics.get("avg_error"), 1e9)
    steady_state_error = _to_float(metrics.get("steady_state_error"), 1e9)
    overshoot          = _to_float(metrics.get("overshoot"), 1e9)
    status             = str(metrics.get("status", "UNKNOWN")).upper()

    status_penalty = 0.0
    if status == "OVERSHOOTING":
        status_penalty = 8.0
    elif status == "OSCILLATING":
        status_penalty = 12.0
    elif status != "STABLE":
        status_penalty = 20.0

    return (
        avg_error
        + steady_state_error * 1.2
        + overshoot * 0.6
        + status_penalty
        + yaw_coupling_penalty(metrics, tune_axis=tune_axis)
    )


def is_better_metrics(candidate: Dict[str, float], baseline: Dict[str, float], epsilon: float = 1e-6) -> bool:
    axis = candidate.get("tune_axis") or baseline.get("tune_axis")
    return score_metrics(candidate, tune_axis=axis) + epsilon < score_metrics(baseline, tune_axis=axis)


def maybe_update_best_result(
    best_result: Dict[str, Any] | None,
    pid        : Dict[str, float],
    metrics    : Dict[str, float],
    round_num  : int,
    yaw_pid    : Dict[str, float] | None = None,
) -> Dict[str, Any] | None:
    """只记录稳定状态下的最佳 PID，避免回滚到坏参数。"""
    if str(metrics.get("status", "UNKNOWN")).upper() != "STABLE":
        return best_result

    candidate = {
        "round"  : round_num,
        "pid"    : {key: float(pid.get(key, 0.0)) for key in PID_KEYS},
        "metrics": dict(metrics),
    }
    if yaw_pid is not None:
        candidate["yaw_pid"] = {key: float(yaw_pid.get(key, 0.0)) for key in PID_KEYS}

    if best_result is None or is_better_metrics(candidate["metrics"], best_result["metrics"]):
        return candidate
    return best_result


def is_good_enough(metrics: Dict[str, float], rules: Dict[str, float] | None = None) -> bool:
    """判断系统是否已经达到“用户可接受”的稳定状态。"""
    rules              = rules or DEFAULT_CONVERGENCE_RULES
    status             = str(metrics.get("status", "UNKNOWN")).upper()
    # Hardware acceptance includes TARGET and axis/yaw constraints. Transient
    # averages must not veto a completed point-to-point move.
    if "hardware_accepted" in metrics:
        return status == "STABLE" and metrics["hardware_accepted"] is True
    avg_error          = _to_float(metrics.get("avg_error"), float("inf"))
    steady_state_error = _to_float(metrics.get("steady_state_error"), float("inf"))
    overshoot          = _to_float(metrics.get("overshoot"), float("inf"))

    return (
        status == "STABLE"
        and avg_error          <= rules["avg_error_threshold"]
        and steady_state_error <= rules["steady_state_error_threshold"]
        and overshoot          <= rules["overshoot_threshold"]
    )


def should_rollback_to_best(
    current_metrics: Dict[str, float],
    best_metrics   : Dict[str, float],
    rules          : Dict[str, float] | None = None,
) -> bool:
    """当前表现明显劣化时，建议回滚到历史最佳稳定参数。"""
    rules = rules or DEFAULT_ROLLBACK_RULES

    if not is_better_metrics(best_metrics, current_metrics):
        return False

    current_status = str(current_metrics.get("status", "UNKNOWN")).upper()
    best_status = str(best_metrics.get("status", "UNKNOWN")).upper()
    if best_status == "STABLE" and current_status != "STABLE":
        return True

    current_avg       = _to_float(current_metrics.get("avg_error"), 1e9)
    best_avg          = _to_float(best_metrics.get("avg_error"), 1e9)
    current_steady    = _to_float(current_metrics.get("steady_state_error"), 1e9)
    best_steady       = _to_float(best_metrics.get("steady_state_error"), 1e9)
    current_overshoot = _to_float(current_metrics.get("overshoot"), 1e9)
    best_overshoot    = _to_float(best_metrics.get("overshoot"), 1e9)

    avg_regression    = current_avg > max(best_avg * rules["avg_error_ratio"], best_avg + rules["avg_error_margin"])
    steady_regression = current_steady > max(
        best_steady * rules["steady_state_error_ratio"],
        best_steady + rules["steady_state_error_margin"],
    )
    overshoot_regression = current_overshoot > best_overshoot + rules["overshoot_margin"]

    return avg_regression or steady_regression or overshoot_regression
