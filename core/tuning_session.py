from __future__ import annotations

from dataclasses import field
from typing import Any

from core.buffer import AdvancedDataBuffer
from core.compat import slotted_dataclass
from core.config import CONFIG
from core.history import TuningHistory
from pid_safety import (
    apply_pid_guardrails,
    apply_yaw_guardrails,
    build_fallback_suggestion,
    extract_yaw_pid,
    is_good_enough,
    maybe_update_best_result,
    pid_equals,
    should_rollback_to_best,
)


@slotted_dataclass
class TuningSessionState:
    buffer: AdvancedDataBuffer
    history: TuningHistory
    good_enough_rules: dict[str, float]
    round_num: int = 0
    last_round: int = 0
    stable_rounds: int = 0
    best_result: dict[str, Any] | None = None
    last_metrics: dict[str, Any] = field(default_factory=dict)
    completed_reason: str = "max_rounds_reached"
    fallback_count: int = 0
    guardrail_count: int = 0
    rollback_count: int = 0
    current_yaw_pid: dict[str, float] | None = None


@slotted_dataclass
class RoundEvaluation:
    round_index: int
    metrics: dict[str, Any]
    current_pid: dict[str, float]
    stable_rounds: int
    best_result: dict[str, Any] | None = None
    best_result_updated: bool = False
    rollback_pid: dict[str, float] | None = None
    rollback_yaw_pid: dict[str, float] | None = None
    completed_reason: str | None = None


@slotted_dataclass
class DecisionOutcome:
    safe_pid: dict[str, float]
    action: str
    analysis: str
    thought: str
    guardrail_notes: list[str]
    fallback_used: bool
    status: str
    completed_reason: str | None = None
    safe_yaw_pid: dict[str, float] | None = None


def create_tuning_session(
    *,
    initial_pid: dict[str, float] | None = None,
    setpoint: float | None = None,
    max_history: int = 5,
    buffer_size: int | None = None,
    initial_yaw_pid: dict[str, float] | None = None,
) -> TuningSessionState:
    buffer = AdvancedDataBuffer(
        max_size=int(buffer_size if buffer_size is not None else CONFIG["BUFFER_SIZE"])
    )
    if initial_pid is not None:
        buffer.current_pid = dict(initial_pid)
    if setpoint is not None:
        buffer.setpoint = float(setpoint)

    return TuningSessionState(
        buffer=buffer,
        history=TuningHistory(max_history=max_history),
        good_enough_rules={
            "avg_error_threshold": CONFIG["GOOD_ENOUGH_AVG_ERROR"],
            "steady_state_error_threshold": CONFIG["GOOD_ENOUGH_STEADY_STATE_ERROR"],
            "overshoot_threshold": CONFIG["GOOD_ENOUGH_OVERSHOOT"],
        },
        current_yaw_pid=dict(initial_yaw_pid) if initial_yaw_pid is not None else None,
    )


def evaluate_completed_round(
    state: TuningSessionState,
    current_pid: dict[str, float],
    *,
    tune_axis: str | None = None,
    current_yaw_pid: dict[str, float] | None = None,
    round_metrics: dict[str, Any] | None = None,
) -> RoundEvaluation:
    metrics = (dict(round_metrics) if round_metrics is not None
               else state.buffer.calculate_advanced_metrics(tune_axis=tune_axis))
    round_index = state.round_num + 1
    # 检测重试：同一轮因 pause 被中断后重新进入，last_round 已等于 round_index
    is_retry = state.last_round == round_index
    state.last_round = round_index
    state.last_metrics = dict(metrics)

    if is_retry:
        # 重试时不累加 stable_rounds，保留上次结果，避免同一批数据重复计分
        pass
    elif is_good_enough(metrics, state.good_enough_rules):
        state.stable_rounds += 1
    else:
        state.stable_rounds = 0

    previous_best = state.best_result
    state.best_result = maybe_update_best_result(
        state.best_result,
        current_pid,
        metrics,
        round_index,
        yaw_pid=current_yaw_pid if current_yaw_pid is not None else state.current_yaw_pid,
    )
    best_result_updated = (
        state.best_result is not None and state.best_result is not previous_best
    )

    rollback_pid: dict[str, float] | None = None
    rollback_yaw_pid: dict[str, float] | None = None
    completed_reason: str | None = None
    active_yaw = current_yaw_pid if current_yaw_pid is not None else state.current_yaw_pid
    yaw_differs = bool(
        state.best_result
        and state.best_result.get("yaw_pid")
        and active_yaw
        and not pid_equals(active_yaw, state.best_result["yaw_pid"])
    )
    if (
        state.best_result
        and (not pid_equals(current_pid, state.best_result["pid"]) or yaw_differs)
        and should_rollback_to_best(metrics, state.best_result["metrics"])
    ):
        rollback_pid = dict(state.best_result["pid"])
        if state.best_result.get("yaw_pid"):
            rollback_yaw_pid = dict(state.best_result["yaw_pid"])
        if is_good_enough(state.best_result["metrics"], state.good_enough_rules):
            completed_reason = "rollback_to_best"
    elif (
        metrics["avg_error"] < CONFIG["MIN_ERROR_THRESHOLD"]
        and metrics["status"] == "STABLE"
    ):
        completed_reason = "low_error_converged"
    elif state.stable_rounds >= CONFIG["REQUIRED_STABLE_ROUNDS"]:
        completed_reason = "stable_rounds_reached"

    return RoundEvaluation(
        round_index=round_index,
        metrics=metrics,
        current_pid=dict(current_pid),
        stable_rounds=state.stable_rounds,
        best_result=state.best_result,
        best_result_updated=best_result_updated,
        rollback_pid=rollback_pid,
        rollback_yaw_pid=rollback_yaw_pid,
        completed_reason=completed_reason,
    )


def apply_rollback(
    state: TuningSessionState,
    rollback_pid: dict[str, float],
    rollback_yaw_pid: dict[str, float] | None = None,
) -> None:
    state.rollback_count += 1
    state.round_num += 1
    state.buffer.current_pid = dict(rollback_pid)
    if rollback_yaw_pid is not None:
        state.current_yaw_pid = dict(rollback_yaw_pid)
    state.buffer.reset()


def record_rollback_round(
    state: TuningSessionState,
    evaluation: RoundEvaluation,
    rollback_pid: dict[str, float],
    *,
    target_round: int | None = None,
    rollback_yaw_pid: dict[str, float] | None = None,
) -> str:
    target_label = (
        f"round {target_round}" if target_round is not None else "the best stable round"
    )
    analysis = (
        "Automatic rollback triggered because this round regressed against "
        f"{target_label}. Reverted to "
        f"P={rollback_pid['p']:.4f}, I={rollback_pid['i']:.4f}, D={rollback_pid['d']:.4f}."
    )
    if rollback_yaw_pid is not None:
        analysis += (
            f" YAW hold restored to P={rollback_yaw_pid['p']:.4f}, "
            f"I={rollback_yaw_pid['i']:.6f}, D={rollback_yaw_pid['d']:.4f}."
        )
    thought = (
        "This round was evaluated with "
        f"P={evaluation.current_pid['p']:.4f}, I={evaluation.current_pid['i']:.4f}, D={evaluation.current_pid['d']:.4f}. "
        "Its response was worse than the current best stable result, so the attempt was rejected."
    )
    state.history.add_record(
        evaluation.round_index,
        evaluation.current_pid,
        evaluation.metrics,
        analysis,
        thought,
    )
    return analysis


def finalize_decision(
    state: TuningSessionState,
    evaluation: RoundEvaluation,
    result: dict[str, Any] | None,
    *,
    limits: dict[str, dict[str, float]] | None = None,
    freeze_integral: bool = False,
    current_yaw_pid: dict[str, float] | None = None,
    allow_yaw_adjust: bool = False,
) -> DecisionOutcome:
    if not result:
        result = build_fallback_suggestion(
            evaluation.current_pid,
            evaluation.metrics,
            limits=limits,
        )

    safe_pid, guardrail_notes = apply_pid_guardrails(
        evaluation.current_pid,
        result,
        limits=limits,
    )
    if freeze_integral and safe_pid["i"] > evaluation.current_pid["i"]:
        safe_pid["i"] = evaluation.current_pid["i"]
        guardrail_notes.append("输出持续饱和，本轮冻结 I，禁止继续增加积分")
    analysis = str(result.get("analysis_summary", "No analysis summary was provided."))
    thought = str(result.get("thought_process", ""))
    action = str(result.get("tuning_action", "UNKNOWN"))
    fallback_used = bool(result.get("fallback_used"))
    status = str(result.get("status", "TUNING")).upper()

    state.history.add_record(
        evaluation.round_index,
        evaluation.current_pid,
        evaluation.metrics,
        analysis,
        thought,
    )
    state.buffer.current_pid = dict(safe_pid)

    yaw_baseline = dict(
        current_yaw_pid
        if current_yaw_pid is not None
        else (state.current_yaw_pid or {"p": 0.0, "i": 0.0, "d": 0.0})
    )
    if allow_yaw_adjust:
        proposed_yaw = extract_yaw_pid(result, yaw_baseline)
        safe_yaw_pid, yaw_notes = apply_yaw_guardrails(yaw_baseline, proposed_yaw)
        guardrail_notes.extend(yaw_notes)
    else:
        safe_yaw_pid = dict(yaw_baseline)
    state.current_yaw_pid = dict(safe_yaw_pid)

    if fallback_used:
        state.fallback_count += 1
    if guardrail_notes:
        state.guardrail_count += 1
    state.round_num += 1
    state.buffer.reset()

    completed_reason = "llm_marked_done" if status == "DONE" else None
    return DecisionOutcome(
        safe_pid=safe_pid,
        action=action,
        analysis=analysis,
        thought=thought,
        guardrail_notes=list(guardrail_notes),
        fallback_used=fallback_used,
        status=status,
        completed_reason=completed_reason,
        safe_yaw_pid=safe_yaw_pid,
    )


def build_tuning_result(
    state: TuningSessionState, *, final_pid: dict[str, float], stopped: bool
) -> dict[str, Any]:
    return {
        "provider": CONFIG["LLM_PROVIDER"],
        "model": CONFIG["LLM_MODEL_NAME"],
        "rounds_completed": state.last_round,
        "final_pid": dict(final_pid),
        "final_yaw_pid": dict(state.current_yaw_pid or {}),
        "final_metrics": dict(state.last_metrics),
        "stopped": stopped,
        "fallback_count": state.fallback_count,
        "guardrail_count": state.guardrail_count,
        "rollback_count": state.rollback_count,
        "completed_reason": state.completed_reason,
    }
