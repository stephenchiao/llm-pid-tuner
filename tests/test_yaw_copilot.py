#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""X/Y 调参航向协同（方案 A+B）单元测试。"""

from __future__ import annotations

import unittest

from core.buffer import AdvancedDataBuffer
from core.config import CONFIG
from core.tuning_session import (
    DecisionOutcome,
    RoundEvaluation,
    create_tuning_session,
    evaluate_completed_round,
    finalize_decision,
)
from llm.client import LLMTuner
from pid_safety import (
    apply_yaw_guardrails,
    extract_yaw_pid,
    maybe_update_best_result,
    score_metrics,
    yaw_coupling_penalty,
)
from tuner import (
    _hardware_validation_passed,
    _yaw_adjust_allowed,
)


class YawPromptAndScoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {
            key: CONFIG.get(key)
            for key in (
                "HARDWARE_YAW_COPILOT",
                "HARDWARE_YAW_SOFT_BUDGET_DEG",
                "HARDWARE_YAW_VERIFY_LIMIT_DEG",
                "HARDWARE_YAW_HOLD_LIMIT_RADPS",
                "HARDWARE_YAW_HOLD_SAT_RATIO_MAX",
                "HARDWARE_YAW_ADJUST_SAT_RATIO",
            )
        }
        CONFIG["HARDWARE_YAW_COPILOT"] = True
        CONFIG["HARDWARE_YAW_SOFT_BUDGET_DEG"] = 5.0
        CONFIG["HARDWARE_YAW_VERIFY_LIMIT_DEG"] = 8.0
        CONFIG["HARDWARE_YAW_HOLD_LIMIT_RADPS"] = 0.15
        CONFIG["HARDWARE_YAW_HOLD_SAT_RATIO_MAX"] = 0.5
        CONFIG["HARDWARE_YAW_ADJUST_SAT_RATIO"] = 0.25

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                CONFIG.pop(key, None)
            else:
                CONFIG[key] = value

    def _buffer_with_yaw(self, peak: float, sat_ratio: float) -> AdvancedDataBuffer:
        buffer = AdvancedDataBuffer(max_size=20)
        hold = 0.15 if sat_ratio > 0 else 0.0
        for index in range(8):
            buffer.add(
                {
                    "timestamp": float(index * 20),
                    "setpoint": 200.0,
                    "input": 198.0 + 0.2 * index,
                    "pwm": 0.05,
                    "error": 2.0 - 0.2 * index,
                    "yaw_delta": peak if index == 7 else peak * 0.3,
                    "cross_track": 1.0,
                    "hold_yaw_output": hold,
                }
            )
        return buffer

    def test_to_prompt_data_includes_yaw_block_for_x(self) -> None:
        buffer = self._buffer_with_yaw(peak=9.0, sat_ratio=1.0)
        text = buffer.to_prompt_data(
            tune_axis="X",
            current_yaw_pid={"p": 0.02, "i": 0.000015, "d": 0.0},
        )
        self.assertIn("航向与耦合", text)
        self.assertIn("yaw_delta_peak_deg", text)
        self.assertIn("hold_yaw_saturated_ratio", text)
        self.assertIn("current_yaw_pid", text)
        self.assertIn("禁止提高主轴 P", text)

    def test_score_metrics_penalizes_yaw_for_x_only(self) -> None:
        clean = {
            "avg_error": 50.0,
            "steady_state_error": 5.0,
            "overshoot": 0.0,
            "status": "STABLE",
            "tune_axis": "X",
            "yaw_delta_peak_deg": 2.0,
            "hold_yaw_saturated_ratio": 0.0,
        }
        bad = dict(clean)
        bad["yaw_delta_peak_deg"] = 14.0
        bad["hold_yaw_saturated_ratio"] = 0.8
        self.assertLess(score_metrics(clean), score_metrics(bad))

        yaw_axis = dict(bad)
        yaw_axis["tune_axis"] = "YAW"
        self.assertEqual(yaw_coupling_penalty(yaw_axis), 0.0)

    def test_validation_requires_yaw_budget(self) -> None:
        metrics = {
            "overshoot": 0.0,
            "current_error": 2.0,
            "zero_crossings": 0,
            "yaw_delta_peak_deg": 12.0,
            "hold_yaw_saturated_ratio": 0.1,
        }
        self.assertFalse(_hardware_validation_passed(metrics, "TARGET", "X"))
        metrics["yaw_delta_peak_deg"] = 4.0
        self.assertTrue(_hardware_validation_passed(metrics, "TARGET", "X"))
        metrics["hold_yaw_saturated_ratio"] = 0.9
        self.assertFalse(_hardware_validation_passed(metrics, "TARGET", "Y"))

    def test_yaw_adjust_only_when_budget_exceeded(self) -> None:
        quiet = {
            "yaw_delta_peak_deg": 2.0,
            "hold_yaw_saturated_ratio": 0.0,
        }
        hot = {
            "yaw_delta_peak_deg": 9.0,
            "hold_yaw_saturated_ratio": 0.1,
        }
        self.assertFalse(_yaw_adjust_allowed(quiet, "X", "P"))
        self.assertTrue(_yaw_adjust_allowed(hot, "X", "P"))
        self.assertFalse(_yaw_adjust_allowed(hot, "X", "VERIFY"))
        self.assertFalse(_yaw_adjust_allowed(hot, "YAW", "P"))

    def test_yaw_guardrail_limits_step(self) -> None:
        current = {"p": 0.02, "i": 0.0, "d": 0.0}
        safe, notes = apply_yaw_guardrails(current, {"p": 0.05, "i": 0.0, "d": 0.0})
        self.assertLessEqual(safe["p"], 0.02 * 1.2 + 1e-12)
        self.assertTrue(notes)

    def test_extract_yaw_pid_defaults_to_current(self) -> None:
        current = {"p": 0.02, "i": 0.000015, "d": 0.0}
        self.assertEqual(extract_yaw_pid({"p": 0.004}, current), current)
        extracted = extract_yaw_pid({"yaw_p": 0.022, "yaw_i": 0.0, "yaw_d": 0.0}, current)
        self.assertAlmostEqual(extracted["p"], 0.022)

    def test_best_result_prefers_lower_yaw_when_scores_close(self) -> None:
        base = {
            "avg_error": 80.0,
            "steady_state_error": 8.0,
            "overshoot": 0.0,
            "status": "STABLE",
            "tune_axis": "X",
        }
        calm = dict(base, yaw_delta_peak_deg=2.0, hold_yaw_saturated_ratio=0.0)
        twist = dict(base, yaw_delta_peak_deg=11.0, hold_yaw_saturated_ratio=0.4)
        best = maybe_update_best_result(
            None,
            {"p": 0.004, "i": 0.0, "d": 0.0},
            twist,
            1,
            yaw_pid={"p": 0.02, "i": 0.0, "d": 0.0},
        )
        best = maybe_update_best_result(
            best,
            {"p": 0.004, "i": 0.0, "d": 0.0},
            calm,
            2,
            yaw_pid={"p": 0.021, "i": 0.0, "d": 0.0},
        )
        self.assertIsNotNone(best)
        self.assertEqual(best["round"], 2)
        self.assertEqual(best["yaw_pid"]["p"], 0.021)

    def test_finalize_decision_applies_yaw_when_allowed(self) -> None:
        state = create_tuning_session(
            initial_pid={"p": 0.004, "i": 0.0, "d": 0.0},
            setpoint=200.0,
            initial_yaw_pid={"p": 0.02, "i": 0.0, "d": 0.0},
        )
        evaluation = RoundEvaluation(
            round_index=1,
            metrics={
                "avg_error": 80.0,
                "steady_state_error": 8.0,
                "overshoot": 0.0,
                "status": "STABLE",
                "tune_axis": "X",
                "yaw_delta_peak_deg": 10.0,
            },
            current_pid={"p": 0.004, "i": 0.0, "d": 0.0},
            stable_rounds=0,
        )
        decision = finalize_decision(
            state,
            evaluation,
            {
                "analysis_summary": "raise yaw hold",
                "thought_process": "yaw over budget",
                "tuning_action": "ADJUST_PID",
                "p": 0.004,
                "i": 0.0,
                "d": 0.0,
                "yaw_p": 0.03,
                "yaw_i": 0.0,
                "yaw_d": 0.0,
                "status": "TUNING",
            },
            allow_yaw_adjust=True,
            current_yaw_pid={"p": 0.02, "i": 0.0, "d": 0.0},
        )
        self.assertIsInstance(decision, DecisionOutcome)
        self.assertIsNotNone(decision.safe_yaw_pid)
        self.assertLessEqual(decision.safe_yaw_pid["p"], 0.02 * 1.2 + 1e-12)
        self.assertEqual(state.current_yaw_pid["p"], decision.safe_yaw_pid["p"])

    def test_finalize_decision_freezes_yaw_when_not_allowed(self) -> None:
        state = create_tuning_session(
            initial_pid={"p": 0.004, "i": 0.0, "d": 0.0},
            setpoint=200.0,
            initial_yaw_pid={"p": 0.02, "i": 0.0, "d": 0.0},
        )
        evaluation = RoundEvaluation(
            round_index=1,
            metrics={
                "avg_error": 80.0,
                "steady_state_error": 8.0,
                "overshoot": 0.0,
                "status": "STABLE",
                "tune_axis": "X",
                "yaw_delta_peak_deg": 1.0,
            },
            current_pid={"p": 0.004, "i": 0.0, "d": 0.0},
            stable_rounds=0,
        )
        decision = finalize_decision(
            state,
            evaluation,
            {
                "analysis_summary": "try yaw anyway",
                "thought_process": "ignore budget",
                "tuning_action": "ADJUST_PID",
                "p": 0.004,
                "i": 0.0,
                "d": 0.0,
                "yaw_p": 0.03,
                "yaw_i": 0.0,
                "yaw_d": 0.0,
                "status": "TUNING",
            },
            allow_yaw_adjust=False,
            current_yaw_pid={"p": 0.02, "i": 0.0, "d": 0.0},
        )
        self.assertAlmostEqual(decision.safe_yaw_pid["p"], 0.02)

    def test_client_sanitize_keeps_yaw_fields(self) -> None:
        tuner = LLMTuner.__new__(LLMTuner)
        cleaned = tuner._sanitize_result(
            {
                "p": 0.004,
                "i": 0.0,
                "d": 0.0,
                "yaw_p": 0.022,
                "yaw_i": -1.0,
                "yaw_d": 0.0,
                "status": "TUNING",
                "analysis_summary": "ok",
                "thought_process": "ok",
                "tuning_action": "ADJUST_PID",
            }
        )
        self.assertEqual(cleaned["yaw_p"], 0.022)
        self.assertNotIn("yaw_i", cleaned)

    def test_evaluate_completed_round_records_yaw_pid(self) -> None:
        state = create_tuning_session(
            initial_pid={"p": 0.004, "i": 0.0, "d": 0.0},
            setpoint=200.0,
            initial_yaw_pid={"p": 0.02, "i": 0.0, "d": 0.0},
        )
        buffer = self._buffer_with_yaw(peak=2.0, sat_ratio=0.0)
        state.buffer = buffer
        evaluation = evaluate_completed_round(
            state,
            {"p": 0.004, "i": 0.0, "d": 0.0},
            tune_axis="X",
            current_yaw_pid={"p": 0.02, "i": 0.0, "d": 0.0},
        )
        self.assertEqual(evaluation.metrics.get("tune_axis"), "X")
        self.assertEqual(evaluation.metrics.get("status"), "STABLE")
        self.assertIsNotNone(state.best_result)
        self.assertEqual(state.best_result.get("yaw_pid", {}).get("p"), 0.02)


if __name__ == "__main__":
    unittest.main()
