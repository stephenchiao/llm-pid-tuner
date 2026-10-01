"""Replay the 2026-10-01 X-axis baseline without serial or LLM access."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))
import tuner
from core.buffer import AdvancedDataBuffer
from core.config import CONFIG
from core.pid_results import append_pid_result
from core.tuning_session import create_tuning_session, evaluate_completed_round
from hw.bridge import SerialBridge
from llm.prompts import get_system_prompt
from pid_safety import is_good_enough, score_metrics, should_rollback_to_best

FIXTURE = Path(__file__).parent / "fixtures" / "x_p0035_target.csv"


def recorded_buffer():
    buffer = AdvancedDataBuffer()
    for line in FIXTURE.read_text().splitlines()[1:]:
        buffer.add(SerialBridge.parse_data(None, line))
    return buffer


class AcceptanceTests(unittest.TestCase):
    def test_recorded_target_is_accepted_and_saved_as_baseline(self):
        buffer = recorded_buffer()
        self.assertEqual(buffer.calculate_advanced_metrics()["status"], "SLOW_RESPONSE")
        metrics = tuner._evaluate_hardware_metrics(buffer, "X", .2, "TARGET")
        state = create_tuning_session()
        state.buffer = buffer
        result = evaluate_completed_round(state, buffer.current_pid, round_metrics=metrics)
        self.assertTrue(is_good_enough(metrics))
        self.assertEqual(result.stable_rounds, 1)
        self.assertEqual(result.best_result["pid"]["p"], .0035)
        self.assertEqual(metrics["first_in_tolerance_ms"], 1920)
        self.assertAlmostEqual(metrics["speed_saturation_ratio"], 7 / 34)

    def test_timeout_does_not_become_accepted_even_with_small_final_error(self):
        metrics = tuner._evaluate_hardware_metrics(recorded_buffer(), "X", .2, "TIMEOUT")
        self.assertFalse(is_good_enough(metrics))

    def test_yaw_constraint_prevents_saving_an_unsafe_baseline(self):
        buffer = recorded_buffer()
        buffer.buffer[-1]["yaw_delta"] = 12.
        with patch.dict(CONFIG, {"HARDWARE_YAW_COPILOT": True,
                                 "HARDWARE_YAW_VERIFY_LIMIT_DEG": 8.}):
            metrics = tuner._evaluate_hardware_metrics(buffer, "X", .2, "TARGET")
        state = create_tuning_session()
        state.buffer = buffer
        evaluation = evaluate_completed_round(state, buffer.current_pid, round_metrics=metrics)
        self.assertFalse(is_good_enough(metrics))
        self.assertIsNone(evaluation.best_result)

    def test_zero_is_valid_but_missing_and_nonfinite_metrics_are_not(self):
        good = dict(status="STABLE", avg_error=0., steady_state_error=0., overshoot=0.)
        self.assertTrue(is_good_enough(good))
        self.assertEqual(score_metrics(good), 0)
        self.assertTrue(should_rollback_to_best(dict(good, overshoot=2.), good))
        for value in (None, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.assertFalse(is_good_enough(dict(good, overshoot=value)))
                self.assertGreater(score_metrics(dict(good, overshoot=value)), 0)

    def test_prompt_preserves_output_precision_and_real_limits(self):
        prompt = recorded_buffer().to_prompt_data(tune_axis="X")
        self.assertIn("0.1560", prompt)
        context = tuner._build_hardware_prompt_context("COM9", tune_axis="X")
        self.assertEqual(context["pid_limits"]["p"]["max"], .005)
        self.assertEqual(context["pid_limits"]["p"]["max_increase_ratio"], 1.5)
        self.assertNotIn("3x", context["per_round_guardrail_hint"])
        self.assertNotIn("P 要大步探索", get_system_prompt("hardware"))

    def run_replay(self, *, stop_reasons=("TARGET",), initial_p=.0035,
                   proposal_p=.007, statuses=("TUNING",), max_rounds=8):
        commands, summaries, histories = [], [], []
        fixture_lines = FIXTURE.read_text().splitlines()[1:]

        class Bridge:
            last_error = ""
            def __init__(self, *args, **kwargs):
                self.lines = []
                self.round = 0
            def connect(self): return True
            def disconnect(self): pass
            def read_line(self): return self.lines.pop(0) if self.lines else None
            def parse_data(self, line): return SerialBridge.parse_data(self, line)
            def send_command(self, cmd):
                commands.append(cmd)
                if not cmd.startswith("SET P:"):
                    return
                p, i, d = [part.split(":")[1] for part in cmd.split()[1:]]
                self.round += 1
                self.lines = [f"# ROUND START {self.round} AXIS=X DIR {1 if self.round % 2 else -1}"]
                for line in fixture_lines:
                    parts = line.split(",")
                    parts[5:8] = [p, i, d]
                    self.lines.append(",".join(parts))
                reason = stop_reasons[min(self.round - 1, len(stop_reasons) - 1)]
                self.lines.append(f"# ROUND STOP {reason} AXIS=X")

        class Advisor:
            def __init__(self, *args, **kwargs): pass
            def analyze(self, data, history, **kwargs):
                histories.append(history)
                return dict(p=proposal_p, i=0., d=0.,
                            status=statuses[min(len(histories) - 1, len(statuses) - 1)])
            def summarize_tuning_session(self, payload):
                summaries.append(payload)
                return None

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(tuner, "SerialBridge", Bridge), \
             patch.object(tuner, "LLMTuner", Advisor), \
             patch.object(tuner.time, "sleep"), \
             patch.object(tuner, "resolve_project_path", side_effect=lambda _: Path(tmp)), \
             patch.dict(CONFIG, {"HARDWARE_TUNE_AXIS": "X", "HARDWARE_VERIFY_ROUNDS": 3,
                                 "HARDWARE_RESUME_LAST_PID": False, "BUFFER_SIZE": 100,
                                 "MAX_TUNING_ROUNDS": max_rounds}):
            result = tuner._run_hardware_tuning_loop(
                "COM9", emit_console=False, initial_pid=dict(p=initial_p, i=0., d=0.))
            log_path = append_pid_result(result, str(Path(tmp) / "result.jsonl"))
            saved = json.loads(log_path.read_text(encoding="utf-8"))
        return result, commands, summaries, histories, saved

    def test_accepted_baseline_is_verified_without_any_pid_proposals(self):
        result, commands, _, histories, _ = self.run_replay()
        self.assertEqual(histories, [])
        self.assertEqual(result["completed_reason"], "staged_validation_passed")
        self.assertEqual(result["verified_pid"], dict(p=.0035, i=0., d=0.))
        self.assertEqual(result["rounds_completed"], 3)
        self.assertEqual([c for c in commands if c.startswith("SET P:")],
                         ["SET P:0.0035 I:0.0 D:0.0"] * 3)

    def test_clipped_repeated_proposals_stop_without_success(self):
        result, commands, _, histories, _ = self.run_replay(
            stop_reasons=("TIMEOUT",), initial_p=.005)
        self.assertEqual(result["completed_reason"], "no_parameter_progress")
        self.assertIsNone(result["verified_pid"])
        self.assertEqual(len([c for c in commands if c.startswith("SET P:")]), 2)
        self.assertIn("已达上限", histories[1])

    def test_failure_during_verification_is_preserved_in_summary_and_log(self):
        result, _, summaries, histories, saved = self.run_replay(
            stop_reasons=("TARGET", "YAW LIMIT"))
        self.assertEqual(histories, [])
        self.assertEqual(result["completed_reason"], "hardware_stopped")
        self.assertIsNone(result["verified_pid"])
        self.assertEqual(result["rounds_completed"], 1)
        self.assertEqual(result["failed_round"]["round"], 2)
        self.assertIn("YAW LIMIT", summaries[0]["failure_detail"])
        self.assertIn("YAW LIMIT", saved["failed_round"]["stop_reason"])
        self.assertIn("YAW LIMIT", result["ai_summary"]["evaluation"])

    def test_round_limit_does_not_claim_three_passes_from_one_pass(self):
        result, commands, _, _, _ = self.run_replay(max_rounds=1)
        self.assertEqual(result["completed_reason"], "max_rounds_reached")
        self.assertIsNone(result["verified_pid"])
        self.assertEqual(len([c for c in commands if c.startswith("SET P:")]), 1)

    def test_failed_verification_resets_consecutive_pass_count(self):
        result, commands, _, _, _ = self.run_replay(
            stop_reasons=("TARGET", "TIMEOUT", "TARGET", "TARGET", "TARGET"))
        self.assertEqual(result["completed_reason"], "staged_validation_passed")
        self.assertEqual(result["rounds_completed"], 5)
        self.assertEqual(len([c for c in commands if c.startswith("SET P:")]), 5)


if __name__ == "__main__":
    unittest.main()
