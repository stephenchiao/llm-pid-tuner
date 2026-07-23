import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pid_results import append_pid_result, load_last_usable_pid
from pid_safety import get_pid_limits
from tuner import (
    _build_hardware_prompt_context,
    _freeze_pid_terms_for_stage,
    _hardware_validation_passed,
    _run_post_tune_motion_tests,
)


class HardwareStageTests(unittest.TestCase):
    def test_each_stage_freezes_other_pid_terms(self):
        current = {"p": 0.0045, "i": 0.00001, "d": 0.0002}
        proposal = {"p": 0.005, "i": 0.00002, "d": 0.0004, "status": "TUNING"}

        self.assertEqual(_freeze_pid_terms_for_stage(proposal, current, "P")["i"], current["i"])
        self.assertEqual(_freeze_pid_terms_for_stage(proposal, current, "P")["d"], current["d"])
        self.assertEqual(_freeze_pid_terms_for_stage(proposal, current, "I")["p"], current["p"])
        self.assertEqual(_freeze_pid_terms_for_stage(proposal, current, "I")["d"], current["d"])
        self.assertEqual(_freeze_pid_terms_for_stage(proposal, current, "D")["p"], current["p"])
        self.assertEqual(_freeze_pid_terms_for_stage(proposal, current, "D")["i"], current["i"])

    def test_prompt_context_explains_current_stage(self):
        context = _build_hardware_prompt_context("COM3", "I")
        self.assertEqual(context["tuning_stage"], "I")
        self.assertEqual(context["adjustable_terms"], "I")
        self.assertEqual(context["frozen_terms"], "P,D")
        self.assertEqual(context["controller_output_limit"], 0.2)
        self.assertEqual(context["tune_axis"], "Y")

    def test_yaw_prompt_context_uses_angle_units(self):
        context = _build_hardware_prompt_context("COM3", "P", 0.25, "YAW")
        self.assertEqual(context["controller_input_unit"], "degree")
        self.assertEqual(context["controller_output_unit"], "rad/s")
        self.assertEqual(context["target_tolerance"], 1.0)

    def test_validation_requires_target_and_safe_metrics(self):
        metrics = {
            "overshoot": 1.0,
            # 最后20%窗口包含减速段时可能略大，但TARGET已经保证连续到位。
            "steady_state_error": 12.0,
            "current_error": 1.0,
            "zero_crossings": 2,
        }
        self.assertTrue(_hardware_validation_passed(metrics, "TARGET"))
        self.assertFalse(_hardware_validation_passed(metrics, "TIMEOUT"))

    def test_post_tune_motion_tests_cover_left_right_ccw_cw(self):
        class FakeBridge:
            def __init__(self):
                self.pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
                self.lines = []

            def send_silent_command(self, _cmd):
                return None

            def send_command(self, cmd):
                if cmd == "OPS STATUS":
                    self.lines.append(
                        f"# OPS LINK=OK X={self.pose['x']} Y={self.pose['y']} "
                        f"YAW={self.pose['yaw']} BYTES=1"
                    )
                elif cmd.startswith("MOVE LEFT"):
                    self.pose["x"] -= 18.0
                    self.lines.append("# MOVE AUTO STOP")
                elif cmd.startswith("MOVE RIGHT"):
                    self.pose["x"] += 18.0
                    self.lines.append("# MOVE AUTO STOP")
                elif cmd.startswith("TURN CCW"):
                    self.pose["yaw"] += 5.0
                    self.lines.append("# MOVE AUTO STOP")
                elif cmd.startswith("TURN CW"):
                    self.pose["yaw"] -= 5.0
                    self.lines.append("# MOVE AUTO STOP")

            def read_line(self):
                return self.lines.pop(0) if self.lines else None

        with patch("tuner.time.sleep", return_value=None):
            results = _run_post_tune_motion_tests(FakeBridge(), emit_console=False)

        self.assertEqual([item["action"] for item in results], ["LEFT", "RIGHT", "CCW", "CW"])
        self.assertTrue(all(item["passed"] for item in results))


class PidResultStoreTests(unittest.TestCase):
    def test_append_pid_result_keeps_previous_sessions(self):
        result = {
            "provider": "openai",
            "model": "demo",
            "rounds_completed": 12,
            "completed_reason": "staged_validation_passed",
            "final_pid": {"p": 0.0045, "i": 0.0, "d": 0.0},
            "final_metrics": {"overshoot": 1.7},
            "motion_tests": [{"action": "LEFT", "passed": True}],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pid_results.jsonl"
            append_pid_result(result, str(path))
            append_pid_result(result, str(path))
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(records), 2)
        self.assertEqual(records[-1]["final_pid"]["p"], 0.0045)
        self.assertEqual(records[-1]["completed_reason"], "staged_validation_passed")
        self.assertTrue(records[-1]["motion_tests"][0]["passed"])

    def test_load_last_usable_pid_skips_interrupted_newer_record(self):
        records = [
            {
                "completed_reason": "max_rounds_reached",
                "final_pid": {"p": 0.004, "i": 0.0, "d": 0.0},
                "final_metrics": {
                    "current_error": 3.8,
                    "overshoot": 1.9,
                    "zero_crossings": 1,
                },
            },
            {
                "completed_reason": "keyboard_interrupt",
                "final_pid": {"p": 0.001, "i": 0.0, "d": 0.0},
                "final_metrics": {"current_error": 1.0, "overshoot": 0.0},
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pid_results.jsonl"
            path.write_text(
                "\n".join(json.dumps(item) for item in records), encoding="utf-8"
            )
            pid = load_last_usable_pid(str(path), get_pid_limits("hardware"))

        self.assertEqual(pid, {"p": 0.004, "i": 0.0, "d": 0.0})

    def test_pid_history_is_isolated_by_axis(self):
        records = [
            {
                "tune_axis": "Y",
                "completed_reason": "staged_validation_passed",
                "final_pid": {"p": 0.0018, "i": 0.0, "d": 0.0},
            },
            {
                "tune_axis": "YAW",
                "completed_reason": "staged_validation_passed",
                "final_pid": {"p": 0.02, "i": 0.0, "d": 0.0},
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pid_results.jsonl"
            path.write_text(
                "\n".join(json.dumps(item) for item in records), encoding="utf-8"
            )
            pid = load_last_usable_pid(
                str(path), get_pid_limits("hardware"), "Y"
            )
        self.assertEqual(pid["p"], 0.0018)


if __name__ == "__main__":
    unittest.main()
