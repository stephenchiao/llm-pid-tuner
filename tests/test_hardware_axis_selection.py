import unittest
from unittest.mock import Mock, patch

import tuner


def successful_result(axis):
    return {
        "tune_axis": axis,
        "completed_reason": "staged_validation_passed",
        "verified_pid": {"p": 0.003, "i": 0.00001, "d": 0.0},
        "stop_confirmation": "command_sent",
        "motion_tests": [],
    }


class AxisSelectionTests(unittest.TestCase):
    def test_single_axes_and_mixed_separators(self):
        for text, expected in (
            ("x", ["X"]), (" Y ", ["Y"]), ("yaw", ["YAW"]),
            ("X Y", ["X", "Y"]), ("yaw/x/Y", ["YAW", "X", "Y"]),
            ("Y，x、yaw；X", ["Y", "X", "YAW"]),
            ("X,X;YAW", ["X", "YAW"]),
        ):
            with self.subTest(text=text):
                self.assertEqual(tuner.parse_hardware_axes(text), expected)

    def test_invalid_or_empty_selection_never_starts_hardware(self):
        for value in ("", " / , ", "X Z", "XY", "yawx"):
            with self.subTest(value=value), patch.object(tuner, "run_hardware_tuner") as run:
                with self.assertRaises(SystemExit) as exc:
                    tuner.main(["COM3", "--plain", "--axes", value])
                self.assertEqual(exc.exception.code, 2)
                run.assert_not_called()

    def test_sequence_preserves_order_and_previous_verified_gains(self):
        with patch.object(tuner, "run_hardware_tuner", side_effect=lambda *a, **kw: successful_result(kw["tune_axis"])) as run:
            results = tuner.run_hardware_axis_sequence(["yaw", "x", "Y", "X"], "COM3", True)
        self.assertEqual([r["tune_axis"] for r in results], ["YAW", "X", "Y"])
        self.assertEqual([c.kwargs["tune_axis"] for c in run.call_args_list], ["YAW", "X", "Y"])
        self.assertEqual(run.call_args_list[0].kwargs["preserved_pids"], {})
        self.assertEqual(run.call_args_list[1].kwargs["preserved_pids"], {"YAW": results[0]["verified_pid"]})
        self.assertEqual(set(run.call_args_list[2].kwargs["preserved_pids"]), {"YAW", "X"})

    def test_single_selection_only_runs_that_axis(self):
        with patch.object(tuner, "run_hardware_tuner", return_value=successful_result("Y")) as run:
            tuner.main(["COM3", "--plain", "--axes", "y"])
        run.assert_called_once_with("COM3", force_plain=True, tune_axis="Y", preserved_pids={})

    def test_failure_interrupt_or_unconfirmed_stop_blocks_remaining_axes(self):
        failures = [
            {"completed_reason": reason}
            for reason in ("hardware_error", "keyboard_interrupt", "max_rounds_reached", "no_serial_port", "ops_not_ready", "stopped_by_user")
        ] + [
            {"stop_confirmation": "unconfirmed"},
            {"failure_detail": "cleanup failed"},
            {"motion_tests": [{"passed": False}]},
            {"verified_pid": None},
        ]
        for override in failures:
            with self.subTest(override=override):
                result = {**successful_result("X"), **override}
                with patch.object(tuner, "run_hardware_tuner", return_value=result) as run:
                    with self.assertRaises(SystemExit) as exc:
                        tuner.main(["COM3", "--axes", "X", "Y", "YAW"])
                    self.assertEqual(exc.exception.code, 1)
                self.assertEqual(run.call_count, 1)

    def test_no_axes_keeps_legacy_dispatch(self):
        with patch.object(tuner, "run_hardware_tuner") as run:
            tuner.main(["COM9", "--plain"])
        run.assert_called_once_with("COM9", force_plain=True)

    def test_selection_wins_after_config_reload_and_freezes_yaw_copilot(self):
        def reload_config(**kwargs):
            tuner.CONFIG.update(HARDWARE_TUNE_AXIS="X", HARDWARE_YAW_COPILOT=True, HARDWARE_RESUME_LAST_PID=False)

        def run_plain(*args, **kwargs):
            self.assertEqual(tuner.CONFIG["HARDWARE_TUNE_AXIS"], "Y")
            self.assertFalse(tuner.CONFIG["HARDWARE_YAW_COPILOT"])
            self.assertEqual(kwargs["preserved_pids"], saved)
            return successful_result("Y")

        saved = {"X": {"p": 0.004, "i": 0.0, "d": 0.0}}
        with patch.dict(tuner.CONFIG), patch.object(tuner, "initialize_runtime_config", side_effect=reload_config), \
             patch.object(tuner, "_run_hardware_tuning_plain", side_effect=run_plain), \
             patch.object(tuner, "append_pid_result"):
            tuner.run_hardware_tuner("COM3", True, tune_axis="y", preserved_pids=saved)

    def test_prior_axis_pid_is_loaded_even_without_history_resume(self):
        class Bridge:
            last_error = ""
            requires_hardware_preflight = False

            def __init__(self):
                self.commands = []

            def connect(self):
                return True

            def send_command(self, command):
                self.commands.append(command)

            def disconnect(self):
                pass

        for resume in (True, False):
            with self.subTest(resume=resume):
                bridge = Bridge()
                preserved = {"X": {"p": 0.007, "i": 0.00002, "d": 0.0}}
                with patch.dict(tuner.CONFIG, {"HARDWARE_TUNE_AXIS": "Y", "HARDWARE_RESUME_LAST_PID": resume}), \
                     patch.object(tuner, "SerialBridge", return_value=bridge), \
                     patch.object(tuner, "LLMTuner", return_value=Mock()), \
                     patch.object(tuner, "HardwareRoundCsvRecorder"), \
                     patch.object(tuner, "load_last_usable_pid", return_value=None) as load, \
                     patch.object(tuner, "_send_next_round_commands", side_effect=RuntimeError("test stop before motion")), \
                     patch.object(tuner, "_confirm_hardware_stop", return_value="command_sent"), \
                     patch.object(tuner, "_read_pid_snapshot", return_value={}), \
                     patch.object(tuner.time, "sleep"):
                    tuner._run_hardware_tuning_loop("COM3", emit_console=False, preserved_pids=preserved)
                self.assertIn("PID SET X 0.007 2e-05 0.0", bridge.commands)
                self.assertIn("TUNE AXIS Y", bridge.commands)
                self.assertNotIn("X", [call.args[2] for call in load.call_args_list])


if __name__ == "__main__":
    unittest.main()
