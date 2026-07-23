import math
import sys
import unittest
from pathlib import Path
from queue import Queue
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).parent.parent))

import tuner
from sim.runtime import (
    EVENT_DECISION,
    EVENT_LIFECYCLE,
    EVENT_LOG,
    EVENT_ROUND_METRICS,
    EVENT_SAMPLE,
    QueueEventSink,
    SimulationController,
    drain_event_queue,
)


def _make_csv_line(timestamp: int, temp: float, pwm: float = 200.0) -> str:
    setpoint = 200.0
    error = setpoint - temp
    return f"{timestamp},{setpoint},{temp},{pwm},{error},1.0,0.1,0.05"


class HardwareTuiLoopTests(unittest.TestCase):
    def test_ops_offset_compensation_removes_pure_rotation_arc(self):
        start_center = tuner._ops_to_chassis_center(0.0, 25.0, 0.0)
        yaw_deg = 29.05
        yaw_rad = math.radians(yaw_deg)
        rotated_ops_x = -25.0 * math.sin(yaw_rad)
        rotated_ops_y = 25.0 * math.cos(yaw_rad)
        rotated_center = tuner._ops_to_chassis_center(
            rotated_ops_x, rotated_ops_y, yaw_deg
        )

        self.assertAlmostEqual(start_center[0], 0.0, places=6)
        self.assertAlmostEqual(start_center[1], 0.0, places=6)
        self.assertAlmostEqual(rotated_center[0], 0.0, places=6)
        self.assertAlmostEqual(rotated_center[1], 0.0, places=6)

    def test_hardware_round_metrics_include_cross_track_yaw_and_saturation(self):
        metrics = {"current_error": 1.7, "overshoot": 0.8}
        samples = [
            {"pwm": 0.20, "cross_track": 0.0, "yaw_delta": 0.0},
            {"pwm": 0.10, "cross_track": 6.0, "yaw_delta": 0.4},
            {"pwm": 0.20, "cross_track": 4.0, "yaw_delta": -0.2},
        ]

        tuner._augment_hardware_round_metrics(
            metrics,
            samples,
            output_limit=0.20,
            stop_reason="TARGET",
        )

        self.assertAlmostEqual(metrics["speed_saturation_ratio"], 2 / 3)
        self.assertEqual(metrics["cross_track_final_mm"], 4.0)
        self.assertEqual(metrics["cross_track_peak_mm"], 6.0)
        self.assertEqual(metrics["yaw_delta_final_deg"], -0.2)
        self.assertFalse(metrics["return_to_start_checked"])

    def test_hardware_sample_safety_detects_wrong_direction_and_wrapped_yaw(self):
        self.assertIn(
            "WRONG DIR",
            tuner._hardware_sample_safety_reason({"input": -20.1}, None),
        )
        self.assertIsNone(
            tuner._hardware_sample_safety_reason(
                {"input": 1.0, "yaw": -179.0}, 179.0
            )
        )
        self.assertIn(
            "YAW LIMIT",
            tuner._hardware_sample_safety_reason(
                {"input": 1.0, "yaw": -160.0}, 179.0
            ),
        )

    def test_hardware_loop_applies_initial_pid_before_tuning(self):
        sent_commands: list[str] = []

        class FakeBridge:
            def __init__(self, _port, _baudrate, emit_console=True):
                self.emit_console = emit_console
                self.last_error = ""

            def connect(self):
                return True

            def disconnect(self):
                return None

            def read_line(self):
                return None

            def parse_data(self, line):
                return None

            def send_command(self, cmd):
                sent_commands.append(cmd)

        with patch.object(tuner, "SerialBridge", FakeBridge):
            with patch.dict(
                tuner.CONFIG,
                {
                    "BUFFER_SIZE": 3,
                    "MAX_TUNING_ROUNDS": 0,
                    "HARDWARE_TUNE_AXIS": "Y",
                    "HARDWARE_OUTPUT_LIMIT_MPS": 0.15,
                },
                clear=False,
            ):
                tuner._run_hardware_tuning_loop(
                    "COM9",
                    emit_console=False,
                    initial_pid={"p": 2.5, "i": 0.4, "d": 0.1},
                )

        self.assertEqual(
            sent_commands[:7],
            [
                "PID LIMIT X 0.150",
                "PID LIMIT Y 0.150",
                "PID LIMIT YAW 0.250",
                "TUNE AXIS Y",
                "STATUS",
                "TUNE LIMIT 0.150",
                "SET P:2.5 I:0.4 D:0.1",
            ],
        )

    def test_hardware_loop_emits_stream_and_decision_events(self):
        event_queue = Queue()
        event_sink = QueueEventSink(event_queue)
        controller = SimulationController()
        sent_commands: list[str] = []
        captured = {}

        class FakeBridge:
            def __init__(self, _port, _baudrate, emit_console=True):
                self.emit_console = emit_console
                self.last_error = ""
                self._lines = iter(
                    [
                        "# ROUND START 1 DIR 1",
                        _make_csv_line(0, 100.0),
                        _make_csv_line(1, 120.0),
                        _make_csv_line(2, 150.0),
                        "# ROUND STOP TARGET",
                    ]
                )

            def connect(self):
                return True

            def disconnect(self):
                return None

            def read_line(self):
                return next(self._lines, None)

            def parse_data(self, line):
                parts = line.split(",")
                return {
                    "timestamp": float(parts[0]),
                    "setpoint": float(parts[1]),
                    "input": float(parts[2]),
                    "pwm": float(parts[3]),
                    "error": float(parts[4]),
                    "p": float(parts[5]),
                    "i": float(parts[6]),
                    "d": float(parts[7]),
                }

            def send_command(self, cmd):
                sent_commands.append(cmd)

        class FakeTuner:
            def __init__(
                self,
                *_args,
                stream_callback=None,
                log_callback=None,
                emit_console=True,
                **_kwargs,
            ):
                self.stream_callback = stream_callback
                self.log_callback = log_callback
                self.emit_console = emit_console

            def analyze(
                self,
                _prompt_data,
                _history_text,
                tuning_mode="generic",
                prompt_context=None,
            ):
                captured["tuning_mode"] = tuning_mode
                captured["prompt_context"] = prompt_context
                if self.log_callback:
                    self.log_callback("llm", "  LLM 正在思考...")
                if self.stream_callback:
                    self.stream_callback('{"thought_process":"he', False)
                    self.stream_callback('{"thought_process":"hello"}', True)
                return {
                    "analysis_summary": "Stop after one hardware round.",
                    "tuning_action": "HOLD",
                    "p": 1.2,
                    "i": 0.1,
                    "d": 0.05,
                    "status": "DONE",
                }

        with patch.object(tuner, "SerialBridge", FakeBridge):
            with patch.object(tuner, "LLMTuner", FakeTuner):
                with patch.dict(
                    tuner.CONFIG,
                    {"BUFFER_SIZE": 3, "MAX_TUNING_ROUNDS": 2},
                    clear=False,
                ):
                    with patch.object(tuner, "time") as fake_time:
                        fake_time.time.side_effect = __import__("time").time
                        fake_time.monotonic.side_effect = __import__("time").monotonic
                        fake_time.sleep.return_value = None
                        result = tuner._run_hardware_tuning_loop(
                            "COM9",
                            event_sink=event_sink,
                            controller=controller,
                            emit_console=False,
                        )

        events = drain_event_queue(event_queue)
        event_types = {event["type"] for event in events}
        self.assertIn(EVENT_SAMPLE, event_types)
        self.assertIn(EVENT_ROUND_METRICS, event_types)
        self.assertIn(EVENT_DECISION, event_types)
        self.assertIn(EVENT_LOG, event_types)
        self.assertIn(EVENT_LIFECYCLE, event_types)
        self.assertGreaterEqual(result["rounds_completed"], 1)
        self.assertTrue(any(event.get("label") == "llm_stream" for event in events))
        self.assertTrue(any(cmd.startswith("SET P:") for cmd in sent_commands))
        self.assertEqual(captured["tuning_mode"], "hardware")
        self.assertEqual(captured["prompt_context"]["serial_port"], "COM9")

    def test_hardware_loop_runs_p_i_d_then_three_verification_rounds(self):
        stages: list[str] = []
        sent_pid_commands: list[str] = []

        class FakeBridge:
            def __init__(self, _port, _baudrate, emit_console=True):
                self.emit_console = emit_console
                self.last_error = ""
                self._lines: list[str] = []
                self._round = 0
                self._pid = {"p": 0.001, "i": 0.0, "d": 0.0}

            def connect(self):
                return True

            def disconnect(self):
                return None

            def clear_input_buffer(self):
                return None

            def read_line(self):
                return self._lines.pop(0) if self._lines else None

            def parse_data(self, line):
                parts = line.split(",")
                return {
                    "timestamp": float(parts[0]),
                    "setpoint": float(parts[1]),
                    "input": float(parts[2]),
                    "pwm": float(parts[3]),
                    "error": float(parts[4]),
                    "p": float(parts[5]),
                    "i": float(parts[6]),
                    "d": float(parts[7]),
                }

            def send_silent_command(self, _cmd):
                return None

            def send_command(self, cmd):
                if not cmd.startswith("SET P:"):
                    return
                sent_pid_commands.append(cmd)
                fields = cmd.replace("SET P:", "").replace(" I:", ",").replace(" D:", ",").split(",")
                self._pid = {"p": float(fields[0]), "i": float(fields[1]), "d": float(fields[2])}
                self._round += 1
                p, i, d = self._pid["p"], self._pid["i"], self._pid["d"]
                self._lines.extend(
                    [
                        f"# ROUND START {self._round} DIR 1",
                        f"0,200,180,0.05,20,{p},{i},{d}",
                        f"20,200,198,0.01,2,{p},{i},{d}",
                        f"40,200,200,0.0,0,{p},{i},{d}",
                        "# ROUND STOP TARGET",
                    ]
                )

        class FakeTuner:
            def __init__(self, *_args, **_kwargs):
                pass

            def analyze(self, _data, _history, tuning_mode="generic", prompt_context=None):
                stage = prompt_context["tuning_stage"]
                stages.append(stage)
                proposals = {
                    "P": {"p": 0.0045, "i": 0.00004, "d": 0.001},
                    "I": {"p": 0.005, "i": 0.00001, "d": 0.001},
                    "D": {"p": 0.005, "i": 0.00004, "d": 0.0002},
                }
                return {
                    "analysis_summary": f"{stage} complete",
                    "tuning_action": "HOLD",
                    **proposals[stage],
                    "status": "DONE",
                }

        with patch.object(tuner, "SerialBridge", FakeBridge):
            with patch.object(tuner, "LLMTuner", FakeTuner):
                with patch.dict(
                    tuner.CONFIG,
                    {
                        "BUFFER_SIZE": 3,
                        "MAX_TUNING_ROUNDS": 10,
                        "HARDWARE_STAGE_MAX_ROUNDS": 5,
                        "HARDWARE_VERIFY_ROUNDS": 3,
                    },
                    clear=False,
                ):
                    with patch.object(tuner.time, "sleep", return_value=None):
                        result = tuner._run_hardware_tuning_loop("COM9", emit_console=False)

        self.assertEqual(stages, ["P", "I", "D"])
        self.assertEqual(result["completed_reason"], "staged_validation_passed")
        self.assertEqual(result["final_pid"], {"p": 0.0015, "i": 0.00001, "d": 0.0002})
        self.assertEqual(len(sent_pid_commands), 6)

    def test_hardware_connection_failure_reports_error_result(self):
        event_queue = Queue()
        event_sink = QueueEventSink(event_queue)

        class FailingBridge:
            def __init__(self, _port, _baudrate, emit_console=True):
                self.emit_console = emit_console
                self.last_error = "port busy"

            def connect(self):
                return False

            def disconnect(self):
                return None

        with patch.object(tuner, "SerialBridge", FailingBridge):
            with patch.dict(tuner.CONFIG, {"BUFFER_SIZE": 3}, clear=False):
                result = tuner._run_hardware_tuning_loop(
                    "COM9",
                    event_sink=event_sink,
                    controller=None,
                    emit_console=False,
                )

        events = drain_event_queue(event_queue)
        self.assertEqual(result["completed_reason"], "error")
        self.assertTrue(any(event.get("phase") == "error" for event in events))

    def test_run_hardware_tuner_tui_failure_falls_back_to_plain_runner(self):
        with patch.object(tuner, "initialize_runtime_config"):
            with patch.object(tuner, "resolve_serial_port", return_value="COM9"):
                with patch.dict(
                    tuner.CONFIG,
                    {"LLM_DEBUG_OUTPUT": False, "HARDWARE_RESUME_LAST_PID": False},
                    clear=False,
                ):
                    with patch.object(
                        tuner,
                        "_run_hardware_tuning_with_tui",
                        side_effect=RuntimeError("tui boom"),
                    ):
                        with patch.object(
                            tuner,
                            "_run_hardware_tuning_plain",
                            return_value={"mode": "plain"},
                        ) as plain:
                            result = tuner.run_hardware_tuner(force_plain=False)

        self.assertEqual(result, {"mode": "plain"})
        plain.assert_called_once_with("COM9", initial_pid=None)


if __name__ == "__main__":
    unittest.main()
