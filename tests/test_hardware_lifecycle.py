import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))
import tuner
from hw.bridge import SerialBridge
from sim.runtime import SimulationController
from hw.session import configuration_reply
from core.pid_results import has_verified_result


class Device:
    is_open = True

    def __init__(self, lines=()):
        self.lines = list(lines)
        self.writes = []

    def write(self, payload):
        self.writes.append(payload)
        return len(payload)

    def readline(self):
        return (self.lines.pop(0) + "\n").encode() if self.lines else b""


class TransportTests(unittest.TestCase):
    def test_protocol_confirmation_rejects_unknown_version(self):
        accepts = configuration_reply("PROTO VERSION")
        self.assertTrue(accepts("# PROTO VERSION=4 MODES=WORK,TUNE"))
        self.assertFalse(accepts("# PROTO VERSION=4 MODES=WORK,TUNE,PLOT"))
        self.assertFalse(accepts("# PROTO VERSION=99 MODES=WORK,TUNE"))

    def test_verified_evidence_must_match_pid_metrics_and_stop(self):
        record = {"completed_reason": "staged_validation_passed", "tune_axis": "X",
                  "final_pid": {"p": 0.003, "i": 0, "d": 0}, "final_metrics": {"current_error": 1},
                  "stop_confirmation": "can_stop_sent"}
        record["verified_pid"] = dict(record["final_pid"])
        record["tested_result"] = {"verified": True, "axis": "X", "pid": dict(record["final_pid"]),
                                   "metrics": dict(record["final_metrics"])}
        self.assertTrue(has_verified_result(record))
        record["final_pid"]["p"] = 0.004
        self.assertFalse(has_verified_result(record))
        record["final_pid"]["p"] = 0.003
        record["stop_confirmation"] = "unconfirmed"
        self.assertFalse(has_verified_result(record))

    def test_configuration_ack_keeps_unrelated_diagnostics(self):
        bridge = SerialBridge("COM9", 115200, False)
        bridge.serial = Device(["# CAN READY=0", "# PID LOADED AXIS=X P=0.003 I=0 D=0"])
        self.assertTrue(bridge.send_command("PID SET X 0.003 0 0"))
        self.assertEqual(bridge.read_line(), "# CAN READY=0")

    def test_configuration_rejection_and_short_write_are_errors(self):
        bridge = SerialBridge("COM9", 115200, False)
        bridge.serial = Device(["# ERROR PID LIMIT"])
        with self.assertRaises(RuntimeError):
            bridge.send_command("PID SET X 1 0 0")
        bridge.serial.write = lambda _: 1
        with self.assertRaises(ConnectionError):
            bridge.send_command("SET P:1 I:0 D:0")

    def test_csv_requires_finite_values_and_real_pid_fields(self):
        bridge = SerialBridge("COM9", 115200, False)
        for line in ("0,1,0,0,1", "0,1,nan,0,1,0.003,0,0", "0,1,0,inf,1,0.003,0,0"):
            self.assertIsNone(bridge.parse_data(line))

    def test_pause_confirms_stop_and_keeps_reading_until_resume(self):
        controller = SimulationController()
        controller.pause()
        bridge = SerialBridge("COM9", 115200, False)
        bridge.serial = Device(["# MOTOR STOP STATE=WAIT FRESH=0", "# STOP MODE=TUNE", "# MOTOR STOP STATE=SENT FRESH=0 EVIDENCE=CAN_TX_ONLY"])
        def read():
            controller.resume()
            return "# CAN READY=1"
        bridge.read_line = read
        tuner._pause_hardware(bridge, controller)
        self.assertEqual(bridge.serial.writes, [b"MOTOR STOP STATUS\n", b"STOP\n", b"MOTOR STOP STATUS\n"])


class RoundTests(unittest.TestCase):
    def run_loop(self, *, missing_start=False, stop_during_analysis=False, failed_checkpoint=None):
        commands = []
        controller = SimulationController()
        clock = [0.0]
        def now():
            clock[0] += 0.05
            return clock[0]

        class Bridge:
            last_error = ""
            def __init__(self, *a, **kw):
                self.lines = []
                self.t = 0
                self.started = False
            def connect(self): return True
            def disconnect(self): pass
            def checkpoint(self, stage):
                if stage == failed_checkpoint:
                    raise RuntimeError(f"CAN checkpoint failed at {stage}")
            def send_command(self, cmd):
                commands.append(cmd)
                if cmd.startswith("SET P:"):
                    self.started = True
                    if not missing_start:
                        self.lines = ["# ROUND START 1 AXIS=X"] + [
                            f"{n},200,{n},0.01,{200-n},0.003,0,0" for n in range(3)
                        ] + ["# ROUND STOP TARGET AXIS=X"]
            def read_line(self):
                if self.lines: return self.lines.pop(0)
                if missing_start and self.started:
                    self.t += 1
                    return f"{self.t},200,1,0.01,199,0.003,0,0"
                return None
            def parse_data(self, line): return SerialBridge.parse_data(self, line)

        class Tuner:
            def __init__(self, *a, **kw): pass
            def analyze(self, *a, **kw):
                if stop_during_analysis:
                    controller.request_stop()
                return {"p": 0.004, "i": 0, "d": 0, "status": "TUNING"}

        with tempfile.TemporaryDirectory() as tmp, patch.object(tuner, "SerialBridge", Bridge), \
             patch.object(tuner, "LLMTuner", Tuner), \
             patch.object(tuner.time, "monotonic", side_effect=now), \
             patch.object(tuner.time, "sleep"), \
             patch.object(tuner, "resolve_project_path", side_effect=lambda _: Path(tmp)), \
             patch.dict(tuner.CONFIG, {"BUFFER_SIZE": 3, "MAX_TUNING_ROUNDS": 1,
                                       "HARDWARE_TUNE_AXIS": "X", "HARDWARE_RESUME_LAST_PID": False}):
            result = tuner._run_hardware_tuning_loop("COM9", controller=controller,
                      emit_console=False, initial_pid={"p": 0.003, "i": 0, "d": 0})
        return commands, result

    def test_missing_start_times_out_despite_continuous_csv(self):
        commands, result = self.run_loop(missing_start=True)
        self.assertEqual(result["completed_reason"], "start_timeout")
        self.assertEqual(sum(c.startswith("SET P:") for c in commands), 1)
        self.assertIsNone(result["tested_result"])

    def test_failed_can_checkpoint_never_starts_motion(self):
        for stage in ("after_mode_tune", "before_round"):
            with self.subTest(stage=stage):
                commands, result = self.run_loop(failed_checkpoint=stage)
                self.assertEqual(result["completed_reason"], "hardware_error")
                self.assertIn(stage, result["failure_detail"])
                self.assertFalse(any(c.startswith("SET P:") for c in commands))
                self.assertIn("STOP", commands)
                self.assertNotIn("MODE WORK", commands)

    def test_last_round_does_not_start_proposal_or_mislabel_metrics(self):
        commands, result = self.run_loop()
        self.assertEqual(sum(c.startswith("SET P:") for c in commands), 1)
        self.assertEqual(result["completed_reason"], "max_rounds_reached")
        self.assertEqual(result["final_pid"]["p"], 0.003)
        self.assertEqual(result["tested_result"]["pid"], result["final_pid"])
        self.assertEqual(result["tested_result"]["metrics"], result["final_metrics"])
        self.assertNotEqual(result["suggested_pid"], result["final_pid"])

    def test_stop_during_llm_request_prevents_next_motion(self):
        commands, result = self.run_loop(stop_during_analysis=True)
        self.assertEqual(sum(c.startswith("SET P:") for c in commands), 1)
        self.assertEqual(result["completed_reason"], "stopped_by_user")


if __name__ == "__main__":
    unittest.main()
