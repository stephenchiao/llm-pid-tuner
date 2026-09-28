import json
import tempfile
import unittest
from unittest.mock import Mock, patch

import tuner
from hw.bridge import SerialBridge
from hw.diagnostics import SerialTranscript, capture_failure, healthy_progress, wait_healthy


def sample(seq):
    return {"can": dict(STATE="2", READY="1", MASK="0x0F", ESR="0x0",
                        TX_FAULT="0", TX_OK=str(seq), TX_ERR="5", TX_TIMEOUT="2"),
            "stop": dict(STATE="SENT", MASK="0x0F", FRESH="0", EVIDENCE="CAN_TX_ONLY")}


class CanDiagnosticsTests(unittest.TestCase):
    def test_stable_ready_accepts_quiet_bus_and_historical_errors(self):
        self.assertTrue(healthy_progress(sample(10), sample(10)))
        self.assertTrue(healthy_progress(sample(10), sample(11)))

    def test_rejects_unready_bus_faults_or_unsent_stop(self):
        for mutate in (
            lambda s: s["can"].update(READY="0"),
            lambda s: s["can"].update(TX_FAULT="1"),
            lambda s: s["can"].update(ESR="0x4"),
            lambda s: s["can"].update(TX_ERR="6"),
            lambda s: s["can"].update(TX_TIMEOUT="3"),
            lambda s: s["stop"].update(STATE="WAIT"),
            lambda s: s["stop"].update(EVIDENCE="UNKNOWN"),
            lambda s: s["stop"].update(MASK="0x07"),
        ):
            after = sample(11)
            mutate(after)
            self.assertFalse(healthy_progress(sample(10), after))
        warning = sample(11)
        warning["can"]["ESR"] = "0x3"
        self.assertTrue(healthy_progress(sample(10), warning))

    def test_checkpoint_timeout_cannot_succeed_when_can_unready(self):
        unready = sample(10)
        unready["can"]["READY"] = "0"
        with patch("hw.diagnostics.snapshot", return_value=unready), \
             patch("hw.diagnostics.time.sleep"), \
             patch("hw.diagnostics.time.monotonic", side_effect=[0, 0, 4]):
            with self.assertRaisesRegex(RuntimeError, "after_mode_tune"):
                wait_healthy(Mock(), "after_mode_tune")

    def test_capture_collects_two_read_only_snapshots(self):
        bridge = Mock()
        def reply(command, predicate, **kwargs):
            if command == "CAN STATUS":
                for line in ("# CAN STATE=2 TX_ERR=5 TX_TIMEOUT=2", "# CAN READY=0 MASK=0x0F TX_FAULT=1", "# CAN ESR=0x840002"):
                    predicate(line)
            else:
                predicate("# MOTOR STOP STATE=WAIT MASK=0x0F FRESH=0 EVIDENCE=CAN_TX_ONLY")
        bridge.request.side_effect = reply
        with patch("hw.diagnostics.time.sleep"):
            capture_failure(bridge)
        self.assertEqual([c.args[0] for c in bridge.request.call_args_list],
                         ["CAN STATUS", "MOTOR STOP STATUS"] * 2)
        records = [json.loads(c.args[1]) for c in bridge.trace.call_args_list if c.args[0] == "DIAGNOSTIC"]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[1]["stop"]["STATE"], "WAIT")

    def test_status_error_does_not_suppress_stop(self):
        bridge = Mock(requires_hardware_preflight=True, is_demo=False)
        bridge.request.side_effect = [RuntimeError("# CAN SAFETY"), "# STOP MODE=TUNE"]
        self.assertEqual(tuner._confirm_hardware_stop(bridge), "can_stop_sent")
        self.assertEqual([c.args[0] for c in bridge.request.call_args_list], ["MOTOR STOP STATUS", "STOP"])
        bridge.wait_stopped.assert_called_once()

    def test_sent_stop_is_not_reissued(self):
        bridge = Mock(requires_hardware_preflight=True, is_demo=False)
        bridge.request.return_value = "# MOTOR STOP STATE=SENT MASK=0x0F FRESH=0 EVIDENCE=CAN_TX_ONLY"
        self.assertEqual(tuner._confirm_hardware_stop(bridge), "can_stop_sent")
        self.assertEqual(bridge.request.call_count, 1)

    def test_timeout_fragment_is_preserved_and_transcript_flushed(self):
        bridge = SerialBridge("COM9", 115200, False)
        bridge.serial = Mock(is_open=True)
        bridge.serial.readline.side_effect = [b"# CAN RE", b"ADY=0\r\n"]
        with tempfile.TemporaryDirectory() as root:
            log = SerialTranscript(root)
            bridge.on_io = log.write
            self.assertIsNone(bridge.read_line())
            self.assertEqual(bridge.read_line(), "# CAN READY=0")
            row = json.loads(log.path.read_text(encoding="utf-8"))
            self.assertEqual(row["text"], "# CAN READY=0")
            log.close()
