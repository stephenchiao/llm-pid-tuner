import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).parent.parent))

from hw.bridge import DEMO_SERIAL_PORT, SerialBridge, select_serial_port


class FakeSerial:
    def __init__(self, responses):
        self.responses = list(responses)
        self.writes = []
        self.is_open = True

    def write(self, payload):
        self.writes.append(bytes(payload))
        if payload == b"MOTOR STOP STATUS\n":
            self.responses.insert(0, "# MOTOR STOP STATE=SENT MASK=0x0F FRESH=0 EVIDENCE=CAN_TX_ONLY")
        elif payload == b"CAN STATUS\n":
            self.responses.extend([
                f"# CAN STATE=2 TX_OK={len(self.writes)} TX_ERR=0 TX_TIMEOUT=0",
                "# CAN READY=1 MASK=0x0F TX_FAULT=0", "# CAN ESR=0x0"])
        elif payload == b"MOTOR FEEDBACK\n":
            self.responses.extend(f"# MOTOR FEEDBACK ID={i} VALID=1 RPM=0 AGE_MS=10 SEQ={len(self.writes)}" for i in range(1, 5))
        return len(payload)

    def open(self):
        assert self.dtr is False and self.rts is False
        self.is_open = True

    def readline(self):
        if self.responses:
            return (self.responses.pop(0) + "\n").encode("utf-8")
        return b""

    def close(self):
        self.is_open = False


class DemoSerialBridgeTests(unittest.TestCase):
    def test_claim_accepts_current_firmware_extended_reply(self):
        device = FakeSerial(["# STOP MODE=WORK", "# HOST LINK COM OK HEARTBEAT=OFF"])
        with patch("hw.bridge.serial.Serial", return_value=device):
            bridge = SerialBridge("COM9", 115200, emit_console=False)
            self.assertTrue(bridge.connect())

    def test_claim_rejects_wrong_host_and_non_ok_token(self):
        for response in ("# HOST LINK RPI OK", "# HOST LINK COM OKAY", "# HOST LINK COM BUSY"):
            with self.subTest(response=response):
                device = FakeSerial(["# STOP MODE=WORK", response])
                with patch("hw.bridge.serial.Serial", return_value=device), patch(
                    "hw.bridge.HOST_LINK_TIMEOUT_SEC", 0.01
                ):
                    bridge = SerialBridge("COM9", 115200, emit_console=False)
                    self.assertFalse(bridge.connect())

    def test_claim_waits_for_stop_before_sending_link(self):
        class OrderedDevice(FakeSerial):
            def readline(self):
                if self.responses and self.responses[0].startswith("# STOP"):
                    self.assert_stop_only()
                return super().readline()

            def assert_stop_only(self):
                assert self.writes == [b"STOP\n"]

        device = OrderedDevice(["# STOP MODE=WORK", "# HOST LINK COM OK HEARTBEAT=OFF"])
        with patch("hw.bridge.serial.Serial", return_value=device):
            self.assertTrue(SerialBridge("COM9", 115200, emit_console=False).connect())

    def test_missing_stop_ack_does_not_send_host_link(self):
        device = FakeSerial([])
        with patch("hw.bridge.serial.Serial", return_value=device), patch(
            "hw.bridge.HOST_LINK_TIMEOUT_SEC", 0.01
        ):
            self.assertFalse(SerialBridge("COM9", 115200, emit_console=False).connect())
        self.assertNotIn(b"HOST LINK COM\n", device.writes)
        self.assertEqual(device.writes[0], b"STOP\n")

    def test_parse_extended_hardware_csv_pose_fields(self):
        bridge = SerialBridge("COM9", 115200, emit_console=False)
        data = bridge.parse_data(
            "20,200,4,0.004,196,0.004,0,0,12.5,34.5,-179.0,"
            "8.7,0.41,-0.0287,-0.0082,12.06,59.50"
        )

        self.assertEqual(data["x"], 12.5)
        self.assertEqual(data["y"], 34.5)
        self.assertEqual(data["yaw"], -179.0)
        self.assertEqual(data["cross_track"], 8.7)
        self.assertEqual(data["yaw_delta"], 0.41)
        self.assertEqual(data["hold_cross_output"], -0.0287)
        self.assertEqual(data["hold_yaw_output"], -0.0082)
        self.assertEqual(data["center_x"], 12.06)
        self.assertEqual(data["center_y"], 59.50)

    def test_demo_port_streams_parseable_hardware_data(self):
        bridge = SerialBridge(DEMO_SERIAL_PORT, 115200, emit_console=False)

        self.assertTrue(bridge.connect())
        first_line = bridge.read_line()
        first_data = bridge.parse_data(first_line)

        bridge.send_command("SET P:2.5 I:0.4 D:0.1")
        second_line = bridge.read_line()
        second_data = bridge.parse_data(second_line)
        bridge.disconnect()

        self.assertIsNotNone(first_data)
        self.assertIsNotNone(second_data)
        self.assertAlmostEqual(second_data["p"], 2.5, places=3)
        self.assertAlmostEqual(second_data["i"], 0.4, places=3)
        self.assertAlmostEqual(second_data["d"], 0.1, places=3)
        self.assertGreaterEqual(second_data["timestamp"], first_data["timestamp"])

    def test_real_port_claims_com_before_connect_succeeds(self):
        device = FakeSerial(["# STOP MODE=WORK", "# HOST LINK COM OK"])
        with patch("hw.bridge.serial.Serial", return_value=device):
            bridge = SerialBridge("COM9", 115200, emit_console=False)
            self.assertTrue(bridge.connect())
        self.assertEqual(device.writes[:3], [b"STOP\n", b"HOST LINK COM\n", b"MOTOR STOP STATUS\n"])

    def test_unconfirmed_stop_is_diagnosed_after_claim_without_starting_motion(self):
        device = FakeSerial(["# STOP MODE=WORK HOST=WAITING", "# HOST LINK COM OK"])
        with patch("hw.bridge.serial.Serial", return_value=device), patch.object(
            SerialBridge, "wait_stopped", side_effect=TimeoutError("stop not sent")
        ), patch("hw.diagnostics.time.sleep"):
            bridge = SerialBridge("COM9", 115200, emit_console=False)
            self.assertFalse(bridge.connect())
        self.assertEqual(device.writes[:2], [b"STOP\n", b"HOST LINK COM\n"])
        self.assertIn(b"CAN STATUS\n", device.writes)
        self.assertNotIn(b"MODE TUNE\n", device.writes)


class SelectSerialPortTests(unittest.TestCase):
    def test_returns_demo_port_when_no_devices_and_user_requests_demo(self):
        with patch("hw.bridge.serial.tools.list_ports.comports", return_value=[]):
            with patch("builtins.input", return_value="d"):
                port = select_serial_port()

        self.assertEqual(port, DEMO_SERIAL_PORT)

    def test_returns_demo_port_when_devices_exist_and_user_requests_demo(self):
        fake_port = types.SimpleNamespace(device="COM7", description="USB Serial")
        with patch("hw.bridge.serial.tools.list_ports.comports", return_value=[fake_port]):
            with patch("builtins.input", return_value="d"):
                port = select_serial_port()

        self.assertEqual(port, DEMO_SERIAL_PORT)


if __name__ == "__main__":
    unittest.main()
