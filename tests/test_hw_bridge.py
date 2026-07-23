import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).parent.parent))

from hw.bridge import DEMO_SERIAL_PORT, SerialBridge, select_serial_port


class DemoSerialBridgeTests(unittest.TestCase):
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
