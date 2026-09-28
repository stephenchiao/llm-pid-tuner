#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hw/bridge.py - serial bridge helpers for real hardware and demo mode.
"""

from __future__ import annotations

import re
import math
from collections import deque
import time

import serial
import serial.tools.list_ports


DEMO_SERIAL_PORT = "COM_FAKE"
DEMO_SERIAL_PORT_ALIASES = {
    DEMO_SERIAL_PORT,
    "FAKE",
    "DEMO",
    "DEMO_HW",
    "VIRTUAL",
}
HOST_LINK_TIMEOUT_SEC = 1.5


def _is_demo_port(port: str | None) -> bool:
    return str(port or "").strip().upper() in DEMO_SERIAL_PORT_ALIASES


class _DemoSerialDevice:
    """In-process fake serial device so the hardware TUI can be previewed."""

    _set_pid_re = re.compile(
        r"SET\s+P:(?P<p>-?\d+(?:\.\d+)?)\s+I:(?P<i>-?\d+(?:\.\d+)?)\s+D:(?P<d>-?\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        from sim.model import HeatingSimulator

        self.is_open = True
        self._sim = HeatingSimulator(random_seed=7)
        self._last_command = ""

    def close(self) -> None:
        self.is_open = False

    def readline(self) -> bytes:
        if not self.is_open:
            return b""

        self._sim.compute_pid()
        self._sim.update()
        data = self._sim.get_data()
        # Slow the preview down a bit so the TUI remains readable.
        time.sleep(0.05)
        return (
            f"{data['timestamp']:.0f},{data['setpoint']:.3f},{data['input']:.3f},"
            f"{data['pwm']:.3f},{data['error']:.3f},{data['p']:.4f},"
            f"{data['i']:.4f},{data['d']:.4f}\n"
        ).encode("utf-8")

    def write(self, payload: bytes) -> None:
        if not self.is_open:
            return

        command = payload.decode("utf-8", errors="ignore").strip()
        self._last_command = command
        if not command:
            return
        if command.upper() == "STATUS":
            return len(payload)

        match = self._set_pid_re.fullmatch(command)
        if not match:
            return len(payload)

        self._sim.set_pid(
            float(match.group("p")),
            float(match.group("i")),
            float(match.group("d")),
        )
        return len(payload)


class SerialBridge:
    requires_hardware_preflight = True

    def __init__(self, port: str, baudrate: int, emit_console: bool = True):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.emit_console = emit_console
        self.last_error = ""
        self.pending_lines = deque()
        self.is_demo = _is_demo_port(port)
        self.on_line = None
        self.on_io = None
        self._rx_fragment = b""

    def connect(self) -> bool:
        try:
            if _is_demo_port(self.port):
                self.serial = _DemoSerialDevice()
                self.last_error = ""
                if self.emit_console:
                    print(f"[INFO] Connected to virtual hardware feed: {DEMO_SERIAL_PORT}")
                return True

            # Configure modem lines before opening; do not toggle them as a reset.
            self.serial = serial.Serial(port=None, baudrate=self.baudrate,
                                        timeout=0.2, write_timeout=1.0,
                                        xonxoff=False, rtscts=False, dsrdtr=False)
            self.serial.dtr = False
            self.serial.rts = False
            self.serial.port = self.port
            self.trace("OPEN", f"port={self.port} baud={self.baudrate} DTR=0 RTS=0")
            self.serial.open()
            if not self._claim_com_host():
                raise ConnectionError(self.last_error)
            self.last_error = ""
            if self.emit_console:
                print(f"[INFO] Connected to {self.port}")
            return True
        except Exception as e:
            self.last_error = str(e)
            self.trace("ERROR", self.last_error)
            if self.serial and self.serial.is_open:
                try:
                    from hw.diagnostics import capture_failure
                    capture_failure(self)
                except Exception as diagnostic_error:
                    self.trace("ERROR", f"failure capture: {diagnostic_error}")
            self.disconnect()
            if self.emit_console:
                print(f"[ERROR] Connection failed: {e}")
            return False

    def _claim_com_host(self) -> bool:
        """真机连接后先取得 COM 所有权；BUSY/超时都禁止继续调参。"""
        self.trace("STAGE", "initial_stop")
        self.send_silent_command("STOP")
        # STOP 会清除固件待执行队列，确认后才能提交新的握手命令。
        deadline = time.monotonic() + HOST_LINK_TIMEOUT_SEC
        while time.monotonic() < deadline:
            line = self.read_line()
            if line and (line.startswith("# STOP MODE=") or
                         line.startswith("# ROUND STOP HOST")):
                break
        else:
            self.last_error = "timeout waiting for STOP acknowledgement"
            return False
        self.trace("STAGE", "claim_com")
        self.send_silent_command("HOST LINK COM")
        deadline = time.monotonic() + HOST_LINK_TIMEOUT_SEC
        while time.monotonic() < deadline:
            line = self.read_line()
            # 按完整 token 判断，兼容旧版及 HEARTBEAT 等扩展字段。
            if line and line.split()[:5] == ["#", "HOST", "LINK", "COM", "OK"]:
                # Claim ownership first so CAN diagnostics are available on failure.
                # Motion remains blocked until the stop frames are sent and CAN is ready.
                self.wait_stopped()
                self.checkpoint("after_host_link")
                return True
        self.last_error = "timeout waiting for # HOST LINK COM OK"
        return False

    def trace(self, direction, text):
        if self.on_io is not None:
            self.on_io(direction, text)

    def checkpoint(self, stage):
        if not self.is_demo:
            from hw.diagnostics import wait_healthy
            wait_healthy(self, stage)

    def wait_stopped(self, timeout=2.0):
        from hw.diagnostics import fields
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self.request("MOTOR STOP STATUS", lambda text: text.startswith("# MOTOR STOP STATE="),
                                timeout=min(1.0, max(0.05, deadline - time.monotonic())))
            data = fields(line)
            if data.get("STATE") == "SENT" and data.get("EVIDENCE") == "CAN_TX_ONLY":
                return
            time.sleep(0.1)
        raise TimeoutError("STOP acknowledged but CAN stop-frame delivery is unconfirmed")

    def disconnect(self) -> None:
        if self.serial:
            self.serial.close()
            self.trace("CLOSE", self.port)
            self.serial = None

    def _read_device_line(self):
        if not self.serial or not self.serial.is_open:
            raise ConnectionError("serial port is closed")
        try:
            payload = self.serial.readline()
            if not payload:
                return None
            self._rx_fragment += payload
            if len(self._rx_fragment) > 4096:
                raise ValueError("serial line exceeds 4096 bytes")
            if not self._rx_fragment.endswith(b"\n"):
                return None
            line = self._rx_fragment.decode("utf-8", errors="strict").strip()
            self._rx_fragment = b""
            self.trace("RX", line)
            if self.on_line is not None:
                self.on_line(line)
            return line
        except Exception as exc:
            self.last_error = str(exc)
            raise ConnectionError(self.last_error) from exc

    def read_line(self):
        if self.pending_lines:
            return self.pending_lines.popleft()
        return self._read_device_line()

    def _write(self, cmd: str, *, quiet: bool = False) -> bool:
        if not self.serial or not self.serial.is_open:
            raise ConnectionError("serial port is closed")
        payload = f"{cmd}\n".encode("utf-8")
        self.trace("TX", cmd)
        try:
            count = self.serial.write(payload)
            if count != len(payload):
                raise OSError(f"short serial write: {count}/{len(payload)}")
        except Exception as exc:
            self.last_error = str(exc)
            raise ConnectionError(self.last_error) from exc
        if self.emit_console and not quiet:
            print(f"[CMD] Sent: {cmd}")
        return True

    def request(self, command, predicate, timeout=1.5, *, tolerate_errors=False):
        """One serial reader; unrelated replies remain available to the loop."""
        self._write(command)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self._read_device_line()
            if not line:
                continue
            if not tolerate_errors and line.startswith(("# ERROR", "# MOTION STOP SAFETY", "# CAN SAFETY", "# CAN FEEDBACK LOST")):
                raise RuntimeError(line)
            if predicate(line):
                return line
            self.pending_lines.append(line)
        raise TimeoutError(f"timeout waiting for {command}")

    def send_command(self, cmd: str) -> bool:
        from hw.session import configuration_reply
        predicate = configuration_reply(cmd)
        if predicate is not None and not self.is_demo:
            self.request(cmd, predicate)
            return True
        return self._write(cmd)

    def send_silent_command(self, cmd: str) -> bool:
        return self._write(cmd, quiet=True)

    def clear_input_buffer(self) -> None:
        """Discard replies left from the completed hardware round."""
        if self.serial and self.serial.is_open:
            try:
                reset_input = getattr(self.serial, "reset_input_buffer", None)
                if callable(reset_input):
                    reset_input()
            except Exception as e:
                self.last_error = str(e)

    def parse_data(self, line: str):
        if not line or line.startswith("#"):
            return None
        parts = line.split(",")
        if len(parts) >= 8:
            try:
                if not all(math.isfinite(float(value)) for value in parts):
                    return None
                return {
                    "timestamp": float(parts[0]),
                    "setpoint": float(parts[1]),
                    "input": float(parts[2]),
                    "pwm": float(parts[3]),
                    "error": float(parts[4]),
                    "p": float(parts[5]) if len(parts) > 5 else 1.0,
                    "i": float(parts[6]) if len(parts) > 6 else 0.1,
                    "d": float(parts[7]) if len(parts) > 7 else 0.05,
                    "x": float(parts[8]) if len(parts) > 8 else None,
                    "y": float(parts[9]) if len(parts) > 9 else None,
                    "yaw": float(parts[10]) if len(parts) > 10 else None,
                    "cross_track": float(parts[11]) if len(parts) > 11 else None,
                    "yaw_delta": float(parts[12]) if len(parts) > 12 else None,
                    "hold_cross_output": float(parts[13]) if len(parts) > 13 else None,
                    "hold_yaw_output": float(parts[14]) if len(parts) > 14 else None,
                    "center_x": float(parts[15]) if len(parts) > 15 else None,
                    "center_y": float(parts[16]) if len(parts) > 16 else None,
                }
            except Exception:
                pass
        return None


def safe_pause(message: str = "按回车键退出...") -> None:
    try:
        input(message)
    except EOFError:
        pass


def select_serial_port() -> str:
    """Interactively choose a serial port, or start the virtual demo feed."""
    print("\n[INFO] 正在扫描可用串口...")
    ports = list(serial.tools.list_ports.comports())

    if not ports:
        print("[WARN] 未发现任何串口设备。")
        choice = input(
            f"输入串口号（例如 COM3），或输入 'd' 进入虚拟硬件演示模式 [{DEMO_SERIAL_PORT}]: "
        ).strip()
        if choice.lower() == "d":
            return DEMO_SERIAL_PORT
        return choice

    print(f"发现 {len(ports)} 个设备:")
    for i, p in enumerate(ports):
        print(f"  [{i + 1}] {p.device} - {p.description}")
    print(f"  [D] 虚拟硬件演示模式 - {DEMO_SERIAL_PORT}")

    while True:
        choice = (
            input(
                f"\n请选择序号 (1-{len(ports)})、输入 'd' 演示，或输入 'm' 手动指定: "
            )
            .strip()
            .lower()
        )
        if choice == "d":
            return DEMO_SERIAL_PORT
        if choice == "m":
            return input("请输入串口号: ").strip()

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(ports):
                return ports[idx].device

        print("[ERROR] 输入无效，请重试。")
