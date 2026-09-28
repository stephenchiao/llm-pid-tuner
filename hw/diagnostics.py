"""Read-only CAN checkpoints and a durable, timestamped serial transcript."""
from datetime import datetime
import json
from pathlib import Path
import re
import time


def fields(line):
    return dict(re.findall(r"(\w+)=([^\s]+)", line))


class SerialTranscript:
    def __init__(self, root):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jsonl")
        self.file = self.path.open("x", encoding="utf-8")

    def write(self, direction, text):
        self.file.write(json.dumps({"time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                                   "monotonic": time.monotonic(), "direction": direction,
                                   "text": text}, ensure_ascii=False) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


def snapshot(bridge, *, tolerate_errors=False):
    can, stop = {}, {}

    def can_line(line):
        if line.startswith("# CAN "):
            can.update(fields(line))
        return line.startswith("# CAN ESR=")

    def stop_line(line):
        if line.startswith("# MOTOR STOP STATE="):
            stop.update(fields(line))
            return True
        return False

    errors = []
    for command, predicate in (("CAN STATUS", can_line), ("MOTOR STOP STATUS", stop_line)):
        try:
            bridge.request(command, predicate, timeout=1.0, tolerate_errors=tolerate_errors)
        except (ConnectionError, RuntimeError, TimeoutError, ValueError) as exc:
            if not tolerate_errors:
                raise
            errors.append(str(exc))
    return {"can": can, "stop": stop, "errors": errors}


def healthy_progress(before, after):
    """Require stable CAN readiness and proof that the STOP frames were sent."""
    try:
        a, b = before["can"], after["can"]
        mask = int(b["MASK"], 0)
        if not 0 < mask <= 15 or mask != int(a["MASK"], 0):
            return False
        for sample, stop in ((a, before["stop"]), (b, after["stop"])):
            if (int(sample["STATE"]) != 2 or int(sample["READY"]) != 1 or
                    int(sample["TX_FAULT"]) != 0 or int(sample["ESR"], 0) & 4):
                return False
            if (stop.get("STATE") != "SENT" or stop.get("EVIDENCE") != "CAN_TX_ONLY" or
                    int(stop["MASK"], 0) != mask):
                return False
        if any(int(a[k]) != int(b[k]) for k in ("TX_TIMEOUT", "TX_ERR")):
            return False
        return True
    except (KeyError, ValueError, TypeError):
        return False


def wait_healthy(bridge, stage, timeout=3.0):
    bridge.trace("STAGE", stage)
    deadline = time.monotonic() + timeout
    previous = snapshot(bridge)
    while time.monotonic() < deadline:
        time.sleep(0.1)
        current = snapshot(bridge)
        if healthy_progress(previous, current):
            bridge.trace("CHECKPOINT_OK", stage)
            return
        previous = current
    raise RuntimeError(f"CAN checkpoint failed at {stage}: {previous}")


def capture_failure(bridge):
    """Two observations, no reset, mode switch, enable or movement command."""
    for index in range(2):
        bridge.trace("STAGE", f"failure_snapshot_{index + 1}")
        data = snapshot(bridge, tolerate_errors=True)
        bridge.trace("DIAGNOSTIC", json.dumps(data, ensure_ascii=False))
        if index == 0:
            time.sleep(0.2)
