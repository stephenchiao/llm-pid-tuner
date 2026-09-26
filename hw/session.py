"""Acknowledgements and round admission shared by hardware tuning paths."""
import math
import re


def configuration_reply(command):
    parts = command.split()
    if command == "PROTO VERSION":
        def compatible(line):
            match = re.match(r"# PROTO VERSION=(\d+)\b", line)
            return bool(match and int(match.group(1)) == 4 and "MODES=WORK,TUNE,PLOT" in line)
        return compatible
    if command.startswith("MODE "):
        return lambda line: line.split()[:3] == ["#", "MODE", parts[1]]
    if command.startswith("TUNE AXIS "):
        return lambda line: line.split()[:4] == ["#", "TUNE", "AXIS", parts[2]]
    if command.startswith(("PID SET ", "PID LIMIT ", "TUNE LIMIT ")):
        prefix = "# PID LOADED " if parts[:2] == ["PID", "SET"] else "# " + " ".join(parts[:2]) + " "
        def matches(line):
            if not line.startswith(prefix):
                return False
            kv = dict(re.findall(r"(\w+)=([^\s]+)", line))
            try:
                if parts[0] == "PID" and kv.get("AXIS") != parts[2]:
                    return False
                if parts[1] == "SET":
                    return all(math.isclose(float(kv[k]), float(v), rel_tol=1e-6, abs_tol=1e-8)
                               for k, v in zip(("P", "I", "D"), parts[3:]))
                return math.isclose(float(kv["OUTPUT"]), float(parts[-1]), abs_tol=0.00051)
            except (KeyError, ValueError):
                return False
        return matches
    return None


def can_start_round(rounds, maximum, controller=None):
    return rounds < int(maximum) and not (
        controller is not None and (controller.should_stop or controller.is_paused)
    )
