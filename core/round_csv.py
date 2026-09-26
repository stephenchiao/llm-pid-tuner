"""保存硬件调参的逐轮原始串口采样。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TextIO


ROUND_CSV_HEADER = (
    "elapsed_ms,target,input,output,error,p,i,d,ops_x,ops_y,ops_yaw,"
    "cross_track,yaw_delta,hold_cross_output,hold_yaw_output,center_x,center_y"
)


class HardwareRoundCsvRecorder:
    def __init__(self, root: Path, axis: str) -> None:
        session_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{axis}"
        self.session_dir = root / session_name
        self.round_number = 0
        self.path: Path | None = None
        self._file: TextIO | None = None

    def start(self) -> Path:
        self.close()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.round_number += 1
        self.path = self.session_dir / f"round_{self.round_number:03d}.csv"
        self._file = self.path.open("x", encoding="utf-8", newline="")
        self._file.write(ROUND_CSV_HEADER + "\n")
        self._file.flush()
        return self.path

    def append(self, raw_line: str) -> None:
        if self._file is not None:
            self._file.write(raw_line.rstrip("\r\n") + "\n")
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
