"""逐轮原始采样留存测试。"""

import tempfile
import unittest
from pathlib import Path

from core.round_csv import HardwareRoundCsvRecorder, ROUND_CSV_HEADER


class HardwareRoundCsvRecorderTests(unittest.TestCase):
    def test_keeps_complete_and_interrupted_rounds_separately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = HardwareRoundCsvRecorder(Path(directory), "X")
            first = recorder.start()
            recorder.append("60,200,0,0.012,200,0.00495,0,0,1,2,3,0,0,0,0,1,2\r\n")
            recorder.close()

            second = recorder.start()
            recorder.append("60,-200,0,-0.012,-200,0.00495,0,0,1,2,3,0,-1,0,0.15,1,2")
            recorder.close()

            self.assertEqual(first.name, "round_001.csv")
            self.assertEqual(second.name, "round_002.csv")
            self.assertEqual(
                first.read_text(encoding="utf-8").splitlines(),
                [ROUND_CSV_HEADER, "60,200,0,0.012,200,0.00495,0,0,1,2,3,0,0,0,0,1,2"],
            )
            self.assertEqual(
                second.read_text(encoding="utf-8").splitlines(),
                [ROUND_CSV_HEADER, "60,-200,0,-0.012,-200,0.00495,0,0,1,2,3,0,-1,0,0.15,1,2"],
            )


if __name__ == "__main__":
    unittest.main()
