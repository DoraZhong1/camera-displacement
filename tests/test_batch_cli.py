import os
import tempfile
import unittest
from pathlib import Path

from main import collect_batch_inputs


class CollectBatchInputsTest(unittest.TestCase):
    def test_collects_videos_from_directory_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mount = root / "displacement-mount3"
            control = root / "displacement-control2"
            mount.mkdir()
            control.mkdir()

            (mount / "cam1.mp4").write_bytes(b"video")
            (mount / "cam2.mov").write_bytes(b"video")
            (mount / "cam3.avi").write_bytes(b"video")

            (control / "cam1.mp4").write_bytes(b"video")
            (control / "cam2.mov").write_bytes(b"video")
            (control / "cam3.avi").write_bytes(b"video")

            entries = collect_batch_inputs([str(mount), str(control)])

            self.assertEqual(2, len(entries))
            self.assertEqual("displacement-mount3", entries[0][0])
            self.assertEqual("displacement-control2", entries[1][0])
            self.assertEqual(3, len(entries[0][1]))
            self.assertEqual(3, len(entries[1][1]))
            self.assertTrue(all(os.path.isfile(p) for _, files in entries for p in files))


if __name__ == "__main__":
    unittest.main()
