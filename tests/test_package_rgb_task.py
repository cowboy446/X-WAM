import tarfile
import tempfile
import unittest
from pathlib import Path

from evaluation.package_rgb_task import collect_rgb_files, package_rgb_task


class PackageRGBTaskTest(unittest.TestCase):
    def test_packages_all_rgb_files_with_task_root_and_original_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = root / "PnPCounterToCab"
            chunk = task / "0_7_4d/chunks/step_000000"
            expected = {
                chunk / "predicted/rgb/camera_0/frame_0000.png",
                chunk / "predicted/rgb/camera_1/frame_0001.jpg",
                chunk / "groundtruth/rgb/camera_0/frame_0000.png",
            }
            for path in expected:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"rgb")
            excluded = chunk / "predicted/depth/camera_0/frame_0000.png"
            excluded.parent.mkdir(parents=True)
            excluded.write_bytes(b"depth")

            output = root / "bundle.tar.gz"
            package_rgb_task(task, output)

            with tarfile.open(output, "r:gz") as archive:
                names = set(archive.getnames())
            self.assertEqual(
                names,
                {str(path.relative_to(task.parent)) for path in expected},
            )

    def test_supports_ground_truth_directory_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = Path(tmp) / "PnPCounterToCab"
            rgb = task / "1_0_4d/chunks/step_000001/ground_truth/rgb/frame.png"
            rgb.parent.mkdir(parents=True)
            rgb.write_bytes(b"rgb")

            self.assertEqual(collect_rgb_files(task), [rgb.resolve()])

    def test_fails_when_no_rgb_files_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = Path(tmp) / "PnPCounterToCab"
            (task / "0_0_4d").mkdir(parents=True)

            with self.assertRaises(FileNotFoundError):
                collect_rgb_files(task)


if __name__ == "__main__":
    unittest.main()
