import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path

from evaluation.package_4d_task import package_4d_task


class Package4DTaskTest(unittest.TestCase):
    def test_packages_referenced_ply_and_json_with_original_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = root / "PnPCounterToCab"
            rollout = task / "0_0_4d"
            timeline = rollout / "timeline"
            cloud = rollout / "chunks/step_000000/predicted/pointclouds"
            timeline.mkdir(parents=True)
            (cloud / "robot").mkdir(parents=True)
            (cloud / "environment").mkdir(parents=True)
            paths = {
                "full": cloud / "frame_0000.ply",
                "robot": cloud / "robot/frame_0000.ply",
                "environment": cloud / "environment/frame_0000.ply",
            }
            for path in paths.values():
                path.write_bytes(b"ply")
            (cloud / "unused.ply").write_bytes(b"ply")
            metadata = rollout / "chunks/step_000000/metadata.json"
            metadata.write_text("{}")
            manifest = {
                "sources": {
                    "imagined": [{
                        "files": {
                            name: os.path.relpath(path, timeline) for name, path in paths.items()
                        }
                    }],
                    "simulation": [],
                }
            }
            (timeline / "manifest.json").write_text(json.dumps(manifest))

            output = root / "bundle.tar.gz"
            package_4d_task(task, output)
            with tarfile.open(output, "r:gz") as archive:
                names = set(archive.getnames())
            prefix = "PnPCounterToCab/0_0_4d/"
            self.assertIn(prefix + "timeline/manifest.json", names)
            self.assertIn(prefix + "chunks/step_000000/metadata.json", names)
            self.assertIn(prefix + "chunks/step_000000/predicted/pointclouds/frame_0000.ply", names)
            self.assertNotIn(prefix + "chunks/step_000000/predicted/pointclouds/unused.ply", names)


if __name__ == "__main__":
    unittest.main()
