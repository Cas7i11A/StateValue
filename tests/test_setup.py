"""Exercise setup control flow with local stubs; never download or install packages."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SetupTest(unittest.TestCase):
    def fixture(self, root):
        shutil.copy(ROOT / "setup.sh", root / "setup.sh")
        shutil.copytree(ROOT / "environment", root / "environment")
        tools = root / "tools"
        tools.mkdir()
        cuda = root / "cuda/bin"
        cuda.mkdir(parents=True)
        self.stub(cuda / "nvcc", "echo 'Cuda compilation tools, release 13.0, V13.0.88'")
        self.stub(tools / "nvidia-smi", 'echo fake-gpu; exit "${FAKE_GPU_EXIT:-0}"')
        for name in ("c++", "make", "pkg-config"):
            self.stub(tools / name, "exit 0")
        self.stub(tools / "uv", '''
printf '%s\\n' "$*" >> "$FAKE_CALLS"
case "$1 ${2:-}" in
  'venv --seed')
    mkdir -p .venv/bin
    printf '#!/bin/bash\\necho fake-runtime-check\\n' > .venv/bin/python
    chmod +x .venv/bin/python ;;
  'pip check') exit "${FAKE_CHECK_EXIT:-0}" ;;
  'pip freeze') echo 'fake-package==1' ;;
  '--version ') echo 'uv 0.13.1' ;;
esac
''')
        return dict(os.environ, PATH=str(tools) + ":" + os.environ["PATH"],
                    CUDA_HOME=str(root / "cuda"), FAKE_CALLS=str(root / "calls"))

    def stub(self, path, body):
        path.write_text("#!/bin/bash\nset -eu\n" + body + "\n")
        path.chmod(0o755)

    def run_setup(self, root, env, *args):
        return subprocess.run(["bash", str(root / "setup.sh"), *args], env=env,
                              capture_output=True, text=True, timeout=10)

    def test_dry_run_does_not_touch_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = self.fixture(root)
            before = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
            result = self.run_setup(root, env, "--dry-run")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--torch-backend cu130", result.stdout)
            self.assertEqual(before, sorted(str(p.relative_to(root)) for p in root.rglob("*")))

    def test_existing_venv_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = self.fixture(root)
            (root / ".venv").mkdir()
            marker = root / ".venv/keep"
            marker.write_text("keep")
            result = self.run_setup(root, env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(".venv already exists", result.stdout)
            self.assertEqual(marker.read_text(), "keep")
            self.assertNotIn("pip install", (root / "calls").read_text())

    def test_bad_driver_stops_before_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = self.fixture(root)
            env["FAKE_GPU_EXIT"] = "1"
            result = self.run_setup(root, env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("NVIDIA GPU/driver unavailable", result.stdout)
            self.assertFalse((root / ".venv").exists())

    def test_install_order_and_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = self.fixture(root)
            result = self.run_setup(root, env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = (root / "calls").read_text()
            self.assertLess(calls.index("python install"), calls.index("venv --seed"))
            self.assertLess(calls.index("pip install"), calls.index("pip check"))
            self.assertIn("--torch-backend cu130", calls)
            record = next((root / "logs").glob("setup-*"))
            self.assertEqual((record / "installed-requirements.txt").read_text(), "fake-package==1\n")
            self.assertTrue((record / "requirements-deploy.txt").exists())
            for name in ("python-version.txt", "os-release.txt", "kernel.txt",
                         "disk.txt", "uv-version.txt", "nvidia-smi.txt", "nvcc.txt"):
                self.assertTrue((record / name).read_text(), name)

    def test_check_never_installs_and_propagates_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = self.fixture(root)
            (root / ".venv/bin").mkdir(parents=True)
            self.stub(root / ".venv/bin/python", "echo fake-runtime-check")
            result = self.run_setup(root, env, "--check")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = (root / "calls").read_text()
            self.assertNotIn("install", calls)
            self.assertNotIn("venv --seed", calls)
            env["FAKE_CHECK_EXIT"] = "9"
            result = self.run_setup(root, env, "--check")
            self.assertEqual(result.returncode, 9)
            self.assertIn("Setup failed", result.stdout)


if __name__ == "__main__":
    unittest.main()
