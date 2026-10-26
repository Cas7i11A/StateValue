"""Validate the declared runtime and run a small CUDA operation; no model download."""
import importlib
from importlib.metadata import version
from pathlib import Path
import sys


def main():
    if sys.version_info[:3] != (3, 12, 15):
        raise RuntimeError(f"Expected Python 3.12.15, got {sys.version.split()[0]}")
    manifest = Path(__file__).with_name("requirements-deploy.txt")
    for line in manifest.read_text().splitlines():
        if not line or line.startswith(("#", "-")):
            continue
        name, expected = line.split("==", 1)
        actual = version(name)
        if actual != expected:
            raise RuntimeError(f"Expected {name}=={expected}, got {actual}")
        print(f"{name}=={actual}", flush=True)
    import torch
    if torch.version.cuda != "13.0":
        raise RuntimeError(f"Expected PyTorch CUDA 13.0, got {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access an NVIDIA GPU")
    value = torch.ones((16, 16), dtype=torch.float16, device="cuda")
    result = value @ value
    torch.cuda.synchronize()
    if result[0, 0].item() != 16:
        raise RuntimeError("CUDA matrix multiplication produced an unexpected result")
    for module in ("vllm", "flashinfer", "zstandard", "lmcache.cuda_ops", "lmcache.lmcache_native"):
        importlib.import_module(module)
        print(f"Import OK: {module}", flush=True)
    connector = importlib.import_module("lmcache.integration.vllm.lmcache_mp_connector")
    if not hasattr(connector, "LMCacheMPConnector"):
        raise RuntimeError("LMCacheMPConnector is missing")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print("CUDA computation and imports passed; serving/KV restore not tested.", flush=True)


if __name__ == "__main__":
    main()
