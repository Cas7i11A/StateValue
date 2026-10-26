# Deploy StateValue on a new VM

This procedure targets Linux x86_64, Python 3.12.15, and CUDA toolkit 13.0.
The reference environment uses Ubuntu 22.04.5 and an NVIDIA L4. The deployment
manifest and setup control flow have local checks; a complete installation on a
clean GPU VM has **not yet been validated**. Package availability and native-wheel
compatibility must be verified on the target VM.

## 1. Prepare the VM

Use an NVIDIA GPU VM with a working driver and sufficient CPU RAM for the 4 GB
LMCache cache plus model-serving overhead. Allow disk space for Python packages,
model weights, download/build caches, and traces; `.venv` alone was about 8.1 GB
on the working machine. Keep hardware and service settings matched for comparisons.

On GCP, confirm billing, the Compute Engine API, applicable project-wide and
regional GPU quotas, and L4 availability in the selected zone before creating the
VM. G2 machine types include their GPUs; record the actual machine type rather
than adding an unrelated GPU specification. See [GCP GPU machine types](https://docs.cloud.google.com/compute/docs/gpus).
Record the base image, machine type, zone, CPU/RAM and disk settings; these values
are not inferred from the historical package snapshot.

On Ubuntu, install the basic shell/build tools:

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates python3 build-essential \
  pkg-config libnuma-dev tmux pciutils mokutil
uname -m
cat /etc/os-release
uname -r
free -h
df -h /
lspci | grep -i nvidia
mokutil --sb-state
```

Install a suitable NVIDIA driver and **CUDA toolkit 13.0** using your GPU VM image
or the official [NVIDIA driver guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/)
and [CUDA 13.0 Linux installation guide](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-installation-guide-linux/index.html).
Select toolkit 13.0 rather than blindly installing the latest CUDA metapackage.
Driver/kernel setup is specific to the VM image and may require a reboot; it is
not automated by `setup.sh`. The historical driver was 615.71.09; this is a record,
not a claim that this exact driver is required or available on every image.

### GCP driver installation when needed

If `nvidia-smi` already works, keep the driver and proceed to toolkit verification.
For a VM with **Secure Boot disabled**, the [GCP installation script](https://docs.cloud.google.com/compute/docs/gpus/install-drivers-gpu)
provides the following path. These commands run on the VM, not the local laptop:

```bash
mkdir -p ~/statevalue-install
cd ~/statevalue-install
curl -fSsL \
  https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz \
  -o cuda_installer.pyz
sudo python3 cuda_installer.pyz install_driver \
  --installation-mode=repo \
  --installation-branch=prod
```

If the installer reboots the VM, reconnect and rerun the same install command to
resume. Reboot when required by the installation, then confirm `nvidia-smi` works.
The `prod` branch and `latest` installer can change; record the installer checksum
and resulting driver version for the new VM. If Secure Boot is enabled, follow
the signed-driver procedure in the same GCP guide instead. If its state cannot
be determined with `mokutil`, check the VM's Shielded VM settings.

### Install the toolkit for FlashInfer compilation

The CUDA libraries in Python wheels do not by themselves supply the toolkit
compiler required by this workflow. On **Ubuntu 22.04 x86_64**, install toolkit
13.0 if it is missing (choose the matching NVIDIA repository for other OS versions):

```bash
mkdir -p ~/statevalue-install
cd ~/statevalue-install
curl -fSsL \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
  -o cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-0
```

This addresses the previously observed FlashInfer `Could not find nvcc` error.
The installer download is not a runtime dependency. `statevalue.sh` automatically
sets `CUDA_HOME` and updates `PATH`; manual commands in a new shell must do so too.

Confirm both commands work:

```bash
nvidia-smi
/usr/local/cuda-13.0/bin/nvcc --version
```

The `CUDA` value shown by NVIDIA-SMI and the installed toolkit version can differ.
The latter must show `release 13.0` for this deployment manifest.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/); the reference
environment uses version 0.13.1:

```bash
curl -LsSf https://astral.sh/uv/0.13.1/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

The VM needs outbound access to Python package indexes, uv's Python downloads,
the PyTorch CUDA 13.0 index, and Hugging Face for the model and traces. The local
workflow needs no public port 8000 or 5555.

## 2. Clone and inspect the plan

Clone your existing repository into `~/statevalue` (replace `<repository-url>`):

```bash
git clone <repository-url> ~/statevalue
cd ~/statevalue
bash setup.sh --dry-run
```

`--dry-run` prints commands without changing files or resolving/downloading
packages. It does not establish that dependencies can be installed.

## 3. Install the Python environment

```bash
bash setup.sh
```

The script:

1. Checks platform, shell/build tools, CUDA toolkit 13.0 and NVIDIA driver access.
2. Requires `.venv` to be absent, then downloads Python 3.12.15 and creates it.
3. Installs `environment/requirements-deploy.txt` with `uv --torch-backend cu130`.
4. Runs dependency consistency, package-version, native-extension import and small GPU-computation checks.
5. Captures logs, input manifests and resolved installed versions in
   `logs/setup-<UTC timestamp>-<unique suffix>/`, including Python version, CPU/RAM,
   disk information and Git commit/status when the project is a Git checkout.

The manifest pins the six direct dependencies below and constrains transitive
dependencies with `requirements-lmcache-working.txt`:

| Package | Version |
|---|---|
| torch | 2.13.0+cu130 |
| vllm | 0.30.0 |
| lmcache | 0.5.5 |
| flashinfer-python | 0.6.18.post1 |
| numpy | 2.2.6 |
| zstandard | 0.25.0 |

The earlier `requirements-vllm-working.txt` is historical and is not installed.
The constraints file limits versions of dependencies that are needed; it does not
force installation of every unrelated package in the historical snapshot. This
manifest is not a hash-locked archive or proof of clean-machine reproducibility.
The explicit CUDA backend avoids selecting a CPU or different CUDA build of
PyTorch. See the [vLLM GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
and [LMCache installation guide](https://docs.lmcache.ai/getting_started/installation.html).

An existing `.venv` is never replaced or modified by install mode. To inspect it:

```bash
bash setup.sh --check
```

This validates the existing environment without installing packages. If an
installation fails partway through, keep its log; move the partial `.venv` aside
manually before retrying. Do not run setup against a live serving environment.

For a nondefault CUDA toolkit location:

```bash
CUDA_HOME=/path/to/cuda-13.0 bash setup.sh
```

Pass the same `CUDA_HOME` when starting services. If cloning outside `~/statevalue`,
set `STATEVALUE_DIR` for each `statevalue.sh` invocation; setup itself always uses
the directory containing `setup.sh`.

## 4. Validate the serving workflow

Installation resolves the integrated environment in one step. It does not need
the earlier manual sequence of installing vLLM, creating an environment, then
adding LMCache. Do not manually create `.venv` before running install mode;
`setup.sh` deliberately refuses to replace one. If following the manual reference
procedure instead, use `--check` afterwards to inspect the environment.

The validation helper explicitly imports `lmcache.cuda_ops` and
`lmcache.lmcache_native`; this version uses `cuda_ops`, not `lmcache.c_ops`.
Successful imports check native-library loading, but do not prove an end-to-end
KV transfer. A successful small PyTorch operation also does not exercise the
FlashInfer sampler's JIT path; validate real generation next.

### Optional: isolate plain vLLM before checking the connector

If integrated startup fails, first test vLLM without the LMCache connector.
In Terminal B, with no other process listening on port 8000:

```bash
cd ~/statevalue
source .venv/bin/activate
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:$PATH"
mkdir -p logs
set -o pipefail
VLLM_USE_FLASHINFER_SAMPLER=1 \
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 --port 8000 --dtype half \
  --max-model-len 8192 --gpu-memory-utilization 0.7 \
  --enable-prefix-caching \
  2>&1 | tee "logs/vllm-plain-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
```

In Terminal C, wait for `bash statevalue.sh health` to report HTTP 200, then send
one request. `-f` makes an HTTP error fail the command:

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","messages":[{"role":"user","content":"Explain KV caching in one sentence."}],"temperature":0,"max_tokens":80}'
```

Confirm there is generated content, then stop plain vLLM with `Ctrl+C` before
starting integrated vLLM below. If diagnosing `Could not find nvcc`, verify
`CUDA_HOME` and `PATH` in Terminal B first. For this manual diagnostic command,
temporarily setting `VLLM_USE_FLASHINFER_SAMPLER=0` may isolate a sampler issue;
record that change and repeat validation with the project's normal sampler
setting before comparing performance. The normal startup script sets it to 1.

### Integrated LMCache and vLLM

Run the CPU-only regression checks:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

In separate terminals, from the project root:

```bash
# Terminal A; wait for startup before starting B
bash statevalue.sh lmcache

# Terminal B
bash statevalue.sh vllm

# Terminal C; repeat health until HTTP 200, then run smoke
bash statevalue.sh health
bash statevalue.sh smoke
```

The first vLLM startup downloads `Qwen/Qwen2.5-1.5B-Instruct` if not cached.
Model download time is not part of setup validation. The model revision is not
currently pinned; record the revision/cache snapshot for reproducible comparisons.
Inspect the printed service log paths and the smoke output directory. Healthy
serving alone does not prove KV restoration. To check restoration, preserve
LMCache, restart vLLM and replay matching input, then inspect `Retrieved` records;
controlled GPU-pressure validation remains a separate experiment.

### Dependency troubleshooting

The earlier manual installation encountered an extra
`opentelemetry-exporter-otlp-common==0.66b1` package causing dependency conflicts.
This package is absent from the integrated snapshot used as constraints. Do not
add an unconditional uninstall step to setup. If the same issue occurs, retain
`uv pip check` output, inspect installed requirements/reverse dependencies, and
remove the package only after confirming that it is the conflicting unused extra.
A fresh environment and a single constrained resolution reduce environment drift;
they are not proof that the same conflict can never recur.

## 5. Restore data and keep records

Copy or download traces into `data/freeinference/`, convert them with
`trace_workload.py`, and replay them as described in README.md. Neither traces,
`.venv`, downloaded models, `logs/`, nor `runs/` are transferred by `git clone`.
Back up important experiment outputs and logs separately.

For each new VM, retain its base image/machine type, CPU/RAM/disk configuration,
setup logs, installed package list, Git commit and working-tree state, model
revision, source trace hashes and service/workload settings. Update the system
record after a successful deployment; do not overwrite historical GPU evidence
with an unvalidated claim.
