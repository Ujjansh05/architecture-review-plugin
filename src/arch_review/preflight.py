"""No installation, model import, deletion, or cloud provisioning during inspection."""
import json
import shutil
import subprocess
from pathlib import Path

import psutil


def inspect_hardware(workspace):
    workspace = Path(workspace).resolve()
    if not workspace.is_dir():
        raise ValueError("The selected workspace must already exist")
    free = shutil.disk_usage(workspace).free
    gpu, error = [], None
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            result = subprocess.run([executable, "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=10, check=True)
            for line in result.stdout.splitlines():
                name, total, available, driver = [v.strip() for v in line.split(",")]
                gpu.append({"name": name, "total_vram_mib": int(total), "free_vram_mib": int(available), "driver": driver})
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            error = type(exc).__name__
    return {"workspace": str(workspace), "free_disk_bytes": free, "required_workspace_bytes": 30 * 1024 ** 3,
            "large_model_storage_ready": free >= 30 * 1024 ** 3, "ram_bytes": psutil.virtual_memory().total,
            "available_ram_bytes": psutil.virtual_memory().available, "gpus": gpu, "gpu_probe_error": error,
            "cuda_runtime_tested": False, "rental_budget_usd": 15,
            "next_action": "Run an isolated CUDA/model preflight" if free >= 30 * 1024 ** 3 else
                           "Provide a workspace with at least 30 GiB free before downloading larger models or training.",
            "notice": "Hardware availability does not prove a model or training run fits. No files were removed or compute rented."}


if __name__ == "__main__":
    print(json.dumps(inspect_hardware(Path.cwd()), indent=2))
