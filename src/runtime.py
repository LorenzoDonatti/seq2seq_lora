"""Explicit device selection and reproducible execution."""
import os
import random
import numpy as np
import torch


def start_run_log(output_dir, arguments):
    """Mirror progress/errors to an exclusive run log and preserve CLI arguments."""
    import atexit
    import json
    import sys
    from pathlib import Path
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    log = (directory / "run.log").open("x", encoding="utf-8", buffering=1)
    stdout, stderr = sys.stdout, sys.stderr

    class Mirror:
        def __init__(self, stream):
            self.stream = stream

        def write(self, text):
            self.stream.write(text)
            log.write(text)
            return len(text)

        def flush(self):
            self.stream.flush()
            log.flush()

        def isatty(self):
            return False

    sys.stdout, sys.stderr = Mirror(stdout), Mirror(stderr)
    def close():
        sys.stdout, sys.stderr = stdout, stderr
        log.close()
    atexit.register(close)
    (directory / "run_arguments.json").write_text(json.dumps(arguments, indent=2))


def resolve_device(requested="auto"):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; check the driver and CUDA PyTorch build.")
    return requested


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
