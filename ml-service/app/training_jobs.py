"""Run training outside inference threads and activate only verified exports."""
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock, Thread

from app.config import get_settings

ROOT = Path(__file__).resolve().parents[1]
_LOCK = Lock()
_STATUS = {"status": "idle", "version": None, "message": None, "completed_folds": 0}
logger = logging.getLogger(__name__)


def status():
    with _LOCK:
        return {**_STATUS, "enabled": bool(get_settings().ml_training_token)}


def _run(version, root):
    env = {**os.environ, "PYTHONUTF8": "1", "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2"}
    args = ["--version", version, "--artifact-root", str(root)]
    try:
        with subprocess.Popen(
            [sys.executable, "-u", "-m", "training.train", "--mode", "train", "--quiet", *args],
            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8",
        ) as proc:
            tail = []
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                tail = (tail + [line])[-12:]
                logger.info("Training %s: %s", version, line)
                if line.startswith(("CV ", "Selection locked", "Independent Test")):
                    with _LOCK:
                        _STATUS["message"] = line
                        _STATUS["completed_folds"] += int(line.startswith("CV "))
            if proc.wait():
                raise RuntimeError(" ".join(tail))
        with _LOCK:
            _STATUS["message"] = "Đang kiểm tra artifact và kích hoạt phiên bản"
        activation = subprocess.run(
            [sys.executable, "-m", "training.activate", *args],
            cwd=ROOT, env=env, capture_output=True, encoding="utf-8",
        )
        if activation.returncode:
            raise RuntimeError("Activation failed: " + activation.stderr[-1200:])
    except (OSError, RuntimeError) as exc:
        logger.error("Training %s failed: %s", version, exc)
        with _LOCK:
            _STATUS.update(status="failed", message=str(exc)[-1500:])
    else:
        with _LOCK:
            _STATUS.update(status="succeeded", message="Đã đánh giá và kích hoạt phiên bản")


def start():
    with _LOCK:
        if _STATUS["status"] == "running":
            return None
        version = datetime.now(timezone.utc).strftime("home-credit-%Y%m%dT%H%M%S%fZ")
        _STATUS.update(status="running", version=version, message="Đang đọc dữ liệu Home Credit", completed_folds=0)
        root = Path(get_settings().artifact_dir).resolve()
        Thread(target=_run, args=(version, root), daemon=True).start()
        return {**_STATUS, "enabled": True}
