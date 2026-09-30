"""
Full training-state checkpoints for resumable runs.

A checkpoint holds everything needed to continue a run bit-for-bit where it
stopped: model, optimizer, LR scheduler, GradScaler, trainer counters (step,
tokens, epoch, batches consumed = data position) and RNG states.

Layout under a run directory::

    <run_dir>/checkpoints/step_00001200.pt
    <run_dir>/checkpoints/latest          # text file: name of the newest .pt

Free-tier machines (Kaggle, Colab) lose their disk when a session ends. Set
``hub_repo_id`` to also push the newest checkpoint to a private Hugging Face
model repo, and to pull it back when the local directory is empty.
"""

import glob
import os
import random
import re
import threading

import numpy as np
import torch

from src.tinyllm.logger.logger_utils import logger

_STEP_RE = re.compile(r"step_(\d+)\.pt$")
_HUB_LATEST = "checkpoints/latest.pt"


def capture_rng() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        cuda_states = state["cuda"]
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)


def load_checkpoint(path: str) -> dict:
    # weights_only=False: the checkpoint holds RNG states and plain Python
    # counters, and it is a file this code wrote itself.
    return torch.load(path, map_location="cpu", weights_only=False)


class CheckpointManager:
    def __init__(
        self,
        run_dir: str,
        keep_last: int = 3,
        keep_every_steps: int = 0,
        hub_repo_id: str | None = None,
        is_main: bool = True,
    ):
        self.ckpt_dir = os.path.join(run_dir, "checkpoints")
        self.keep_last = keep_last
        self.keep_every_steps = keep_every_steps
        self.hub_repo_id = hub_repo_id
        self.is_main = is_main
        self._upload_thread: threading.Thread | None = None
        if is_main:
            os.makedirs(self.ckpt_dir, exist_ok=True)

    # --- save ----------------------------------------------------------------
    def save(self, state: dict, step: int) -> str | None:
        """Atomically write ``state`` as step_<step>.pt. Main rank only."""
        if not self.is_main:
            return None
        name = f"step_{step:08d}.pt"
        path = os.path.join(self.ckpt_dir, name)
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)  # a crash mid-write never leaves a torn file

        pointer_tmp = os.path.join(self.ckpt_dir, "latest.tmp")
        with open(pointer_tmp, "w") as f:
            f.write(name)
        os.replace(pointer_tmp, os.path.join(self.ckpt_dir, "latest"))

        self._prune()
        logger.info(f"Saved checkpoint: {path}")
        if self.hub_repo_id:
            self._upload_async(path, step)
        return path

    def _prune(self) -> None:
        """Keep the newest ``keep_last`` checkpoints plus every milestone."""
        paths = sorted(self._local_checkpoints(), key=lambda p: p[0])
        removable = [
            (s, p)
            for s, p in paths
            if not (self.keep_every_steps and s % self.keep_every_steps == 0)
        ]
        excess = len(removable) - self.keep_last
        for _, p in removable[: max(0, excess)]:
            os.remove(p)

    def _local_checkpoints(self) -> list[tuple[int, str]]:
        out = []
        for p in glob.glob(os.path.join(self.ckpt_dir, "step_*.pt")):
            m = _STEP_RE.search(p)
            if m:
                out.append((int(m.group(1)), p))
        return out

    # --- hub sync ------------------------------------------------------------
    def _upload_async(self, path: str, step: int) -> None:
        # One upload at a time: a new save waits for the previous upload, so
        # uploads never pile up on a slow link.
        self.wait()
        milestone = bool(
            self.keep_every_steps and step % self.keep_every_steps == 0
        )

        def _upload():
            try:
                from huggingface_hub import HfApi

                api = HfApi()
                api.create_repo(self.hub_repo_id, private=True, exist_ok=True)
                api.upload_file(
                    path_or_fileobj=path,
                    path_in_repo=_HUB_LATEST,
                    repo_id=self.hub_repo_id,
                    commit_message=f"checkpoint step {step}",
                )
                if milestone:
                    api.upload_file(
                        path_or_fileobj=path,
                        path_in_repo=f"checkpoints/{os.path.basename(path)}",
                        repo_id=self.hub_repo_id,
                        commit_message=f"milestone step {step}",
                    )
                logger.info(
                    f"Uploaded checkpoint step {step} to {self.hub_repo_id}"
                )
            except Exception as e:  # never kill training over an upload
                logger.error(f"Checkpoint upload to hub failed: {e}")

        self._upload_thread = threading.Thread(target=_upload, daemon=False)
        self._upload_thread.start()

    def wait(self) -> None:
        """Block until a pending hub upload finishes (call before exit)."""
        if self._upload_thread is not None:
            self._upload_thread.join()
            self._upload_thread = None

    # --- resume --------------------------------------------------------------
    def latest_path(self) -> str | None:
        """Newest local checkpoint; falls back to the hub copy if configured."""
        pointer = os.path.join(self.ckpt_dir, "latest")
        if os.path.exists(pointer):
            with open(pointer) as f:
                path = os.path.join(self.ckpt_dir, f.read().strip())
            if os.path.exists(path):
                return path
        local = self._local_checkpoints()
        if local:
            return max(local)[1]
        # A previous hub download lands at <run_dir>/checkpoints/latest.pt.
        downloaded = os.path.join(os.path.dirname(self.ckpt_dir), _HUB_LATEST)
        if os.path.exists(downloaded):
            return downloaded
        # Only the main rank downloads; the caller puts a barrier after this
        # and the other ranks then find the file above.
        if self.hub_repo_id and self.is_main:
            return self._download_latest()
        return None

    def _download_latest(self) -> str | None:
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id=self.hub_repo_id,
                filename=_HUB_LATEST,
                local_dir=os.path.dirname(self.ckpt_dir),
            )
            logger.info(f"Downloaded latest checkpoint from {self.hub_repo_id}")
            return path
        except Exception as e:
            logger.warning(f"No checkpoint on hub {self.hub_repo_id}: {e}")
            return None
