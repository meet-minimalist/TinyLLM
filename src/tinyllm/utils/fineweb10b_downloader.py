# Reference: https://github.com/KellerJordan/modded-nanogpt/blob/master/data/cached_fineweb10B.py

import argparse
import os
import sys

from huggingface_hub import hf_hub_download

REPO_ID = "kjj0/fineweb10B-gpt2"
ROOT_DIR = "./"
DEFAULT_NUM_CHUNKS = 103


def get_local_dir(path: str | None = None) -> str:
    if path:
        return path
    default_dir = os.path.join(ROOT_DIR, "fineweb10B")
    os.makedirs(default_dir, exist_ok=True)
    return default_dir


LOCAL_DIR = get_local_dir()


def get(fname: str, local_dir: str = LOCAL_DIR, force: bool = False) -> str:
    local_path = os.path.join(local_dir, fname)
    if os.path.exists(local_path) and not force:
        print(f"Skipping {fname} (already exists)")
        return local_path

    print(f"Downloading {fname}...")
    try:
        hf_hub_download(
            repo_id=REPO_ID,
            filename=fname,
            repo_type="dataset",
            local_dir=local_dir,
        )
    except Exception as e:
        print(f"Error downloading {fname}: {e}", file=sys.stderr)
        raise

    return local_path


def download_val(local_dir: str = LOCAL_DIR, force: bool = False) -> str:
    return get("fineweb_val_000000.bin", local_dir=local_dir, force=force)


def download_train_chunks(
    num_chunks: int = DEFAULT_NUM_CHUNKS,
    local_dir: str = LOCAL_DIR,
    force: bool = False,
) -> list[str]:
    return [
        get(f"fineweb_train_{i:06d}.bin", local_dir=local_dir, force=force)
        for i in range(1, num_chunks + 1)
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Download Fineweb10B GPT-2 tokens"
    )
    parser.add_argument(
        "num_chunks",
        nargs="?",
        type=int,
        default=DEFAULT_NUM_CHUNKS,
        help=f"Number of training chunks to download (default: {DEFAULT_NUM_CHUNKS})",
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=None,
        help="Custom download directory (default: /mnt/d/d/DeepLearning/datasets/fineweb10B)",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Re-download existing files"
    )
    args = parser.parse_args()

    local_dir = get_local_dir(args.path)

    download_val(local_dir=local_dir, force=args.force)
    download_train_chunks(
        num_chunks=args.num_chunks, local_dir=local_dir, force=args.force
    )
    print(f"Downloaded {args.num_chunks + 1} files total to {local_dir}")


if __name__ == "__main__":
    main()
