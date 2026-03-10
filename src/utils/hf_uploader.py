"""Hugging Face asset uploader utilities.

Provides a reusable `upload_assets` function that can be imported by
CLI wrappers or CI, plus a small `main()` so it can still be invoked
as a script for local usage.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional

from dotenv import load_dotenv

load_dotenv()

try:
    from huggingface_hub import HfApi, login
except Exception:  # pragma: no cover - runtime environment may not have hf installed
    HfApi = None
    login = None


def upload_assets(
    token: Optional[str],
    repo_id: str,
    repo_type: str = "model",
    paths: Optional[Iterable[Path]] = None,
    root: Optional[Path] = None,
):
    """Upload the provided files (Paths) to a Hugging Face `repo_id`.

    - `token` is the HF token (string) or None to rely on existing login.
    - `paths` is an iterable of Path objects (relative to `root` if provided)
      If omitted, defaults to files under `data/models/` and `data/vectorizers/`.
    - Returns a list of tuples (path, success_bool, message).
    """
    if HfApi is None:
        raise RuntimeError("huggingface_hub is not installed")

    if token:
        # prefer not to persist credentials to git-credential here
        login(token=token, add_to_git_credential=False)

    api = HfApi()

    root = Path(root or Path.cwd())

    if paths is None:
        candidates = list(root.glob("data/models/**/*")) + list(root.glob("data/vectorizers/**/*"))
        files = [p for p in candidates if p.is_file()]
    else:
        files = [p if p.is_absolute() else (root / p) for p in paths]
        files = [p for p in files if p.exists() and p.is_file()]

    results: List[tuple] = []

    if not files:
        return results

    for f in files:
        path_in_repo = str(f.relative_to(root)).replace("\\", "/")
        try:
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
            )
            results.append((f, True, "OK"))
        except Exception as exc:
            results.append((f, False, str(exc)))

    return results


def main():
    token = os.getenv("HUGGINGFACE_API_KEY")
    repo_id = os.getenv("HF_ASSETS_REPO")
    repo_type = os.getenv("HF_ASSETS_REPO_TYPE") or "model"

    if not repo_id:
        print("HF_ASSETS_REPO not set in environment. Aborting.")
        return

    try:
        results = upload_assets(token=token, repo_id=repo_id, repo_type=repo_type)
    except Exception as e:
        print(f"Upload failed: {e}")
        return

    if not results:
        print("No files found to upload.")
        return

    print(f"Uploaded {len(results)} candidate files to {repo_id} (repo_type={repo_type})")
    for f, ok, msg in results:
        status = "OK" if ok else f"FAILED: {msg}"
        print(f"{f} -> {status}")


if __name__ == "__main__":
    main()
