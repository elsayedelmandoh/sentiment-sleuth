from pathlib import Path
import os
import joblib
from src.config import settings

# Read asset list and cache dir from settings (single source of truth)
ASSET_PATHS = list(settings.ASSET_PATHS)
CACHE_DIR = Path(settings.ASSET_CACHE_DIR)


def load_assets_hf():
    """Minimal asset loader that prefers local files and can download from Hugging Face Hub.
    Returns a tuple of loaded objects or None for missing entries.
    """
    # optional HF Hub download function
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except Exception:
        hf_hub_download = None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    hf_repo = os.environ.get('HF_ASSETS_REPO')
    hf_repo_type = os.environ.get('HF_ASSETS_REPO_TYPE') or None

    assets = []
    for p in ASSET_PATHS:
        lp = Path(p)
        loaded = None
        if lp.exists():
            try:
                loaded = joblib.load(lp)
                print(f"Loaded local asset: {lp}")
            except Exception as e:
                print(f"Failed to load local asset {lp}: {e}")
                loaded = None
        else:
            print(f"Local asset not found: {lp}")

        if loaded is None and hf_hub_download is not None and hf_repo:
            filename = lp.name
            try:
                print(f"Attempting to download '{filename}' from HF repo '{hf_repo}'...")
                if hf_repo_type:
                    downloaded_path = hf_hub_download(repo_id=hf_repo, filename=filename, repo_type=hf_repo_type)
                else:
                    downloaded_path = hf_hub_download(repo_id=hf_repo, filename=filename)

                downloaded = Path(downloaded_path)
                target = CACHE_DIR / filename
                try:
                    if not target.exists():
                        downloaded.replace(target)
                except Exception:
                    try:
                        import shutil
                        if not target.exists():
                            shutil.copy2(downloaded, target)
                    except Exception:
                        pass

                load_from = target if target.exists() else downloaded
                loaded = joblib.load(load_from)
                print(f"Downloaded and loaded asset: {load_from}")
            except Exception as e:
                print(f"Failed to download/load '{filename}' from HF Hub: {e}")
                loaded = None
        else:
            if loaded is None:
                if hf_hub_download is None:
                    print("huggingface_hub not installed; skipping HF download attempts.")
                elif not hf_repo:
                    print("Environment variable 'HF_ASSETS_REPO' not set; skipping HF download attempts.")

        assets.append(loaded)

    return tuple(assets)
