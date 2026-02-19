"""Scrape SharePoint for download URLs, then download via aria2c.

Usage:
    python scrape_and_download.py                        # generate URLs + download
    python scrape_and_download.py --only 20251113_02 20251204_01
    python scrape_and_download.py --exclude 20251113_01
    python scrape_and_download.py --after 2026-01-01
    python scrape_and_download.py --tree                 # print tree and exit
    python scrape_and_download.py --urls-only            # generate urls.txt, skip download
    python scrape_and_download.py --download-only        # skip scraping, use existing urls.txt
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests
import typer
from loguru import logger

from ent_cv.config import INTERIM_DATA_DIR, EXTERNAL_DATA_DIR, PRIORITY_CASES, RAW_DATA_DIR
from ent_cv.utils import notify

app = typer.Typer()

# ── SharePoint config ────────────────────────────────────────────────────────

_SITE_URL = "https://mtsinai-my.sharepoint.com/personal/turner_baker_mssm_edu"
_FOLDER_REL = (
    "/personal/turner_baker_mssm_edu/Documents/"
    "Pharyvac-Related OR Exposure Case Videos/OR GoPro Videos/"
    "OR Gopro Videos for Jae + Abdel"
)
_BASE_URL = (
    "https://mtsinai-my.sharepoint.com/:v:/r/personal/turner_baker_mssm_edu/"
    "Documents/Pharyvac-Related%20OR%20Exposure%20Case%20Videos/"
    "OR%20GoPro%20Videos/OR%20Gopro%20Videos%20for%20Jae%20+%20Abdel"
)
_SUFFIX = "?csf=1&download=1"
_HEADERS = {"Accept": "application/json;odata=verbose", "User-Agent": "Mozilla/5.0"}

_CASE_ID_RE = re.compile(r"\d{8}_\d{2}")

_MAX_RETRIES = 3
_MIN_FREE_SPACE_GB = 5.0  # Minimum free disk space required

# ── Exceptions ───────────────────────────────────────────────────────────────

class AuthExpiredError(Exception):
    pass

class RateLimitError(Exception):
    pass

# ── SharePoint helpers ───────────────────────────────────────────────────────

def _load_cookies(path: Path) -> dict[str, str]:
    cookies: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 7:
            cookies[parts[5]] = parts[6]
    return cookies


def _api_get(session: requests.Session, endpoint: str) -> list[dict]:
    for attempt in range(1, _MAX_RETRIES + 1):
        resp = session.get(endpoint, headers=_HEADERS, timeout=30)
        
        if resp.status_code == 403:
            raise AuthExpiredError("SharePoint returned 403. Refresh cookies.txt.")
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 60))
            if attempt < _MAX_RETRIES:
                time.sleep(retry_after)
                continue
            raise RateLimitError("Exceeded max retries for 429s.")
            
        resp.raise_for_status()
        return resp.json().get("d", {}).get("results", [])
    return []


def _scrape_tree(session: requests.Session) -> dict[str, list[str]]:
    folders_url = f"{_SITE_URL}/_api/web/GetFolderByServerRelativeUrl('{quote(_FOLDER_REL)}')/Folders"
    folders = sorted(r["Name"] for r in _api_get(session, folders_url))

    tree: dict[str, list[str]] = {}
    for folder in folders:
        rel = f"{_FOLDER_REL}/{folder}"
        files_url = f"{_SITE_URL}/_api/web/GetFolderByServerRelativeUrl('{quote(rel)}')/Files"
        tree[folder] = sorted(r["Name"] for r in _api_get(session, files_url))
    return tree


def _filter_tree(
    tree: dict[str, list[str]],
    only: Optional[list[str]],
    exclude: Optional[list[str]],
    after: Optional[str],
    skip_empty: bool,
) -> dict[str, list[str]]:
    after_dt = datetime.strptime(after, "%Y-%m-%d") if after else None
    filtered: dict[str, list[str]] = {}
    for folder, files in tree.items():
        if skip_empty and not files: continue
        if only and folder not in only: continue
        if exclude and folder in exclude: continue
        if after_dt:
            try:
                if datetime.strptime(folder[:8], "%Y%m%d") < after_dt: continue
            except ValueError:
                pass
        filtered[folder] = files
    return filtered


def _build_urls(tree: dict[str, list[str]]) -> list[str]:
    return [
        f"{_BASE_URL}/{folder}/{fname}{_SUFFIX}"
        for folder, files in sorted(tree.items())
        for fname in files
    ]

# ── Download helper ──────────────────────────────────────────────────────────

def _download_results_fn(result: tuple[int, list[str]]) -> tuple[str, list[Path]]:
    num_cases, case_ids = result
    lines = ["  Cases:", *[f"    {c}" for c in sorted(case_ids)]]
    return "\n".join(lines), []


def _run_downloads(urls: list[str], output_dir: Path, cookies_path: Path) -> tuple[int, list[str]]:
    case_ids = sorted({m.group() for url in urls for m in [_CASE_ID_RE.search(url)] if m})
    if not case_ids:
        logger.error("No case IDs found in URL list")
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []

    for i, case_id in enumerate(case_ids, 1):
        # Prevent the "fake 429" loop by checking local disk space first
        free_space_gb = shutil.disk_usage(output_dir).free / (1024**3)
        if free_space_gb < _MIN_FREE_SPACE_GB:
            logger.error(f"Insufficient disk space! Only {free_space_gb:.2f} GB free. Aborting.")
            raise typer.Exit(code=1)

        case_dir = output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        case_urls = [u for u in urls if case_id in u]
        
        logger.info(f"[{i}/{len(case_ids)}] {case_id} — {len(case_urls)} file(s) ({free_space_gb:.1f} GB free)")

        success = False
        for attempt in range(1, _MAX_RETRIES + 1):
            process = subprocess.Popen(
                [
                    "aria2c",
                    "--load-cookies", str(cookies_path),
                    "--save-cookies", str(cookies_path),
                    "--continue=true",
                    "--console-log-level=notice",
                    "--summary-interval=2",
                    "--max-concurrent-downloads=1",  # Keeps connections clean without messy Python loops
                    "--max-connection-per-server=1",
                    "-d", str(case_dir),
                    "-i", "-",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Bulk pass URLs for this case and close stdin to start the download
            process.stdin.write("\n".join(case_urls) + "\n")
            process.stdin.close()

            # Stream live output
            output_log = []
            for line in process.stdout:
                print(line, end="")
                output_log.append(line)

            process.wait()

            if process.returncode == 0:
                success = True
                break

            stderr_str = "".join(output_log)
            if "status=403" in stderr_str:
                logger.error(f"Auth expired (403) on {case_id}. Refresh cookies.txt.")
                raise AuthExpiredError()
            
            # Simple fallback error handling
            if attempt < _MAX_RETRIES:
                logger.warning(f"Failed (exit code {process.returncode}). Retrying in 10s...")
                time.sleep(10)
            else:
                logger.error(f"Failed on {case_id} after {_MAX_RETRIES} attempts.")

        if success:
            logger.success(f"  {case_id}: done")
        else:
            failed.append(case_id)

    if failed:
        logger.error(f"Failed ({len(failed)}): {', '.join(sorted(failed))}")
        raise typer.Exit(code=1)

    return len(case_ids), case_ids

# ── CLI ──────────────────────────────────────────────────────────────────────

@app.command()
@notify("Downloads", results_fn=_download_results_fn)
def main(
    urls_file:     Path                = typer.Option(INTERIM_DATA_DIR / "urls.txt"),
    output_dir:    Path                = typer.Option(RAW_DATA_DIR),
    cookies_path:  Path                = typer.Option(EXTERNAL_DATA_DIR / "cookies.txt"),
    only:          Optional[list[str]] = typer.Option(PRIORITY_CASES),
    exclude:       Optional[list[str]] = typer.Option(None),
    after:         Optional[str]       = typer.Option(None, help="YYYY-MM-DD"),
    skip_empty:    bool                = typer.Option(True),
    tree:          bool                = typer.Option(False, "--tree"),
    urls_only:     bool                = typer.Option(False, "--urls-only"),
    download_only: bool                = typer.Option(False, "--download-only"),
):
    if download_only:
        if not urls_file.exists():
            raise typer.Exit(code=1)
        urls = [l.strip() for l in urls_file.read_text().splitlines() if l.strip()]
    else:
        with requests.Session() as session:
            session.cookies.update(_load_cookies(cookies_path))
            try:
                raw_tree = _scrape_tree(session)
            except (AuthExpiredError, RateLimitError) as e:
                logger.error(str(e))
                raise typer.Exit(code=1)

        filtered = _filter_tree(raw_tree, only, exclude, after, skip_empty)

        if tree:
            print(f"\nOR Gopro Videos for Jae + Abdel/  ({len(filtered)} folders)\n")
            for folder, files in sorted(filtered.items()):
                print(f"├── {folder}/  ({len(files)} files)")
                for j, f in enumerate(files):
                    prefix = "│   ├──" if j < len(files) - 1 else "│   └──"
                    print(f"{prefix} {f}")
            raise typer.Exit()

        urls = _build_urls(filtered)
        urls_file.parent.mkdir(parents=True, exist_ok=True)
        urls_file.write_text("\n".join(urls) + "\n")

        if urls_only:
            raise typer.Exit()

    try:
        return _run_downloads(urls, output_dir, cookies_path)
    except AuthExpiredError:
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()