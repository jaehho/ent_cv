# Scraping Mount Sinai OneDrive

This project uses `aria2c` and SharePoint cookies to download large OR GoPro
videos from a secure OneDrive into `data/raw/`.

## Files

- `scripts/scrape/generate_urls.py`  
  Generates `scripts/scrape/urls.txt` from a single Part10 OneDrive "Copy link" URL.

- `scripts/scrape/run_downloads.sh`  
  Runs the full pipeline and stores videos in `data/raw/20251124_01/`.

- `scripts/scrape/cookies.txt`  
  Netscape-style cookie file (not committed). Must contain `FedAuth`, `rtFa`, etc.
  use `Get cookies.txt` browser extension to export cookies.

## Usage

1. Update `PART10_URL` and `PART_COUNT` in `scripts/scrape/generate_urls.py`.
2. Create `scripts/scrape/cookies.txt` from browser cookies for
   `mtsinai-my.sharepoint.com`.
3. From the project root:

   ```bash
   make scrape
   ```

4. Resulting MP4s appear in data/raw/20251124_01/, ready for further
processing by the CCDS pipeline.
