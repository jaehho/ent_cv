#!/usr/bin/env python3
"""
scripts/onedrive/generate_urls.py

Generate OneDrive direct-download URLs (Part1..PartN) into urls.txt for aria2c.
"""

from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

PART10_URL = (
    "https://mtsinai-my.sharepoint.com/:v:/r/personal/turner_baker_mssm_edu/Documents/"
    "Pharyvac-Related%20OR%20Exposure%20Case%20Videos/OR%20GoPro%20Videos/OR%20Videos%20for%20Jae/"
    "20251124_01/20251124_01_Part1.mp4?csf=1&web=1"
)

PART_COUNT = 10
OUTPUT_FILE = "urls.txt"   # will live in scripts/onedrive/


# --------------------------------------------------------------------------
# INTERNALS
# --------------------------------------------------------------------------

def make_part_url(part_number: int) -> str:
    before, after = PART10_URL.rsplit("Part1.mp4", 1)
    return f"{before}Part{part_number}.mp4{after}"


def to_download_url(url: str) -> str:
    split = urlsplit(url)
    qs = dict(parse_qsl(split.query))
    qs.pop("web", None)
    qs["download"] = "1"
    new_query = urlencode(qs)
    return urlunsplit(split._replace(query=new_query))


def main():
    urls = []
    for i in range(1, PART_COUNT + 1):
        web_url = make_part_url(i)
        dl_url = to_download_url(web_url)
        urls.append(dl_url)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for u in urls:
            f.write(u + "\n")

    print(f"Generated {len(urls)} URLs into {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
