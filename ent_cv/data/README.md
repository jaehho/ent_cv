# Usage:

1. Scrape videos from `sharepoint`
    1. Get sharepoint cookies using the `Get cookies.txt` browser extension and save to `cookies.txt`.
    2. Run `sharepoint/scrape.py` to scrape the directories and output a `urls.txt` file. 
    3. Run `sharepoint/download.py` to download the videos. Videos will be saved to `data/raw/`.
2. Extract and process frames using `extract_frames.py`. Frames will be saved to `data/processed/`.
3. Upload the frames to CVAT with `cvat/upload.py`.
4. After annotating in CVAT, export the annotations with `cvat/export.py`. The exported annotations will be saved to `data/datasets/`.
