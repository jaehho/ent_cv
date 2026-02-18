import os

def generate_sharepoint_urls():
    # Base URL structure
    base_path = "https://mtsinai-my.sharepoint.com/:v:/r/personal/turner_baker_mssm_edu/Documents/Pharyvac-Related%20OR%20Exposure%20Case%20Videos/OR%20GoPro%20Videos/OR%20Gopro%20Videos%20for%20Jae%20+%20Abdel"
    suffix = "?csf=1&download=1"

    # Configuration: "folder_name": number_of_parts
    # You can easily add or remove items from this list
    collections = {
        "20251113_02": 4,
        "20251124_01": 10,
        "20251204_01": 8,
        "20251208_01": 8,
        "20251208_02": 7,
        "20251210_01": 3,
        "20251210_02": 4,
        "20251210_03": 7,
        "20251217_01": 1,
        "20251217_02": 5,
        "20251217_03": 4,
        "20251218_01": 4,
        "20251218_02": 7,
        "20260108_01": 3,
    }

    url_list = []

    for folder, part_count in collections.items():
        for part_num in range(1, part_count + 1):
            # Construct the specific URL
            # Note: Folder and File Prefix are the same in your example
            url = f"{base_path}/{folder}/{folder}_Part{part_num}.mp4{suffix}"
            url_list.append(url)

    # Save to file
    with open("urls.txt", "w") as f:
        for line in url_list:
            f.write(line + "\n")

    print(f"Successfully generated urls.txt with {len(url_list)} entries.")

if __name__ == "__main__":
    generate_sharepoint_urls()