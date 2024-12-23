import csv
import os
import re
import base64
from process_image2 import process_image2
from constants import CSV_FILENAME, FIELDNAMES
from get_image_names import get_image_names
from PIL import Image


def read_image_as_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_data = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_data}"


def main():
    # Path to the images directory
    images_dir = os.path.expanduser("statista_dataset/dataset/imgs")

    # Get list of all image files in the directory
    image_files = [
        f for f in os.listdir(images_dir) if f.lower().endswith(
            ('.png', '.jpg', '.jpeg', '.gif'))]

    # Check if the CSV file exists
    file_exists = os.path.isfile(CSV_FILENAME)

    # Initialize CSV with headers if it doesn't exist
    if not file_exists:
        with open(CSV_FILENAME, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
            writer.writeheader()
        processed_count = 0
    else:
        with open(CSV_FILENAME, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            processed_count = sum(1 for row in reader) - 1  # Subtract header row

    image_names = get_image_names()

    # Process images starting from where we left off
    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        image = Image.open(image_path)
        image_id = int(re.search(r'(\d+)', image_file).group(1))
        if image_id in image_names:
            process_image2(image, image_id)


if __name__ == "__main__":
    main()
