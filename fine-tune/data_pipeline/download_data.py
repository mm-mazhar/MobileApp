# -*- coding: utf-8 -*-
# """
# download_data.py
# Created on Aug 21, 2025
# @ Author: Mazhar
# """

import os
import zipfile

import requests
import yaml
from tqdm import tqdm

CONFIGS_FILE = "./fine-tune/configs/configs.yaml"


def download_and_unzip(url, download_dir, extract_dir, description):
    """
    Downloads a file from a URL, shows a progress bar, saves it,
    extracts its contents, and then cleans up the downloaded zip file.
    (This function remains unchanged)
    """
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)

    filename = url.split("/")[-1]
    zip_path = os.path.join(download_dir, filename)

    print(f"Downloading {description}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024

        with tqdm(
            total=total_size_in_bytes, unit="iB", unit_scale=True, desc=description
        ) as progress_bar:
            with open(zip_path, "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong during download.")
            return
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return

    print(f"\nUnzipping {filename}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for member in tqdm(zip_ref.infolist(), desc="Extracting "):
                zip_ref.extract(member, extract_dir)
        print(f"Successfully extracted to: {extract_dir}")
    except zipfile.BadZipFile:
        print(
            f"Error: The downloaded file '{filename}' is not a valid zip file or is corrupted."
        )
    finally:
        print(f"Cleaning up {zip_path}...")
        os.remove(zip_path)
        print("-" * 30)


def main():
    """
    Main function to download and set up the VizWiz dataset by reading from config.yaml.
    """
    # --- Load Configuration from YAML file ---
    try:
        with open(CONFIGS_FILE, "r") as f:
            config = yaml.safe_load(f)

        # Get URLs and Paths from the loaded config
        urls = config["dataset"]["urls"]
        paths = config["dataset"]["paths"]

    except FileNotFoundError:
        print("Error: config.yaml not found. Please make sure the file exists.")
        return
    except KeyError as e:
        print(f"Error: Missing key in config.yaml: {e}")
        return

    # --- Define specific paths using the loaded config ---
    temp_zip_dir = paths["temp_zip_dir"]
    annotations_dir = paths["annotations_dir"]
    images_dir = paths["images_dir"]

    train_images_path = os.path.join(images_dir, "train")
    val_images_path = os.path.join(images_dir, "val")
    test_images_path = os.path.join(images_dir, "test")

    print("--- Starting VizWiz Dataset Download (using config.yaml) ---")

    # --- Execute the Pipeline ---
    if not os.path.exists(os.path.join(annotations_dir, "train.json")):
        download_and_unzip(
            urls["annotations"], temp_zip_dir, annotations_dir, "Annotations"
        )
    else:
        print("Annotations already exist. Skipping.")

    if not os.path.exists(train_images_path):
        download_and_unzip(
            urls["train_images"], temp_zip_dir, train_images_path, "Train Images"
        )
    else:
        print("Train images already exist. Skipping.")

    if not os.path.exists(val_images_path):
        download_and_unzip(
            urls["val_images"], temp_zip_dir, val_images_path, "Validation Images"
        )
    else:
        print("Validation images already exist. Skipping.")

    if not os.path.exists(test_images_path):
        download_and_unzip(
            urls["test_images"], temp_zip_dir, test_images_path, "Test Images"
        )
    else:
        print("Test images already exist. Skipping.")

    print(f"\n✅ All necessary files have been downloaded and organized.")
    print(f"Your complete dataset is ready at: {os.path.abspath(paths['base_dir'])}")


if __name__ == "__main__":
    main()
