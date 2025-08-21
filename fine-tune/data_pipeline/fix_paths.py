# -*- coding: utf-8 -*-
# """
# fix_paths.py
# Created on Aug 21, 2025
# @ Author: Mazhar
# """

import os
import shutil

import yaml

CONFIGS_FILE = "./fine-tune/configs/configs.yaml"


def flatten_nested_directory(parent_dir):
    """
    Checks for and fixes the common 'parent/parent/content' nested structure.
    For example, moves content from 'images/train/train' to 'images/train'.
    """
    for root, dirs, _ in os.walk(parent_dir):
        # Check if a directory contains only one subdirectory with the same name
        if len(dirs) == 1 and os.path.basename(root) == dirs[0]:
            nested_dir = os.path.join(root, dirs[0])
            print(f"Fixing nested directory: {nested_dir}")

            # Move all items from the nested directory to the parent
            for item_name in os.listdir(nested_dir):
                source_path = os.path.join(nested_dir, item_name)
                destination_path = os.path.join(root, item_name)
                shutil.move(source_path, destination_path)

            # Remove the now-empty nested directory
            os.rmdir(nested_dir)
            print(f"Successfully moved contents to: {root}")


def main():
    # --- Load Configuration from YAML file ---
    try:
        with open(CONFIGS_FILE, "r") as f:
            config = yaml.safe_load(f)

        # Get base dir path from the loaded config
        base_dir = config["dataset"]["paths"]["base_dir"]
        print(f"Base directory: {base_dir}")

    except FileNotFoundError:
        print("Error: config.yaml not found. Please make sure the file exists.")
        return
    except KeyError as e:
        print(f"Error: Missing key in config.yaml: {e}")
        return

    images_dir = os.path.join(base_dir, "images")
    annotations_dir = os.path.join(base_dir, "annotations")

    print("--- Starting Directory Cleanup ---")

    # Run the flattening function on the key directories
    flatten_nested_directory(images_dir)
    flatten_nested_directory(annotations_dir)

    print("\n--- Directory Cleanup Finished ---")
    print("Your file paths are now clean.")


if __name__ == "__main__":
    main()
