# -*- coding: utf-8 -*-
# """
# prepare_dataset.py
# Created on Aug 21, 2025
# @ Author: Mazhar
# """


import json
import os

import yaml
from tqdm import tqdm

CONFIGS_FILE = "./fine-tune/configs/configs.yaml"


def process_vizwiz_annotations(annotation_path, images_base_dir, output_path):
    """
    Processes a VizWiz COCO-style JSON annotation file with captions (train/val)
    and converts it into a JSON Lines (.jsonl) format suitable for fine-tuning.
    """
    print(f"Loading annotation file: {annotation_path}")
    with open(annotation_path, "r") as f:
        data = json.load(f)

    image_id_to_path = {
        image["id"]: os.path.join(images_base_dir, image["file_name"])
        for image in data["images"]
    }

    processed_records = []
    print("Processing train/val annotations...")

    for annotation in tqdm(data["annotations"]):
        if annotation.get("is_rejected", False):
            continue

        image_id = annotation["image_id"]
        caption = annotation["caption"]

        if image_id in image_id_to_path:
            image_path = image_id_to_path[image_id]
            record = {"image_path": image_path, "caption": caption}
            processed_records.append(record)

    print(f"Saving {len(processed_records)} records to: {output_path}")
    with open(output_path, "w") as f:
        for record in processed_records:
            f.write(json.dumps(record) + "\n")

    print("Processing complete.")


def process_vizwiz_test_images(annotation_path, images_base_dir, output_path):
    """
    Processes a "blind" VizWiz test JSON file (which has no captions)
    and converts it into a clean JSON Lines (.jsonl) format.
    """
    print(f"Loading test annotation file: {annotation_path}")
    with open(annotation_path, "r") as f:
        data = json.load(f)

    processed_records = []
    print("Processing test images...")

    for image_info in tqdm(data["images"]):
        image_path = os.path.join(images_base_dir, image_info["file_name"])
        record = {"image_path": image_path}
        processed_records.append(record)

    print(f"Saving {len(processed_records)} test image records to: {output_path}")
    with open(output_path, "w") as f:
        for record in processed_records:
            f.write(json.dumps(record) + "\n")

    print("Processing complete.")


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

    annotations_dir = os.path.join(base_dir, "annotations")
    images_dir = os.path.join(base_dir, "images")

    train_json_path = os.path.join(annotations_dir, "train.json")
    val_json_path = os.path.join(annotations_dir, "val.json")
    test_json_path = os.path.join(annotations_dir, "test.json")

    # The output .jsonl files will now be saved inside the 'annotations' directory.
    train_output_path = os.path.join(annotations_dir, "train.jsonl")
    val_output_path = os.path.join(annotations_dir, "val.jsonl")
    test_output_path = os.path.join(annotations_dir, "test.jsonl")
    # --- END OF MODIFIED SECTION ---

    print("--- Starting Dataset Preparation ---")

    # Process the training data
    process_vizwiz_annotations(
        train_json_path, os.path.join(images_dir, "train"), train_output_path
    )
    print("-" * 30)

    # Process the validation data
    process_vizwiz_annotations(
        val_json_path, os.path.join(images_dir, "val"), val_output_path
    )
    print("-" * 30)

    # Process the test data
    process_vizwiz_test_images(
        test_json_path, os.path.join(images_dir, "test"), test_output_path
    )

    print("\n--- Dataset Preparation Finished ---")
    print(
        f"Your processed .jsonl files are now located in: {os.path.abspath(annotations_dir)}"
    )


if __name__ == "__main__":
    main()
