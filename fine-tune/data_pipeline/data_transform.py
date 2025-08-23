# -*- coding: utf-8 -*-
# """
# data_transform.py
# Description:
# Contains functions for transforming the prepared data into a model-ready format.
# Created on Aug 21, 2025
# @ Author: Mazhar
# """

import os

import yaml
from datasets import load_dataset
from PIL import Image
from datasets import load_from_disk


def create_conversation(example):
    # ... PASTE THE create_conversation FUNCTION CODE HERE ...
    system_prompt = (
        "You are a helpful assistant for a visually impaired person. "
        "Your task is to describe the scene in the provided image clearly and "
        "concisely, focusing on potential obstacles or key objects."
    )
    user_content = [
        {"type": "image"},
        {"type": "text", "text": "Please describe what you see."},
    ]
    assistant_content = [{"type": "text", "text": example["caption"]}]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    image = Image.open(example["image_path"])
    return {"image": image, "messages": messages}


def transform_data_for_tuning(config_path):
    """
    Loads the prepared .jsonl data, applies the conversational transformation,
    and saves the final dataset to disk, ready for training.
    """
    print("--- Starting Dataset Transformation ---")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    paths = config["dataset"]["paths"]
    annotations_dir = paths["annotations_dir"]
    processed_dir = paths["processed_dir"]

    print(f"Loading .jsonl files from: {annotations_dir}")
    data_files = {
        "train": os.path.join(annotations_dir, "train.jsonl"),
        "val": os.path.join(annotations_dir, "val.jsonl"),
    }
    dataset = load_dataset("json", data_files=data_files)

    print("\nApplying conversational transformation...")
    transformed_dataset = dataset.map(
        create_conversation, remove_columns=["image_path", "caption"], batched=False
    )

    print(f"\nSaving transformed dataset to: {processed_dir}")
    os.makedirs(processed_dir, exist_ok=True)
    transformed_dataset.save_to_disk(processed_dir)

    print(
        f"✅ Your model-ready dataset is now saved at: {os.path.abspath(processed_dir)}"
    )


def verify_dataset_integrity(config_path: str):
    """
    Verifies that the number of examples in the processed Arrow dataset
    matches the number of lines in the original .jsonl files.
    """
    print("--- Starting Dataset Integrity Verification ---")

    # --- Load Configuration ---
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    paths = config["dataset"]["paths"]
    annotations_dir = paths["annotations_dir"]
    processed_dir = paths["processed_dir"]

    all_splits_valid = True

    # --- Verify each split (train, validation) ---
    for split in ["train", "val"]:
        print(f"\nVerifying '{split}' split...")

        jsonl_path = os.path.join(annotations_dir, f"{split}.jsonl")

        if not os.path.exists(jsonl_path):
            print(f"⚠️  WARNING: Source file '{jsonl_path}' not found. Cannot verify.")
            all_splits_valid = False
            continue

        # Count lines in the original .jsonl file
        with open(jsonl_path, "r", encoding="utf-8") as f:
            original_line_count = sum(1 for _ in f)

        # Load the processed Arrow dataset from disk
        processed_dataset = load_from_disk(processed_dir)

        if split not in processed_dataset:
            print(f"❌ ERROR: Split '{split}' not found in the processed dataset.")
            all_splits_valid = False
            continue

        processed_example_count = len(processed_dataset[split])

        # --- Compare the counts ---
        print(f"  - Lines in original '{split}.jsonl': {original_line_count}")
        print(f"  - Examples in processed '{split}' set: {processed_example_count}")

        if original_line_count == processed_example_count:
            print("  ✅ Verification successful for this split.")
        else:
            print("  ❌ VERIFICATION FAILED. The number of examples does not match.")
            all_splits_valid = False

    return all_splits_valid
