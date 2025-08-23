# -*- coding: utf-8 -*-
# """
# fine-data_viewer.py
# Description:
# Utility function to view a single instance from a processed dataset.
# Created on Aug 21, 2025
# @ Author: Mazhar
# """

import json
import os

import yaml
from datasets import load_from_disk
from PIL import Image
from IPython.display import display

try:
    from rich.console import Console
    from rich.syntax import Syntax

    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    import pprint

    RICH_AVAILABLE = False


def view_dataset_instance(config_path, split: str, index):
    """
    Loads a dataset from disk and displays a specific instance.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        processed_dir = config["dataset"]["paths"]["processed_dir"]
    if not os.path.exists(processed_dir):
        print(f"❌ ERROR: Processed dataset directory not found at: {processed_dir}")
        return

    print(f"🔄 Loading dataset from '{processed_dir}'...")
    dataset = load_from_disk(processed_dir)

    if split not in dataset:
        print(f"❌ ERROR: Split '{split}' not found. Available: {list(dataset.keys())}")
        return

    dataset_split = dataset[split]
    if not (0 <= index < len(dataset_split)):
        print(
            f"❌ ERROR: Index {index} is out of bounds for split '{split}' (size: {len(dataset_split)})."
        )
        return

    instance = dataset_split[index]
    image = instance["image"]
    messages = instance["messages"]

    print(
        "\n"
        + "=" * 50
        + f"\n    Displaying instance [{index}] from split '{split}'\n"
        + "=" * 50
        + "\n"
    )
    print("🖼️  Image:")
    print(f"    - Mode: {image.mode}, Size: {image.size}")
    print("    - Displaying image...")
    # image.show()
    display(image)

    print("\n💬 Messages:")

    if RICH_AVAILABLE:
        messages_str = json.dumps(messages, indent=2)
        console.print(Syntax(messages_str, "json", theme="monokai", line_numbers=True))
    else:
        pprint.pprint(messages)
    print("\n" + "=" * 50)
