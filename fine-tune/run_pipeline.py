# -*- coding: utf-8 -*-
# """
# download_data.py
# Description:
# run_pipeline.py
# This master script, located in the project root, executes the entire
# data preparation pipeline from the 'data_pipeline' sub-folder.
# Created on Aug 21, 2025
# @ Author: Mazhar
# """

import sys
import time

# --- Step 1: Update imports to use the 'data_pipeline' package ---
# This is the main change. We use "from package.module import function".
try:
    from data_pipeline.download_data import main as download_main
    from data_pipeline.fix_paths import main as fix_paths_main
    from data_pipeline.prepare_dataset import main as prepare_dataset_main
except ModuleNotFoundError:
    print("❌ ERROR: Could not find the 'data_pipeline' package.")
    print(
        "Please ensure this script is in the parent directory of 'data_pipeline' and that the folder contains an '__init__.py' file."
    )
    sys.exit(1)


def main():
    """
    Orchestrates the entire data preparation pipeline.
    """
    print("🚀 =============================================== 🚀")
    print("    Starting the Full VizWiz Data Pipeline")
    print("🚀 =============================================== 🚀")
    start_time = time.time()

    # The rest of the script remains exactly the same.
    # It will call the imported functions as before.
    print("\n[STEP 1/3] Running the download script...")
    try:
        download_main()
        print("[STEP 1/3] Download script finished successfully.")
    except Exception as e:
        print(f"❌ ERROR in download_data.py: {e}")
        print("Pipeline stopped.")
        return

    print("\n[STEP 2/3] Running the path fixing script...")
    try:
        fix_paths_main()
        print("[STEP 2/3] Path fixing script finished successfully.")
    except Exception as e:
        print(f"❌ ERROR in fix_paths.py: {e}")
        print("Pipeline stopped.")
        return

    print("\n[STEP 3/3] Running the dataset preparation script...")
    try:
        prepare_dataset_main()
        print("[STEP 3/3] Dataset preparation finished successfully.")
    except Exception as e:
        print(f"❌ ERROR in prepare_dataset.py: {e}")
        print("Pipeline stopped.")
        return

    end_time = time.time()
    total_time = end_time - start_time

    print("\n✅ =============================================== ✅")
    print("    Data Pipeline Completed Successfully!")
    print(f"    Total execution time: {total_time:.2f} seconds")
    print("✅ =============================================== ✅")


if __name__ == "__main__":
    main()
