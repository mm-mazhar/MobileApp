# -*- coding: utf-8 -*-
# """
# prepare_data.py
# Description:
# A simple, controllable script to run the data pipeline step-by-step.
# Edit the "CONTROL PANEL" below to choose which steps to execute.
# Created on Aug 21, 2025
# @ Author: Mazhar
# """

import sys
import time

# --- CONTROL PANEL ---
# Pipeline Steps (Set to True to run, False to skip)
RUN_PREPARE_STEP = False  # Downloads and prepares the raw VizWiz data.
RUN_TRANSFORM_STEP = True  # Transforms the data into the conversational format.
CHECK_DATASET_INTEGRITY = True  # Verifies that no data was lost during transformation.
# --- END OF CONTROL PANEL ---

# Import our modular functions and classes
try:
    from data_pipeline import (
        VizWizDataPipeline,
        transform_data_for_tuning,
        verify_dataset_integrity,
    )
except ModuleNotFoundError:
    print("❌ ERROR: Could not find the 'sightguide' package.")
    print(
        "Please ensure this script is in the parent directory of 'sightguide' and that the folder contains an '__init__.py' file."
    )
    sys.exit(1)

# Define the path to the configuration file
CONFIG_FILE_PATH = "fine-tune/configs/configs.yaml"


def run_step(step_name, function_to_run, *args, **kwargs):
    """A wrapper function to run a pipeline step with consistent logging and error handling."""
    print(f"\n--- Running Step: {step_name} ---")
    try:
        function_to_run(*args, **kwargs)
        print(f"\n✅ Step '{step_name}' finished successfully.")
        return True  # Indicate success
    except Exception as e:
        print(f"❌ ERROR in Step '{step_name}': {e}")
        print("Pipeline stopped.")
        return False  # Indicate failure


def main():
    """Orchestrates the pipeline based on the flags set in the CONTROL PANEL."""
    start_time = time.time()

    # --- Run the Data Processing Pipeline ---
    print("🚀 =============================================== 🚀")
    print("         Fine-Tunning | Data Pipeline Runner")
    print("🚀 =============================================== 🚀")

    # Define the functions for each step
    def prepare_step():
        pipeline = VizWizDataPipeline(config_path=CONFIG_FILE_PATH)
        pipeline.run()

    def transform_step():
        transform_data_for_tuning(config_path=CONFIG_FILE_PATH)

    # Execute the pipeline steps sequentially
    if RUN_PREPARE_STEP:
        if not run_step("Prepare Raw Dataset", prepare_step):
            return  # Stop if the step fails

    if RUN_TRANSFORM_STEP:
        if not run_step("Transform Dataset for Tuning", transform_step):
            return  # Stop if the step fails

    if CHECK_DATASET_INTEGRITY:

        def verify_step():
            return verify_dataset_integrity(config_path=CONFIG_FILE_PATH)

        if not run_step("Verify Dataset Integrity", verify_step):
            return

    end_time = time.time()
    total_time = end_time - start_time

    print("\n✅ =============================================== ✅")
    print("    Data Pipeline run has finished!")
    print(f"    Total execution time: {total_time:.2f} seconds")
    print("✅ =============================================== ✅")


if __name__ == "__main__":
    main()
