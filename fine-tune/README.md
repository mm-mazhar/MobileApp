### Fine Tune Gemma 3n

- uv sync
- in cmd run `huggingface-cli login` and paste the Token with Write Permissions
- [Optional] set the paths if needed in `./fine-tune/configs/configs.yaml` file
- uv run python `./fine-tune/prepare_data.py`
- uv run python `./fine-tune/train.py`
