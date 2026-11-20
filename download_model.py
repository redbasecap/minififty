"""Download the mind's model (SmolLM2-135M-Instruct, Apache-2.0) into ./model.

The server and the Docker image load it from there, so nothing is fetched at
run time.
"""
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
OUT_DIR = Path(__file__).parent / "model"
FILES = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
         "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]

if __name__ == "__main__":
    snapshot_download(MODEL_NAME, local_dir=OUT_DIR, allow_patterns=FILES, token=False)
    print(f"Saved {MODEL_NAME} to {OUT_DIR}")
