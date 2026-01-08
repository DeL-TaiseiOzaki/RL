#!/usr/bin/env python3
"""Download HuggingFace dataset and convert to JSONL format for NeMo RL SFT."""

import json
import os
from datasets import load_dataset

def main():
    # Dataset info
    dataset_name = "ft-llm-team-mkj/stackmath-translation-jaen-llmjp"
    output_dir = "data/stackmath"

    print(f"Downloading dataset: {dataset_name}")

    # Load dataset from HuggingFace
    dataset = load_dataset(dataset_name)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save train split
    train_path = os.path.join(output_dir, "train.jsonl")
    print(f"Saving train split to {train_path}")
    with open(train_path, "w", encoding="utf-8") as f:
        for item in dataset["train"]:
            # Keep only 'messages' key for SFT
            output = {"messages": item["messages"]}
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
    print(f"Train: {len(dataset['train'])} samples")

    # Save test split (used as validation)
    test_path = os.path.join(output_dir, "test.jsonl")
    print(f"Saving test split to {test_path}")
    with open(test_path, "w", encoding="utf-8") as f:
        for item in dataset["test"]:
            output = {"messages": item["messages"]}
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
    print(f"Test: {len(dataset['test'])} samples")

    print(f"\nDone! Files saved to {output_dir}/")
    print(f"  - {train_path}")
    print(f"  - {test_path}")

if __name__ == "__main__":
    main()
