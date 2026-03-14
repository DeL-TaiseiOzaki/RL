"""Prepare datasets for math SFT and GRPO training.

Downloads from HuggingFace and converts to NeMo RL JSONL format.

Sources:
  SFT:  ft-llm-team-mkj/cot-gsm8k-stackmath-merge  (35k, chat format with <think> CoT)
  GRPO: ft-llm-team-mkj/math-grpo-200-en            (200, problem/solution flat format)

Output:
  data/math_sft_train.jsonl   # SFT train (~33.9k)
  data/math_sft_val.jsonl     # SFT val (~1.8k)
  data/math_grpo_train.jsonl  # GRPO train (~180)
  data/math_grpo_val.jsonl    # GRPO val (~20)
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

SEED = 42
random.seed(SEED)

DATA_DIR = Path(__file__).parent

SFT_VAL_RATIO = 0.05
GRPO_VAL_RATIO = 0.10


def save_jsonl(data: list[dict], path: Path) -> None:
    """Save list of dicts as JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} samples to {path}")


def prepare_sft_data() -> None:
    """Download and prepare SFT dataset (cot-gsm8k-stackmath-merge).

    The dataset already has 'messages' in OpenAI chat format:
      [system, user, assistant(with <think> CoT)]
    """
    print("\n=== Preparing SFT Data ===")
    print("Loading ft-llm-team-mkj/cot-gsm8k-stackmath-merge ...")

    ds = load_dataset("ft-llm-team-mkj/cot-gsm8k-stackmath-merge", split="train")
    print(f"  Total samples: {len(ds)}")

    samples = []
    for row in ds:
        samples.append({"messages": row["messages"]})

    random.shuffle(samples)

    val_size = max(1, int(len(samples) * SFT_VAL_RATIO))
    save_jsonl(samples[val_size:], DATA_DIR / "math_sft_train.jsonl")
    save_jsonl(samples[:val_size], DATA_DIR / "math_sft_val.jsonl")


def prepare_grpo_data() -> None:
    """Download and prepare GRPO dataset (math-grpo-200-en).

    The dataset has flat format: id, category, unit, problem, solution.
    Save with problem/solution keys — ResponseDataset will create messages
    via input_key="problem" and output_key="solution".
    """
    print("\n=== Preparing GRPO Data ===")
    print("Loading ft-llm-team-mkj/math-grpo-200-en ...")

    ds = load_dataset("ft-llm-team-mkj/math-grpo-200-en", split="train")
    print(f"  Total samples: {len(ds)}")

    samples = []
    for row in ds:
        samples.append({
            "problem": row["problem"],
            "solution": row["solution"],
        })

    random.shuffle(samples)

    val_size = max(1, int(len(samples) * GRPO_VAL_RATIO))
    save_jsonl(samples[val_size:], DATA_DIR / "math_grpo_train.jsonl")
    save_jsonl(samples[:val_size], DATA_DIR / "math_grpo_val.jsonl")


def print_summary() -> None:
    """Print summary of all prepared data."""
    print("\n" + "=" * 60)
    print("MATH DATA PREPARATION COMPLETE")
    print("=" * 60)

    for pattern in ["math_sft_*.jsonl", "math_grpo_*.jsonl"]:
        for f in sorted(DATA_DIR.glob(pattern)):
            line_count = sum(1 for _ in open(f))
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {line_count} samples ({size_mb:.1f} MB)")


def main() -> None:
    print("Starting math data preparation...")
    print(f"Output directory: {DATA_DIR}")

    prepare_sft_data()
    prepare_grpo_data()
    print_summary()


if __name__ == "__main__":
    main()
