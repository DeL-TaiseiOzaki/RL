"""Prepare datasets for SFT and GRPO training.

Downloads from HuggingFace and converts to NeMo RL JSONL format.
Thinking OFF — assistant outputs structured data only (no CoT).

Strategy:
  1. Load all 3 sources and normalize to unified (user, assistant) pairs
  2. Pool all samples together
  3. Split 70% SFT / 30% GRPO
  4. SFT: system + user + assistant
  5. GRPO: system + user (prompt only) + ground_truth in extra_env_info

Output:
  data/sft_train.jsonl                    # SFT train (~9k)
  data/sft_val.jsonl                      # SFT val (~500)
  data/grpo_train.jsonl                   # GRPO train (~3.6k)
  data/grpo_val.jsonl                     # GRPO val (~400)
  data/reference/dpo_rejected_analysis.jsonl
"""

import json
import random
import re
from pathlib import Path

from datasets import load_dataset

SEED = 42
random.seed(SEED)

DATA_DIR = Path(__file__).parent
REF_DIR = DATA_DIR / "reference"

SYSTEM_PROMPT = (
    "You are a structured data generation assistant. "
    "Output ONLY the requested format (JSON, YAML, TOML, XML, or CSV). "
    "Do not include explanations, markdown code blocks, or any additional text. "
    "The output must be syntactically valid and directly parseable."
)

SFT_RATIO = 0.70  # 70% SFT, 30% GRPO


def save_jsonl(data: list[dict], path: Path) -> None:
    """Save list of dicts as JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} samples to {path}")


def extract_output_from_cot(ground_truth: str) -> str:
    """Extract the Output: section from CoT ground truth.

    Format: "Approach:\n1. ...\n...\n\nOutput:\n{actual data}"
    Returns the actual structured data after "Output:\n".
    """
    match = re.search(r"\nOutput:\n", ground_truth)
    if match:
        return ground_truth[match.end():]
    return ground_truth


def load_all_samples() -> list[dict]:
    """Load all datasets and normalize to unified (user_content, assistant_content, metadata)."""
    all_samples = []

    # ── Source 1: structured-5k-mix-sft (5k) ──
    print("\n=== Loading daichira/structured-5k-mix-sft ===")
    ds_main = load_dataset("daichira/structured-5k-mix-sft", split="train")
    count = 0
    for row in ds_main:
        messages = row["messages"]
        user_msgs = [m for m in messages if m["role"] == "user"]
        asst_msgs = [m for m in messages if m["role"] == "assistant"]
        if user_msgs and asst_msgs:
            all_samples.append({
                "user_content": user_msgs[-1]["content"],
                "assistant_content": asst_msgs[-1]["content"],
                "source": "structured-5k-mix-sft",
            })
            count += 1
    print(f"  Loaded: {count} samples")

    # ── Source 2: structured-hard-sft-4k (4k) ──
    print("\n=== Loading daichira/structured-hard-sft-4k ===")
    ds_hard = load_dataset("daichira/structured-hard-sft-4k", split="train")
    count = 0
    for row in ds_hard:
        messages = row["messages"]
        user_msgs = [m for m in messages if m["role"] == "user"]
        asst_msgs = [m for m in messages if m["role"] == "assistant"]
        if user_msgs and asst_msgs:
            all_samples.append({
                "user_content": user_msgs[-1]["content"],
                "assistant_content": asst_msgs[-1]["content"],
                "source": "structured-hard-sft-4k",
                "category": row.get("category", ""),
                "task": row.get("task", ""),
            })
            count += 1
    print(f"  Loaded: {count} samples")

    # ── Source 3: structured_data_with_cot_dataset_512_v5 (~4.5k) ──
    print("\n=== Loading u-10bei/structured_data_with_cot_dataset_512_v5 ===")
    ds_cot = load_dataset(
        "u-10bei/structured_data_with_cot_dataset_512_v5", split="train"
    )
    count = 0
    skipped = 0
    for row in ds_cot:
        messages = row["messages"]
        metadata = row["metadata"]
        user_msgs = [m for m in messages if m["role"] == "user"]
        asst_msgs = [m for m in messages if m["role"] == "assistant"]
        if not user_msgs or not asst_msgs:
            skipped += 1
            continue

        raw_gt = asst_msgs[-1]["content"]
        clean_output = extract_output_from_cot(raw_gt)
        if not clean_output.strip():
            skipped += 1
            continue

        all_samples.append({
            "user_content": user_msgs[-1]["content"],
            "assistant_content": clean_output,
            "source": "structured_data_with_cot_v5",
            "format": metadata.get("format", ""),
            "complexity": metadata.get("complexity", ""),
            "constraint": metadata.get("constraint"),
            "type": metadata.get("type", ""),
        })
        count += 1
    print(f"  Loaded: {count} samples (skipped {skipped})")

    print(f"\n  Total pool: {len(all_samples)} samples")
    return all_samples


def prepare_data() -> None:
    """Load all data, shuffle, and split 70/30 into SFT/GRPO."""
    all_samples = load_all_samples()

    # Shuffle and split
    random.shuffle(all_samples)
    sft_boundary = int(len(all_samples) * SFT_RATIO)
    sft_pool = all_samples[:sft_boundary]
    grpo_pool = all_samples[sft_boundary:]

    print(f"\n=== Split: SFT={len(sft_pool)}, GRPO={len(grpo_pool)} ===")

    # ── Format SFT samples: system + user + assistant ──
    sft_samples = []
    for s in sft_pool:
        sft_samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": s["user_content"]},
                {"role": "assistant", "content": s["assistant_content"]},
            ],
        })

    # ── Format GRPO samples: system + user (prompt only) + ground_truth ──
    grpo_samples = []
    for s in grpo_pool:
        extra = {"ground_truth": s["assistant_content"], "source": s["source"]}
        # Carry over source-specific metadata
        for key in ("category", "task", "format", "complexity", "constraint", "type"):
            if key in s:
                extra[key] = s[key]

        grpo_samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": s["user_content"]},
            ],
            "extra_env_info": extra,
            "task_name": "structured",
        })

    # ── Shuffle and split train/val ──
    random.shuffle(sft_samples)
    random.shuffle(grpo_samples)

    sft_val_size = max(1, int(len(sft_samples) * 0.05))
    grpo_val_size = max(1, int(len(grpo_samples) * 0.10))

    save_jsonl(sft_samples[sft_val_size:], DATA_DIR / "sft_train.jsonl")
    save_jsonl(sft_samples[:sft_val_size], DATA_DIR / "sft_val.jsonl")
    save_jsonl(grpo_samples[grpo_val_size:], DATA_DIR / "grpo_train.jsonl")
    save_jsonl(grpo_samples[:grpo_val_size], DATA_DIR / "grpo_val.jsonl")


def prepare_reference_data() -> None:
    """Save DPO dataset for reward function design reference."""
    print("\n=== Preparing Reference Data (DPO) ===")

    print("Loading u-10bei/dpo-dataset-qwen-cot ...")
    ds_dpo = load_dataset("u-10bei/dpo-dataset-qwen-cot", split="train")

    ref_samples = []
    for row in ds_dpo:
        prompt_text = row["prompt"]

        user_content = ""
        system_content = ""
        parts = prompt_text.split("<|im_start|>")
        for part in parts:
            part = part.strip()
            if part.startswith("user"):
                user_content = (
                    part.replace("user\n", "", 1).replace("<|im_end|>", "").strip()
                )
            elif part.startswith("system"):
                system_content = (
                    part.replace("system\n", "", 1).replace("<|im_end|>", "").strip()
                )

        ref_samples.append({
            "system": system_content,
            "user_prompt": user_content,
            "chosen": row["chosen"],
            "rejected": row["rejected"],
            "strategy": row["strategy"],
        })

    save_jsonl(ref_samples, REF_DIR / "dpo_rejected_analysis.jsonl")


def print_summary() -> None:
    """Print summary of all prepared data."""
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)

    for f in sorted(DATA_DIR.glob("*.jsonl")):
        line_count = sum(1 for _ in open(f))
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {line_count} samples ({size_mb:.1f} MB)")

    if REF_DIR.exists():
        print(f"\n{REF_DIR.relative_to(DATA_DIR)}/")
        for f in sorted(REF_DIR.glob("*.jsonl")):
            line_count = sum(1 for _ in open(f))
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {line_count} samples ({size_mb:.1f} MB)")


def main() -> None:
    print("Starting data preparation...")
    print(f"Output directory: {DATA_DIR}")
    print(f"System prompt: {SYSTEM_PROMPT[:60]}...")
    print(f"SFT/GRPO ratio: {SFT_RATIO:.0%} / {1 - SFT_RATIO:.0%}")

    prepare_data()
    prepare_reference_data()
    print_summary()


if __name__ == "__main__":
    main()
