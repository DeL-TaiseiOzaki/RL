#!/usr/bin/env python3
"""Check the exact format of Output: section in CoT v5 ground truths."""

import json
from pathlib import Path


def extract_output_section(ground_truth: str) -> str:
    """Extract the Output: section from ground truth."""
    if 'Output:' not in ground_truth:
        return None

    # Split by Output: and get the part after it
    parts = ground_truth.split('Output:', 1)
    if len(parts) < 2:
        return None

    output_section = parts[1]
    # Return the output section, stripped of leading/trailing whitespace
    return output_section.strip()


def main():
    """Main function."""
    file_path = Path('/home/usr_ext_taisei_ozaki_ccoe_toyota/RL/data/grpo/train.jsonl')

    print("=" * 80)
    print("CoT v5 Output Format Analysis")
    print("=" * 80)

    sample_count = 0
    max_samples = 5

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)
            env_info = data.get('extra_env_info', {})
            source = env_info.get('source', '')

            if 'cot_v5' in source and sample_count < max_samples:
                sample_count += 1

                # Get metadata
                fmt = env_info.get('format', 'unknown')
                complexity = env_info.get('complexity', 'unknown')
                type_ = env_info.get('type', 'unknown')

                # Get user prompt
                messages = data.get('messages', [])
                user_msg = ''
                for msg in messages:
                    if msg.get('role') == 'user':
                        user_msg = msg.get('content', '')
                        break

                ground_truth = env_info.get('ground_truth', '')
                output_section = extract_output_section(ground_truth)

                print(f"\n{'=' * 80}")
                print(f"Sample {sample_count}")
                print(f"{'=' * 80}")
                print(f"Format: {fmt}, Complexity: {complexity}, Type: {type_}")
                print(f"\nUser Prompt (first 150 chars):")
                print(f"  {user_msg[:150]}...")
                print(f"\nFull Ground Truth (first 400 chars):")
                print(f"  {ground_truth[:400]}...")
                print(f"\nExtracted Output Section (first 300 chars):")
                if output_section:
                    print(f"  {output_section[:300]}")
                    if len(output_section) > 300:
                        print("  ...")
                else:
                    print("  [No output section found]")

            if sample_count >= max_samples:
                break

    print("\n" + "=" * 80)
    print(f"Analyzed {sample_count} CoT v5 samples")
    print("=" * 80)


if __name__ == '__main__':
    main()
