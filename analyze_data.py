#!/usr/bin/env python3
"""Analyze training data files and provide detailed statistics."""

import json
from collections import Counter, defaultdict
from pathlib import Path
import re


def analyze_grpo_file(file_path: Path) -> dict:
    """Analyze GRPO data file."""
    total = 0
    source_counts = Counter()

    # For cot_v5 source
    cot_format_counts = Counter()
    cot_complexity_counts = Counter()
    cot_type_counts = Counter()
    cot_with_output = 0
    cot_without_output = 0
    cot_examples = []

    # For hard source
    hard_category_counts = Counter()
    hard_task_counts = Counter()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            data = json.loads(line)

            # Get metadata from extra_env_info
            env_info = data.get('extra_env_info', {})
            source = env_info.get('source', '')
            source_counts[source] += 1

            if 'cot_v5' in source:
                cot_format_counts[env_info.get('format', 'unknown')] += 1
                cot_complexity_counts[env_info.get('complexity', 'unknown')] += 1
                cot_type_counts[env_info.get('type', 'unknown')] += 1

                ground_truth = env_info.get('ground_truth', '')
                if 'Output:' in ground_truth:
                    cot_with_output += 1
                else:
                    cot_without_output += 1

                # Collect first 3 examples
                if len(cot_examples) < 3:
                    # Get user prompt from messages
                    messages = data.get('messages', [])
                    user_msg = ''
                    for msg in messages:
                        if msg.get('role') == 'user':
                            user_msg = msg.get('content', '')
                            break

                    cot_examples.append({
                        'prompt': user_msg[:100] + '...' if len(user_msg) > 100 else user_msg,
                        'ground_truth': ground_truth
                    })

            elif 'hard' in source:
                hard_category_counts[env_info.get('category', 'unknown')] += 1
                hard_task_counts[env_info.get('task', 'unknown')] += 1

    return {
        'total': total,
        'source_counts': dict(source_counts),
        'cot_v5': {
            'format_counts': dict(cot_format_counts),
            'complexity_counts': dict(cot_complexity_counts),
            'type_counts': dict(cot_type_counts),
            'with_output_prefix': cot_with_output,
            'without_output_prefix': cot_without_output,
            'examples': cot_examples
        },
        'hard': {
            'category_counts': dict(hard_category_counts),
            'task_counts': dict(hard_task_counts)
        }
    }


def analyze_sft_file(file_path: Path) -> dict:
    """Analyze SFT data file."""
    total = 0
    format_mentions = Counter()
    task_types = Counter()
    has_system = 0
    no_system = 0
    response_lengths = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            data = json.loads(line)

            messages = data.get('messages', [])

            # Check for system message
            if messages and messages[0].get('role') == 'system':
                has_system += 1
            else:
                no_system += 1

            # Analyze user prompts for format mentions
            user_content = ''
            assistant_content = ''
            for msg in messages:
                if msg.get('role') == 'user':
                    user_content += msg.get('content', '')
                elif msg.get('role') == 'assistant':
                    assistant_content += msg.get('content', '')

            # Count format mentions
            user_lower = user_content.lower()
            if 'json' in user_lower:
                format_mentions['json'] += 1
            if 'yaml' in user_lower:
                format_mentions['yaml'] += 1
            if 'toml' in user_lower:
                format_mentions['toml'] += 1
            if 'xml' in user_lower:
                format_mentions['xml'] += 1
            if 'csv' in user_lower:
                format_mentions['csv'] += 1

            # Categorize task type
            if 'extract' in user_lower:
                task_types['extraction'] += 1
            if 'convert' in user_lower or 'transform' in user_lower:
                task_types['conversion'] += 1
            if 'generate' in user_lower or 'create' in user_lower:
                task_types['generation'] += 1
            if 'parse' in user_lower:
                task_types['parsing'] += 1
            if 'validate' in user_lower:
                task_types['validation'] += 1

            # Response length
            if assistant_content:
                response_lengths.append(len(assistant_content))

    avg_response_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0

    return {
        'total': total,
        'format_mentions': dict(format_mentions),
        'task_types': dict(task_types),
        'has_system_message': has_system,
        'no_system_message': no_system,
        'avg_response_length': avg_response_length,
        'min_response_length': min(response_lengths) if response_lengths else 0,
        'max_response_length': max(response_lengths) if response_lengths else 0
    }


def main():
    """Main analysis function."""
    base_dir = Path('/home/usr_ext_taisei_ozaki_ccoe_toyota/RL/data')

    print("=" * 80)
    print("GRPO TRAINING DATA ANALYSIS")
    print("=" * 80)
    grpo_train_stats = analyze_grpo_file(base_dir / 'grpo' / 'train.jsonl')

    print(f"\nTotal samples: {grpo_train_stats['total']}")
    print(f"\nSource distribution:")
    for source, count in sorted(grpo_train_stats['source_counts'].items()):
        print(f"  {source}: {count} ({count/grpo_train_stats['total']*100:.1f}%)")

    print(f"\n--- CoT v5 Source Analysis ---")
    cot_stats = grpo_train_stats['cot_v5']
    print(f"Format distribution:")
    for fmt, count in sorted(cot_stats['format_counts'].items()):
        print(f"  {fmt}: {count}")

    print(f"\nComplexity distribution:")
    for complexity, count in sorted(cot_stats['complexity_counts'].items()):
        print(f"  {complexity}: {count}")

    print(f"\nType distribution:")
    for type_, count in sorted(cot_stats['type_counts'].items()):
        print(f"  {type_}: {count}")

    total_cot = cot_stats['with_output_prefix'] + cot_stats['without_output_prefix']
    if total_cot > 0:
        print(f"\nGround truth 'Output:' prefix analysis:")
        print(f"  With 'Output:': {cot_stats['with_output_prefix']} ({cot_stats['with_output_prefix']/total_cot*100:.1f}%)")
        print(f"  Without 'Output:': {cot_stats['without_output_prefix']} ({cot_stats['without_output_prefix']/total_cot*100:.1f}%)")
    else:
        print(f"\nNo CoT v5 samples found.")

    print(f"\nCoT v5 ground_truth examples (first 3):")
    for i, example in enumerate(cot_stats['examples'], 1):
        print(f"\nExample {i}:")
        print(f"  Prompt: {example['prompt']}")
        print(f"  Ground truth:")
        # Show first 500 chars to see the format
        gt = example['ground_truth']
        if len(gt) > 500:
            print(f"    {gt[:500]}...")
        else:
            print(f"    {gt}")

    print(f"\n--- Hard Source Analysis ---")
    hard_stats = grpo_train_stats['hard']
    print(f"Category distribution:")
    for category, count in sorted(hard_stats['category_counts'].items()):
        print(f"  {category}: {count}")

    print(f"\nTask distribution:")
    for task, count in sorted(hard_stats['task_counts'].items()):
        print(f"  {task}: {count}")

    print("\n" + "=" * 80)
    print("SFT TRAINING DATA ANALYSIS")
    print("=" * 80)
    sft_stats = analyze_sft_file(base_dir / 'sft' / 'train.jsonl')

    print(f"\nTotal samples: {sft_stats['total']}")

    print(f"\nFormat mentions in user prompts:")
    for fmt, count in sorted(sft_stats['format_mentions'].items()):
        print(f"  {fmt}: {count} ({count/sft_stats['total']*100:.1f}%)")

    print(f"\nTask type distribution:")
    for task, count in sorted(sft_stats['task_types'].items()):
        print(f"  {task}: {count}")

    print(f"\nSystem message presence:")
    print(f"  Has system message: {sft_stats['has_system_message']}")
    print(f"  No system message: {sft_stats['no_system_message']}")

    print(f"\nAssistant response length statistics:")
    print(f"  Average: {sft_stats['avg_response_length']:.0f} chars")
    print(f"  Min: {sft_stats['min_response_length']} chars")
    print(f"  Max: {sft_stats['max_response_length']} chars")

    print("\n" + "=" * 80)
    print("GRPO VALIDATION DATA ANALYSIS")
    print("=" * 80)
    grpo_val_stats = analyze_grpo_file(base_dir / 'grpo' / 'val.jsonl')

    print(f"\nTotal samples: {grpo_val_stats['total']}")
    print(f"\nSource distribution:")
    for source, count in sorted(grpo_val_stats['source_counts'].items()):
        print(f"  {source}: {count} ({count/grpo_val_stats['total']*100:.1f}%)")

    print(f"\n--- CoT v5 Source Analysis ---")
    val_cot_stats = grpo_val_stats['cot_v5']
    print(f"Format distribution:")
    for fmt, count in sorted(val_cot_stats['format_counts'].items()):
        print(f"  {fmt}: {count}")

    print(f"\nComplexity distribution:")
    for complexity, count in sorted(val_cot_stats['complexity_counts'].items()):
        print(f"  {complexity}: {count}")

    print(f"\nType distribution:")
    for type_, count in sorted(val_cot_stats['type_counts'].items()):
        print(f"  {type_}: {count}")

    total_val_cot = val_cot_stats['with_output_prefix'] + val_cot_stats['without_output_prefix']
    if total_val_cot > 0:
        print(f"\nGround truth 'Output:' prefix analysis:")
        print(f"  With 'Output:': {val_cot_stats['with_output_prefix']} ({val_cot_stats['with_output_prefix']/total_val_cot*100:.1f}%)")
        print(f"  Without 'Output:': {val_cot_stats['without_output_prefix']} ({val_cot_stats['without_output_prefix']/total_val_cot*100:.1f}%)")

    print(f"\n--- Hard Source Analysis ---")
    val_hard_stats = grpo_val_stats['hard']
    print(f"Category distribution:")
    for category, count in sorted(val_hard_stats['category_counts'].items()):
        print(f"  {category}: {count}")

    print(f"\nTask distribution:")
    for task, count in sorted(val_hard_stats['task_counts'].items()):
        print(f"  {task}: {count}")


if __name__ == '__main__':
    main()
