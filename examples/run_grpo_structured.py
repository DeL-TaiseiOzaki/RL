# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GRPO training script for structured data tasks (JSON, YAML, TOML, XML, CSV).

Customizes the standard GRPO pipeline with:
- A data processor that handles OpenAI chat format with extra_env_info
- A StructuredDataEnvironment that evaluates:
  1. Parse validity — can the output be parsed as the target format?
  2. Structural completeness — do key paths from ground truth exist?
"""

import argparse
import csv
import io
import json
import os
import pprint
import re
import tomllib
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any, NotRequired, Optional, TypedDict, Union

import ray
import torch
import yaml
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType, TaskDataSpec
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES, init_ray
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.metrics import calculate_pass_rate_per_prompt
from nemo_rl.environments.utils import chunk_list_to_workers
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)

TokenizerType = PreTrainedTokenizerBase


# =============================================================================
#                         Data Processor
# =============================================================================
def structured_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a structured data datum into a DatumSpec for GRPO training."""
    messages = datum_dict["messages"]
    extra_env_info = datum_dict.get("extra_env_info", {})
    ground_truth = extra_env_info.get("ground_truth", "")

    message_log: LLMMessageLogType = []

    system_msgs = [m for m in messages if m["role"] == "system"]
    user_msgs = [m for m in messages if m["role"] == "user"]

    chat_messages = []
    if system_msgs:
        chat_messages.append(
            {"role": "system", "content": system_msgs[0]["content"]}
        )
    if user_msgs:
        chat_messages.append({"role": "user", "content": user_msgs[0]["content"]})
    elif not chat_messages:
        chat_messages.append({"role": "user", "content": messages[0]["content"]})

    formatted = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )

    token_ids = tokenizer(
        formatted,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]

    prompt_entry: dict[str, Any] = {
        "role": "user",
        "content": formatted,
        "token_ids": token_ids,
    }
    message_log.append(prompt_entry)

    length = len(token_ids)
    loss_multiplier = 1.0
    if length > max_seq_length:
        # Truncate to max_seq_length (not 4 tokens) and skip gradient
        prompt_entry["token_ids"] = prompt_entry["token_ids"][:max_seq_length]
        loss_multiplier = 0.0

    # Detect target format from user prompt
    user_content = user_msgs[0]["content"] if user_msgs else ""
    target_format = _detect_target_format(user_content)

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": {
            "ground_truth": ground_truth,
            "target_format": target_format,
        },
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": datum_dict.get("task_name", "structured"),
    }
    return output


def _detect_target_format(user_content: str) -> str:
    """Detect the target output format from user prompt text."""
    content_lower = user_content.lower()

    # Check for explicit output format mentions
    # Order matters: check more specific patterns first
    if "output json" in content_lower or "into json" in content_lower or "to json" in content_lower:
        return "json"
    if "output yaml" in content_lower or "into yaml" in content_lower or "to yaml" in content_lower:
        return "yaml"
    if "output toml" in content_lower or "into toml" in content_lower or "to toml" in content_lower:
        return "toml"
    if "output xml" in content_lower or "into xml" in content_lower or "to xml" in content_lower:
        return "xml"
    if "output csv" in content_lower or "into csv" in content_lower or "to csv" in content_lower:
        return "csv"

    # Check for format-specific keywords
    if "json" in content_lower:
        return "json"
    if "yaml" in content_lower:
        return "yaml"
    if "toml" in content_lower:
        return "toml"
    if "xml" in content_lower:
        return "xml"
    if "csv" in content_lower:
        return "csv"

    return "unknown"


# =============================================================================
#                     Structured Output Parsing
# =============================================================================
def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences from text."""
    text = re.sub(r"^```\w*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())
    return text.strip()


def _try_parse(text: str, target_format: str) -> tuple[bool, Any]:
    """Try to parse text as the target format.

    Returns (success, parsed_object).
    """
    text = _strip_code_fences(text)

    if target_format == "json":
        return _parse_json(text)
    if target_format == "yaml":
        return _parse_yaml(text)
    if target_format == "toml":
        return _parse_toml(text)
    if target_format == "xml":
        return _parse_xml(text)
    if target_format == "csv":
        return _parse_csv(text)

    # Unknown format: try all parsers
    for parser in (_parse_json, _parse_yaml, _parse_toml, _parse_xml, _parse_csv):
        ok, obj = parser(text)
        if ok:
            return ok, obj
    return False, None


def _parse_json(text: str) -> tuple[bool, Any]:
    try:
        obj = json.loads(text)
        return True, obj
    except (json.JSONDecodeError, ValueError):
        return False, None


def _parse_yaml(text: str) -> tuple[bool, Any]:
    try:
        obj = yaml.safe_load(text)
        if obj is None or isinstance(obj, str):
            return False, None
        return True, obj
    except Exception:
        return False, None


def _parse_toml(text: str) -> tuple[bool, Any]:
    try:
        obj = tomllib.loads(text)
        return True, obj
    except Exception:
        return False, None


def _parse_xml(text: str) -> tuple[bool, Any]:
    try:
        root = ET.fromstring(text)
        return True, root
    except ET.ParseError:
        # Try wrapping in root if it's a fragment
        try:
            root = ET.fromstring(f"<_root>{text}</_root>")
            return True, root
        except ET.ParseError:
            return False, None


def _parse_csv(text: str) -> tuple[bool, Any]:
    try:
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        if len(rows) < 1:
            return False, None
        return True, rows
    except Exception:
        return False, None


# =============================================================================
#                     Key Path Extraction & Matching
# =============================================================================
def _extract_key_paths(obj: Any, prefix: str = "") -> set[str]:
    """Recursively extract all key paths from a parsed object.

    Examples:
      {"title": "X", "authors": [{"name": "A"}]}
      -> {"title", "authors", "authors[0]", "authors[0].name"}
    """
    paths = set()

    if isinstance(obj, dict):
        for key, value in obj.items():
            current = f"{prefix}.{key}" if prefix else key
            paths.add(current)
            paths.update(_extract_key_paths(value, current))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            current = f"{prefix}[{i}]"
            paths.add(current)
            paths.update(_extract_key_paths(item, current))
    elif isinstance(obj, ET.Element):
        paths.update(_extract_xml_key_paths(obj, prefix))

    return paths


def _extract_xml_key_paths(element: ET.Element, prefix: str = "") -> set[str]:
    """Extract key paths from an XML Element tree."""
    paths = set()

    # Group children by tag to detect lists
    tag_counts: dict[str, int] = {}
    tag_indices: dict[str, int] = {}
    for child in element:
        tag_counts[child.tag] = tag_counts.get(child.tag, 0) + 1

    for child in element:
        if tag_counts[child.tag] > 1:
            # Repeated tag = list item
            idx = tag_indices.get(child.tag, 0)
            tag_indices[child.tag] = idx + 1
            current = f"{prefix}.{child.tag}[{idx}]" if prefix else f"{child.tag}[{idx}]"
        else:
            current = f"{prefix}.{child.tag}" if prefix else child.tag

        paths.add(current)

        if len(child) > 0:
            # Has child elements
            paths.update(_extract_xml_key_paths(child, current))
        # Leaf element with text is just the path itself (already added)

    return paths


def _extract_csv_key_paths(rows: list[list[str]]) -> set[str]:
    """Extract key paths from CSV rows (header + data rows)."""
    paths = set()
    if len(rows) < 2:
        return paths

    headers = rows[0]
    for i, header in enumerate(headers):
        if header.strip():
            paths.add(header.strip())

    # Add row indices
    for row_idx in range(1, len(rows)):
        for col_idx, header in enumerate(headers):
            if header.strip() and col_idx < len(rows[row_idx]):
                paths.add(f"row[{row_idx - 1}].{header.strip()}")

    return paths


# Reward weights — all components are competition-aligned
_W_PARSE_BASE = 0.2  # Credit for producing parseable output
_W_KEY_PATH = 0.5  # Continuous: fraction of GT key paths found
_W_CLEAN = 0.1  # No code fences or extraneous text
_W_PERFECT = 0.2  # Step bonus: ALL GT key paths present


def _check_cleanliness(raw_response: str, cleaned: str, target_format: str) -> bool:
    """Check if the raw response is clean (no code fences, no extra text).

    A clean response is directly parseable without stripping code fences
    and does not contain extraneous text before/after the structured data.
    """
    stripped = raw_response.strip()

    # Code fences present → not clean
    if stripped != cleaned:
        return False

    # Check for leading/trailing prose that isn't part of the structured data
    if target_format == "json":
        return stripped.startswith(("{", "["))
    if target_format == "yaml":
        # YAML can start with various chars; reject obvious prose
        first_line = stripped.split("\n", 1)[0]
        return not first_line[0:1].isalpha() or ":" in first_line
    if target_format == "toml":
        first_line = stripped.split("\n", 1)[0]
        return "[" in first_line or "=" in first_line
    if target_format == "xml":
        return stripped.startswith("<")
    if target_format == "csv":
        # CSV should start with header row; reject obvious prose sentences
        first_line = stripped.split("\n", 1)[0]
        return "," in first_line

    return True


def compute_structural_score(
    response: str, ground_truth: str, target_format: str
) -> tuple[float, str | None]:
    """Compute multi-component structural score between response and ground truth.

    Competition-aligned reward with sufficient variance for GRPO:
      0.0                — unparseable or empty
      0.2                — parse success base
      + 0.5 * fraction   — key path coverage (continuous)
      + 0.1              — cleanliness bonus (no code fences / extra text)
      + 0.2              — perfect bonus (all GT key paths matched)
      = max 1.0
    """
    if not response.strip():
        return 0.0, None

    raw_response = response.strip()
    cleaned = _strip_code_fences(raw_response)

    # Step 1: Parse response
    resp_ok, resp_obj = _try_parse(cleaned, target_format)
    if not resp_ok:
        return 0.0, cleaned

    # Step 2: Cleanliness check (before GT comparison)
    is_clean = _check_cleanliness(raw_response, cleaned, target_format)

    # Step 3: Parse ground truth to extract expected key paths
    gt_cleaned = _strip_code_fences(ground_truth)
    gt_ok, gt_obj = _try_parse(gt_cleaned, target_format)

    if not gt_ok:
        # Can't parse ground truth — give credit for parse + cleanliness only
        reward = _W_PARSE_BASE + (_W_CLEAN if is_clean else 0.0)
        return reward, cleaned

    # Step 4: Extract key paths from both
    if target_format == "csv":
        gt_rows = gt_obj if isinstance(gt_obj, list) else []
        resp_rows = resp_obj if isinstance(resp_obj, list) else []
        gt_paths = _extract_csv_key_paths(gt_rows)
        resp_paths = _extract_csv_key_paths(resp_rows)
    elif target_format == "xml":
        gt_paths = _extract_xml_key_paths(gt_obj) if isinstance(gt_obj, ET.Element) else set()
        resp_paths = _extract_xml_key_paths(resp_obj) if isinstance(resp_obj, ET.Element) else set()
    else:
        gt_paths = _extract_key_paths(gt_obj)
        resp_paths = _extract_key_paths(resp_obj)

    if not gt_paths:
        reward = _W_PARSE_BASE + (_W_CLEAN if is_clean else 0.0)
        return reward, cleaned

    # Step 5: Multi-component reward
    matched = gt_paths & resp_paths
    key_fraction = len(matched) / len(gt_paths)

    reward = _W_PARSE_BASE
    reward += _W_KEY_PATH * key_fraction
    if is_clean:
        reward += _W_CLEAN
    if key_fraction >= 1.0:
        reward += _W_PERFECT

    return reward, cleaned


# =============================================================================
#                     Structured Data Verify Worker
# =============================================================================
@ray.remote
class StructuredVerifyWorker:
    """Verifies structured data output correctness."""

    def verify(
        self,
        pred_responses: list[str],
        ground_truths: list[str],
        target_formats: list[str],
        return_extracted_answer: bool = False,
        **kwargs: Any,
    ) -> Union[list[float], tuple[list[float], list[str | None]]]:
        results: list[float] = []
        extracted_answers: list[str | None] = []

        for response, ground_truth, fmt in zip(
            pred_responses, ground_truths, target_formats
        ):
            score, extracted = compute_structural_score(
                response, ground_truth, fmt
            )
            results.append(score)
            if return_extracted_answer:
                extracted_answers.append(extracted)

        if return_extracted_answer:
            return results, extracted_answers
        return results


# =============================================================================
#                     Structured Data Environment
# =============================================================================
class StructuredEnvConfig(TypedDict):
    num_workers: int
    stop_strings: NotRequired[list[str] | None]


class StructuredEnvironmentMetadata(TypedDict):
    ground_truth: str
    target_format: str
    extracted_answer: str | None


@ray.remote(max_restarts=-1, max_task_retries=-1)
class StructuredDataEnvironment(EnvironmentInterface[StructuredEnvironmentMetadata]):
    """Environment for evaluating structured data generation tasks."""

    def __init__(self, cfg: StructuredEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.workers = [
            StructuredVerifyWorker.options(
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote()
            for _ in range(self.num_workers)
        ]

    def shutdown(self) -> None:
        for worker in self.workers:
            ray.kill(worker)

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[StructuredEnvironmentMetadata],
        return_extracted_answer: bool = False,
    ) -> EnvironmentReturn[StructuredEnvironmentMetadata]:
        # Extract assistant responses
        assistant_response_batch = []
        for conversation in message_log_batch:
            assistant_responses = [
                str(interaction["content"])
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            assistant_response_batch.append("".join(assistant_responses))

        ground_truths = [g["ground_truth"] for g in metadata]
        target_formats = [g.get("target_format", "unknown") for g in metadata]

        # Distribute work across verify workers
        chunked_responses = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_truths = chunk_list_to_workers(ground_truths, self.num_workers)
        chunked_formats = chunk_list_to_workers(target_formats, self.num_workers)

        futures = [
            self.workers[i].verify.remote(
                chunk_resp,
                chunk_truth,
                chunk_fmt,
                return_extracted_answer,
            )
            for i, (chunk_resp, chunk_truth, chunk_fmt) in enumerate(
                zip(chunked_responses, chunked_truths, chunked_formats)
            )
        ]

        worker_results = ray.get(futures)

        results: list[float] = []
        extracted_answers: list[str | None] | None = (
            [] if return_extracted_answer else None
        )

        for worker_result in worker_results:
            if return_extracted_answer:
                worker_scores, worker_answers = worker_result
                results.extend(worker_scores)
                extracted_answers.extend(worker_answers)
            else:
                results.extend(worker_result)

        observations = [
            {
                "role": "environment",
                "content": f"Environment: score={result:.2f}",
            }
            for result in results
        ]

        rewards = torch.tensor(results).cpu()
        done = torch.ones_like(rewards).cpu()
        next_stop_strings: list[list[str] | None] = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=done,
            answers=extracted_answers,
        )

    def global_post_process_and_metrics(
        self, batch: Any
    ) -> tuple[Any, dict[str, float | int]]:
        batch["rewards"] = batch["rewards"] * batch["is_end"]

        perfect_threshold = _W_PARSE_BASE + _W_KEY_PATH + _W_PERFECT  # 0.9
        if (batch["rewards"] >= perfect_threshold).float().sum() > 0:
            correct_gen_lengths = (
                (batch["generation_lengths"] - batch["prompt_lengths"])[
                    batch["rewards"] >= perfect_threshold
                ]
                .float()
                .mean()
                .item()
            )
        else:
            correct_gen_lengths = 0

        rewards = batch["rewards"]

        # Reward thresholds aligned with multi-component scoring:
        #   0.0         = parse failure
        #   0.2         = parse only (no key paths matched)
        #   0.2~0.7     = partial key path match (no perfect bonus)
        #   0.7~0.8     = full key path match without cleanliness
        #   0.8~1.0     = full key path match with cleanliness (perfect)
        parse_success_mask = rewards > 0.0

        metrics = {
            "perfect_rate": (rewards >= perfect_threshold).float().mean().item(),
            "mean_reward": rewards.mean().item(),
            "reward_std": rewards.std().item() if rewards.numel() > 1 else 0.0,
            "parse_failure_rate": (~parse_success_mask).float().mean().item(),
            "parse_success_rate": parse_success_mask.float().mean().item(),
            "partial_credit": (
                parse_success_mask & (rewards < perfect_threshold)
            )
            .float()
            .mean()
            .item(),
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(
                batch["text"], rewards
            ),
            "fraction_of_samples_properly_ended": batch["is_end"]
            .float()
            .mean()
            .item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "correct_solution_generation_lengths": correct_gen_lengths,
        }

        return batch, metrics


# =============================================================================
#                         Data Setup
# =============================================================================
def setup_data(
    tokenizer: TokenizerType,
    data_config: dict[str, Any],
    env_configs: dict[str, Any],
    seed: int,
) -> tuple[
    AllTaskProcessedDataset,
    Optional[AllTaskProcessedDataset],
    dict[str, EnvironmentInterface],
    dict[str, EnvironmentInterface],
]:
    print("\n>>> Setting up structured data...")

    structured_task_spec = TaskDataSpec(
        task_name="structured",
        prompt_file=None,
        system_prompt_file=None,
    )

    # Load JSONL data
    train_ds = load_dataset(
        "json", data_files=data_config["train_data_path"], split="train"
    )
    val_ds = load_dataset(
        "json", data_files=data_config["val_data_path"], split="train"
    )
    print(f"  Loaded train={len(train_ds)}, val={len(val_ds)} samples")

    # Add task_name column if not present
    if "task_name" not in train_ds.column_names:
        train_ds = train_ds.map(lambda x: {"task_name": "structured"})
    if "task_name" not in val_ds.column_names:
        val_ds = val_ds.map(lambda x: {"task_name": "structured"})

    # Processor mapping
    task_data_processors: dict[str, tuple[TaskDataSpec, Any]] = defaultdict(
        lambda: (structured_task_spec, structured_data_processor)
    )
    task_data_processors["structured"] = (
        structured_task_spec,
        structured_data_processor,
    )

    # Create environment
    structured_env = StructuredDataEnvironment.options(
        runtime_env={
            "py_executable": PY_EXECUTABLES.SYSTEM,
            "env_vars": dict(os.environ),
        }
    ).remote(env_configs["structured"])

    dataset = AllTaskProcessedDataset(
        train_ds,
        tokenizer,
        structured_task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    val_dataset = AllTaskProcessedDataset(
        val_ds,
        tokenizer,
        structured_task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    task_to_env: dict[str, EnvironmentInterface] = defaultdict(lambda: structured_env)
    task_to_env["structured"] = structured_env

    return dataset, val_dataset, task_to_env, task_to_env


# =============================================================================
#                              Main
# =============================================================================
def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run GRPO training on structured data"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    args, overrides = parser.parse_known_args()
    return args, overrides


def main() -> None:
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_qwen3_4b_structured.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    print("Final config:")
    pprint.pprint(config)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(f"Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}")

    init_ray()

    # Setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None, (
        "A generation config is required for GRPO"
    )
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    # Setup data and environment
    dataset, val_dataset, task_to_env, val_task_to_env = setup_data(
        tokenizer, config["data"], config["env"], config["grpo"]["seed"]
    )

    # Setup GRPO components
    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    print("Running synchronous GRPO training")
    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        val_task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


if __name__ == "__main__":
    main()
