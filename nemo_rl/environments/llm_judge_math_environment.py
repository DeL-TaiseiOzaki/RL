"""LLM-as-Judge Math Environment.

Combines binary math verification (hf_math_verify) with LLM partial credit scoring.
The LLM evaluates reasoning quality and awards partial credit for incorrect answers.

Reward formula:
  - Correct answer: 1.0
  - Incorrect answer: llm_weight * llm_score (0.0 to llm_weight)

This provides gradient information for the RL policy even when the final answer is wrong.
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, NotRequired, TypedDict, Union

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.environments.math_environment import (
    HFVerifyWorker,
    MathEnvironmentMetadata,
)
from nemo_rl.environments.metrics import calculate_pass_rate_per_prompt
from nemo_rl.environments.utils import chunk_list_to_workers

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """\
You are an expert math teacher evaluating a student's solution.

## Problem
{problem}

## Correct Answer
{ground_truth}

## Student's Response
{response}

## Task
Evaluate the mathematical reasoning process holistically. Consider:
- Understanding of the problem
- Correctness of the approach and method
- Logical flow of step-by-step reasoning
- Accuracy of intermediate calculations

Score from 0 to 10:
- 0: No relevant reasoning
- 1-3: Major conceptual errors, fundamentally wrong approach
- 4-5: Right general idea but significant errors
- 6-7: Good approach with minor errors leading to wrong answer
- 8-9: Nearly perfect reasoning with very small mistakes
- 10: Completely correct reasoning and answer

Respond ONLY with JSON: {{"score": <int 0-10>, "reason": "<1 sentence>"}}\
"""


class LLMJudgeConfig(TypedDict):
    enabled: bool
    model: NotRequired[str]
    max_completion_tokens: NotRequired[int]
    num_workers: NotRequired[int]
    max_concurrent_per_worker: NotRequired[int]
    llm_weight: NotRequired[float]
    reasoning_effort: NotRequired[str]


class LLMJudgeMathEnvConfig(TypedDict):
    num_workers: int
    math_verify_impl: NotRequired[str | None]
    verifier_type: NotRequired[str | None]
    stop_strings: NotRequired[list[str] | None]
    llm_judge: NotRequired[LLMJudgeConfig]


@ray.remote
class LLMJudgeWorker:
    """Ray worker that calls OpenAI API for partial credit scoring."""

    def __init__(
        self,
        model: str = "gpt-5-mini",
        max_completion_tokens: int = 16384,
        max_concurrent: int = 8,
        reasoning_effort: str = "low",
    ):
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.max_concurrent = max_concurrent
        self.reasoning_effort = reasoning_effort

    def judge(
        self,
        problems: list[str],
        responses: list[str],
        ground_truths: list[str],
        binary_scores: list[float],
    ) -> list[float]:
        """Judge a batch of responses. Only scores incorrect answers; correct ones get 1.0."""
        scores = [0.0] * len(problems)

        indices_to_judge = []
        for i, binary_score in enumerate(binary_scores):
            if binary_score > 0.0:
                scores[i] = 1.0
            else:
                indices_to_judge.append(i)

        if not indices_to_judge:
            return scores

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_idx = {}
            for idx in indices_to_judge:
                future = executor.submit(
                    self._judge_single,
                    problems[idx],
                    responses[idx],
                    ground_truths[idx],
                )
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    scores[idx] = future.result()
                except Exception as e:
                    logger.warning(f"LLM judge failed for index {idx}: {e}")
                    scores[idx] = 0.0

        return scores

    def _judge_single(self, problem: str, response: str, ground_truth: str) -> float:
        """Judge a single response via OpenAI API. Returns normalized score 0.0-1.0."""
        try:
            prompt = JUDGE_PROMPT.format(
                problem=problem,
                response=response,
                ground_truth=ground_truth,
            )

            create_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": self.max_completion_tokens,
                "response_format": {"type": "json_object"},
            }

            if self.reasoning_effort:
                create_kwargs["reasoning_effort"] = self.reasoning_effort

            result = self.client.chat.completions.create(**create_kwargs)

            content = result.choices[0].message.content
            data = json.loads(content)
            score = float(data["score"]) / 10.0
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"LLM judge API error: {e}")
            return 0.0


@ray.remote(max_restarts=-1, max_task_retries=-1)
class LLMJudgeMathEnvironment(EnvironmentInterface[MathEnvironmentMetadata]):
    """Math environment combining binary correctness with LLM partial credit.

    Reward formula:
      - Correct answer (binary=1): 1.0
      - Incorrect answer (binary=0): llm_weight * llm_score
    """

    def __init__(self, cfg: LLMJudgeMathEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]

        verifier_type = cfg.get("verifier_type", "math")
        assert verifier_type == "math", (
            f"LLMJudgeMathEnvironment only supports verifier_type='math', got '{verifier_type}'"
        )

        self.hf_workers = [
            HFVerifyWorker.options(
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote()
            for _ in range(self.num_workers)
        ]

        llm_cfg = cfg.get("llm_judge", {})
        self.llm_enabled = llm_cfg.get("enabled", False)
        self.llm_weight = llm_cfg.get("llm_weight", 0.5)

        if self.llm_enabled:
            llm_num_workers = llm_cfg.get("num_workers", 4)
            self.llm_workers = [
                LLMJudgeWorker.options(
                    runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
                ).remote(
                    model=llm_cfg.get("model", "gpt-5-mini"),
                    max_completion_tokens=llm_cfg.get("max_completion_tokens", 16384),
                    max_concurrent=llm_cfg.get("max_concurrent_per_worker", 8),
                    reasoning_effort=llm_cfg.get("reasoning_effort", "low"),
                )
                for _ in range(llm_num_workers)
            ]
            print(
                f"  ✓ LLM Judge initialized: model={llm_cfg.get('model', 'gpt-5-mini')}, "
                f"workers={llm_num_workers}, weight={self.llm_weight}"
            )

    def shutdown(self) -> None:
        for worker in self.hf_workers:
            ray.kill(worker)
        if self.llm_enabled:
            for worker in self.llm_workers:
                ray.kill(worker)

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[MathEnvironmentMetadata],
        return_extracted_answer: bool = False,
    ) -> EnvironmentReturn[MathEnvironmentMetadata]:
        # --- Extract assistant responses ---
        assistant_response_batch = []
        for conversation in message_log_batch:
            assistant_responses = [
                str(interaction["content"])
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            assistant_response_batch.append("".join(assistant_responses))

        ground_truths = [g["ground_truth"] for g in metadata]

        # --- Step 1: Binary math verification ---
        chunked_responses = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_ground_truths = chunk_list_to_workers(
            ground_truths, self.num_workers
        )

        hf_futures = [
            self.hf_workers[i].verify.remote(
                chunk,
                gt_chunk,
                return_extracted_answer,
                math_verify_impl=self.cfg.get("math_verify_impl", "hf_math_verify"),
            )
            for i, (chunk, gt_chunk) in enumerate(
                zip(chunked_responses, chunked_ground_truths)
            )
        ]

        hf_worker_results = ray.get(hf_futures)

        binary_scores: list[float] = []
        extracted_answers: list[str | None] | None = (
            [] if return_extracted_answer else None
        )

        for worker_result in hf_worker_results:
            if return_extracted_answer:
                worker_scores, worker_answers = worker_result
                binary_scores.extend(worker_scores)
                extracted_answers.extend(worker_answers)
            else:
                binary_scores.extend(worker_result)

        # --- Step 2: LLM Judge partial credit ---
        if self.llm_enabled:
            problem_batch = []
            for conversation in message_log_batch:
                user_messages = [
                    str(interaction["content"])
                    for interaction in conversation
                    if interaction["role"] == "user"
                ]
                problem_batch.append("".join(user_messages))

            llm_num_workers = len(self.llm_workers)
            chunked_problems = chunk_list_to_workers(problem_batch, llm_num_workers)
            chunked_llm_responses = chunk_list_to_workers(
                assistant_response_batch, llm_num_workers
            )
            chunked_llm_gts = chunk_list_to_workers(ground_truths, llm_num_workers)
            chunked_binary = chunk_list_to_workers(binary_scores, llm_num_workers)

            llm_futures = [
                self.llm_workers[i].judge.remote(
                    chunked_problems[i],
                    chunked_llm_responses[i],
                    chunked_llm_gts[i],
                    chunked_binary[i],
                )
                for i in range(llm_num_workers)
            ]

            llm_worker_results = ray.get(llm_futures)
            llm_scores: list[float] = []
            for worker_result in llm_worker_results:
                llm_scores.extend(worker_result)

            # Combine: correct=1.0, incorrect=llm_weight * llm_score
            combined_results = []
            for binary, llm in zip(binary_scores, llm_scores):
                if binary > 0.0:
                    combined_results.append(1.0)
                else:
                    combined_results.append(self.llm_weight * llm)
        else:
            combined_results = binary_scores

        # --- Build return ---
        observations = [
            {
                "role": "environment",
                "content": "Environment: correct"
                if binary_scores[i] > 0
                else "Environment: incorrect",
            }
            for i in range(len(message_log_batch))
        ]

        rewards = torch.tensor(combined_results).cpu()
        done = torch.ones_like(rewards).cpu()
        next_stop_strings = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=done,
            answers=extracted_answers,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        batch["rewards"] = batch["rewards"] * batch["is_end"]

        # Binary accuracy: correct answers have reward >= 1.0
        binary_correct = (batch["rewards"] >= 1.0).float()

        if binary_correct.sum() > 0:
            correct_solution_generation_lengths = (
                (batch["generation_lengths"] - batch["prompt_lengths"])[
                    binary_correct.bool()
                ]
                .float()
                .mean()
                .item()
            )
        else:
            correct_solution_generation_lengths = 0

        # For partial credit samples (incorrect but with some LLM score)
        incorrect_mask = (batch["rewards"] < 1.0) & (batch["rewards"] > 0.0)
        partial_credit_avg = (
            batch["rewards"][incorrect_mask].mean().item()
            if incorrect_mask.any()
            else 0.0
        )

        metrics = {
            "accuracy": binary_correct.mean().item(),
            "combined_reward": batch["rewards"].mean().item(),
            "partial_credit_avg": partial_credit_avg,
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(
                batch["text"], batch["rewards"]
            ),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
        }

        return batch, metrics
