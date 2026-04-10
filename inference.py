"""
Inference Script for Email Triage OpenEnv Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- Defaults are set only for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

- The inference script must be named `inference.py` and placed in the root directory
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script emits exactly three line types to stdout:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — all from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK = "email-triage-env"
MAX_STEPS = 10
SUCCESS_THRESHOLD = 0.5

SYSTEM_PROMPT = """You are an expert email triage assistant. You are interacting with an email inbox environment. On each turn you receive a JSON observation describing the inbox state and the task you must complete.

You MUST respond with ONLY a valid JSON object matching this schema:
{
  "action_type": "open" | "label" | "prioritise" | "reply" | "archive" | "skip" | "done",
  "email_id": "<string or null>",
  "label": "<string or null>",
  "priority": "urgent" | "normal" | "low" | null,
  "reply_text": "<string or null>"
}

Rules:
- "open" requires email_id.
- "label" requires email_id and label.
- "prioritise" requires email_id and priority.
- "reply" requires email_id and reply_text.
- "archive" requires email_id.
- "skip" and "done" require no extra fields.

Output ONLY the JSON object. No markdown, no explanation, no extra text."""


# ---------------------------------------------------------------------------
# Structured logging helpers (MANDATORY FORMAT)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_user_prompt(observation: dict) -> str:
    """Build the user-turn prompt from the current observation."""
    task = observation.get("task_description", "")
    step = observation.get("step_count", 0)
    done = observation.get("done", False)
    history = observation.get("action_history", [])

    inbox_summary: list[str] = []
    for email in observation.get("inbox", []):
        status_parts: list[str] = []
        if email.get("is_read"):
            status_parts.append("read")
        if email.get("labels"):
            status_parts.append(f"labels={email['labels']}")
        if email.get("priority"):
            status_parts.append(f"priority={email['priority']}")
        status = ", ".join(status_parts) if status_parts else "unread"
        inbox_summary.append(
            f"  - id={email['id']} | from={email['sender']} | "
            f"subject=\"{email['subject']}\" | {status}"
        )

    current = observation.get("current_email")
    current_section = ""
    if current:
        current_section = (
            f"\n\nCurrently open email:\n"
            f"  ID: {current['id']}\n"
            f"  From: {current['sender']}\n"
            f"  Subject: {current['subject']}\n"
            f"  Body:\n{current['body']}\n"
            f"  Labels: {current.get('labels', [])}\n"
            f"  Priority: {current.get('priority')}"
        )

    prompt = (
        f"TASK: {task}\n"
        f"Step: {step} | Done: {done}\n"
        f"Action history: {history[-5:]}\n\n"
        f"Inbox ({len(observation.get('inbox', []))} emails):\n"
        + "\n".join(inbox_summary)
        + current_section
        + "\n\nWhat is your next action? Respond with ONLY a JSON object."
    )
    return prompt


def call_llm(client: OpenAI, messages: list[dict]) -> tuple[dict, Optional[str]]:
    """Call the LLM and parse the response into an action dict.
    
    Returns (action_dict, error_string or None).
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=512,
        )
        content: str = response.choices[0].message.content or ""
        content = content.strip()

        # Strip markdown fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        action = json.loads(content)
        valid_types = {"open", "label", "prioritise", "reply", "archive", "skip", "done"}
        if action.get("action_type") not in valid_types:
            return {"action_type": "skip"}, f"Invalid action_type: {action.get('action_type')}"
        return action, None

    except Exception as exc:
        return {"action_type": "skip"}, str(exc)


def run_task(client: OpenAI, http: httpx.Client, task_name: str) -> float:
    """Run a single task episode and return the final score."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment
        resp = http.post(f"{ENV_URL}/reset", json={"task_name": task_name})
        resp.raise_for_status()
        observation = resp.json()

        for step in range(1, MAX_STEPS + 1):
            if observation.get("done", False):
                break

            # Build prompt and call LLM
            user_prompt = build_user_prompt(observation)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            action, error = call_llm(client, messages)
            
            # Format action string for logging
            action_str = f"{action.get('action_type', 'unknown')}"
            if action.get('email_id'):
                action_str += f"({action['email_id']})"

            # Step environment
            resp = http.post(f"{ENV_URL}/step", json=action)
            resp.raise_for_status()
            result = resp.json()

            observation = result["observation"]
            reward_obj = result["reward"]
            done = result["done"]
            
            reward = reward_obj.get("score", 0.0)
            rewards.append(reward)
            steps_taken = step
            score = reward  # Final score is the last reward

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        log_step(step=steps_taken + 1, action="error", reward=0.0, done=True, error=str(exc))

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return score


def main() -> None:
    """Run inference across all three tasks."""
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is required.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    http = httpx.Client(timeout=60.0)

    # Check environment health
    try:
        health = http.get(f"{ENV_URL}/health")
        health.raise_for_status()
    except Exception as exc:
        print(f"ERROR: Cannot reach environment at {ENV_URL}: {exc}", file=sys.stderr)
        sys.exit(1)

    # Run all three tasks
    scores: dict[str, float] = {}
    for task_name in ["easy", "medium", "hard"]:
        scores[task_name] = run_task(client, http, task_name)

    # Print summary
    avg = sum(scores.values()) / len(scores)
    print(f"\n--- SUMMARY ---", flush=True)
    for task_name, s in scores.items():
        print(f"  {task_name}: {s:.2f}", flush=True)
    print(f"  AVERAGE: {avg:.2f}", flush=True)


if __name__ == "__main__":
    main()
