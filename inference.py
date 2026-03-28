"""Baseline inference script for the Email Triage OpenEnv environment.

Runs all three tasks (easy → medium → hard) using an LLM via the OpenAI
client, communicating with the environment through its HTTP API.

Credentials are read exclusively from environment variables:
    API_BASE_URL  — LLM endpoint (default: HF router)
    MODEL_NAME    — model identifier
    HF_TOKEN      — Hugging Face token (fallback: API_KEY)
"""

from __future__ import annotations

import json
import os
import sys
import time

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — all from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "")
HF_TOKEN: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")

ENV_URL: str = os.getenv("ENV_URL", "http://localhost:7860")
MAX_STEPS_PER_TASK: int = 8

SYSTEM_PROMPT: str = (
    "You are an expert email triage assistant. You are interacting with an email "
    "inbox environment. On each turn you receive a JSON observation describing "
    "the inbox state and the task you must complete.\n\n"
    "You MUST respond with ONLY a valid JSON object matching this schema:\n"
    "{\n"
    '  "action_type": "open" | "label" | "prioritise" | "reply" | "archive" | "skip" | "done",\n'
    '  "email_id": "<string or null>",\n'
    '  "label": "<string or null>",\n'
    '  "priority": "urgent" | "normal" | "low" | null,\n'
    '  "reply_text": "<string or null>"\n'
    "}\n\n"
    "Rules:\n"
    '- "open" requires email_id.\n'
    '- "label" requires email_id and label.\n'
    '- "prioritise" requires email_id and priority.\n'
    '- "reply" requires email_id and reply_text.\n'
    '- "archive" requires email_id.\n'
    '- "skip" and "done" require no extra fields.\n\n'
    "Output ONLY the JSON object. No markdown, no explanation, no extra text."
)


def build_user_prompt(observation: dict) -> str:
    """Build the user-turn prompt from the current observation."""
    # Provide a compact representation
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


def call_llm(client: OpenAI, messages: list[dict]) -> dict:
    """Call the LLM and parse the response into an action dict.

    Falls back to a 'skip' action on any failure.
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
        # Validate action_type
        valid_types = {"open", "label", "prioritise", "reply", "archive", "skip", "done"}
        if action.get("action_type") not in valid_types:
            raise ValueError(f"Invalid action_type: {action.get('action_type')}")
        return action

    except Exception as exc:
        print(f"  [LLM parse error: {exc}] — falling back to 'skip'")
        return {"action_type": "skip"}


def run_task(
    client: OpenAI,
    http: httpx.Client,
    task_name: str,
) -> float:
    """Run a single task episode and return the final score."""
    print(f"\n{'='*50}")
    print(f"  Task: {task_name}")
    print(f"{'='*50}")

    # Reset
    resp = http.post(f"{ENV_URL}/reset", json={"task_name": task_name})
    resp.raise_for_status()
    observation = resp.json()

    score = 0.0
    for step_i in range(MAX_STEPS_PER_TASK):
        if observation.get("done", False):
            print(f"  Step {step_i}: episode already done.")
            break

        # Build prompt and call LLM
        user_prompt = build_user_prompt(observation)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        action = call_llm(client, messages)
        print(f"  Step {step_i + 1}: {action.get('action_type', '?')} "
              f"(email_id={action.get('email_id', '-')})")

        # Step
        resp = http.post(f"{ENV_URL}/step", json=action)
        resp.raise_for_status()
        result = resp.json()

        observation = result["observation"]
        reward = result["reward"]
        done = result["done"]
        score = reward.get("score", 0.0)

        print(f"         score={score:.4f} | feedback: {reward.get('feedback', '')[:80]}")

        if done:
            print(f"  Episode done at step {step_i + 1}.")
            break

    print(f"  Final score: {score:.4f}")
    return score


def main() -> None:
    """Run inference across all three tasks and print the summary table."""
    if not MODEL_NAME:
        print("ERROR: MODEL_NAME environment variable is required.", file=sys.stderr)
        sys.exit(1)
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN (or API_KEY) environment variable is required.", file=sys.stderr)
        sys.exit(1)

    print(f"Model:    {MODEL_NAME}")
    print(f"API base: {API_BASE_URL}")
    print(f"Env URL:  {ENV_URL}")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    http = httpx.Client(timeout=60.0)

    # Check health
    try:
        health = http.get(f"{ENV_URL}/health")
        health.raise_for_status()
        print(f"Health check: {health.json()}")
    except Exception as exc:
        print(f"ERROR: Cannot reach environment at {ENV_URL}: {exc}", file=sys.stderr)
        sys.exit(1)

    scores: dict[str, float] = {}
    start = time.time()

    for task_name in ["easy", "medium", "hard"]:
        scores[task_name] = run_task(client, http, task_name)

    elapsed = time.time() - start
    avg = sum(scores.values()) / len(scores)

    print(f"\n{'='*50}")
    print("  RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"  {'Task':<12}| {'Score':>8}")
    print(f"  {'-'*12}|{'-'*9}")
    for task_name, s in scores.items():
        print(f"  {task_name:<12}| {s:>8.4f}")
    print(f"  {'-'*12}|{'-'*9}")
    print(f"  {'AVERAGE':<12}| {avg:>8.4f}")
    print(f"\n  Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
