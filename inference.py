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
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

ENV_URL            = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK          = "email-triage-env"
MAX_STEPS          = 10
SUCCESS_THRESHOLD  = 0.5
LLM_RETRY_LIMIT    = 3   # retries on invalid/unparseable LLM output

SYSTEM_PROMPT = """You are an expert email triage assistant interacting with an email inbox environment.
On each turn you receive a JSON observation describing the inbox state and the current task.

You MUST respond with ONLY a valid JSON object matching this schema:
{
  "action_type": "open" | "label" | "prioritise" | "reply" | "archive" | "skip" | "done",
  "email_id":   "<string or null>",
  "label":      "<string or null>",
  "priority":   "urgent" | "normal" | "low" | null,
  "reply_text": "<string or null>"
}

Rules:
- "open"       requires email_id.
- "label"      requires email_id and label.
- "prioritise" requires email_id and priority.
- "reply"      requires email_id and reply_text.
- "archive"    requires email_id.
- "skip" and "done" require no extra fields.
- When all required actions are complete, output {"action_type": "done"}.

Output ONLY the JSON object. No markdown, no explanation, no extra text."""

VALID_ACTION_TYPES = {"open", "label", "prioritise", "reply", "archive", "skip", "done"}


# ---------------------------------------------------------------------------
# Structured logging helpers (MANDATORY FORMAT)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_user_prompt(observation: dict) -> str:
    """Build the user-turn prompt from the current observation."""
    task    = observation.get("task_description", "")
    step    = observation.get("step_count", 0)
    done    = observation.get("done", False)
    history = observation.get("action_history", [])

    inbox_summary: list[str] = []
    for email in observation.get("inbox", []):
        parts: list[str] = []
        if email.get("is_read"):
            parts.append("read")
        if email.get("labels"):
            parts.append(f"labels={email['labels']}")
        if email.get("priority"):
            parts.append(f"priority={email['priority']}")
        status = ", ".join(parts) if parts else "unread"
        inbox_summary.append(
            f"  - id={email['id']} | from={email['sender']} | "
            f"subject=\"{email['subject']}\" | {status}"
        )

    current         = observation.get("current_email")
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

    # Only show the last 5 history entries to keep context concise
    recent_history = history[-5:] if len(history) > 5 else history

    prompt = (
        f"TASK: {task}\n"
        f"Step: {step} | Done: {done}\n"
        f"Recent action history: {recent_history}\n\n"
        f"Inbox ({len(observation.get('inbox', []))} emails):\n"
        + "\n".join(inbox_summary)
        + current_section
        + "\n\nWhat is your next action? Respond with ONLY a JSON object."
    )
    return prompt


# ---------------------------------------------------------------------------
# LLM call with retry on bad output
# ---------------------------------------------------------------------------

def call_llm(client: OpenAI, messages: list[dict]) -> tuple[dict, Optional[str]]:
    """Call the LLM and parse the response into an action dict.

    Retries up to LLM_RETRY_LIMIT times if the model returns unparseable
    or invalid JSON, appending the error as a correction nudge.

    Returns (action_dict, llm_error_string_or_None).
    """
    working_messages = list(messages)  # copy so we can append correction turns
    raw = ""

    for attempt in range(LLM_RETRY_LIMIT):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=working_messages,
                temperature=0.0,
                max_tokens=512,
            )
            raw = response.choices[0].message.content or ""
            raw = raw.strip()

            # Strip markdown fences if the model added them
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            action = json.loads(raw)

            if action.get("action_type") not in VALID_ACTION_TYPES:
                raise ValueError(f"Invalid action_type: {action.get('action_type')!r}")

            return action, None

        except Exception as exc:
            err_msg = str(exc)
            if attempt < LLM_RETRY_LIMIT - 1:
                # Nudge the model to correct itself
                working_messages.append({"role": "assistant", "content": raw})
                working_messages.append({
                    "role": "user",
                    "content": (
                        f"Your last response was invalid: {err_msg}. "
                        "Please respond with ONLY a valid JSON object matching the schema."
                    ),
                })
            else:
                return {"action_type": "skip"}, err_msg

    return {"action_type": "skip"}, "Max retries exceeded"


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, http: httpx.Client, task_name: str) -> float:
    """Run a single task episode; returns the final cumulative score."""
    rewards:      List[float]      = []
    steps_taken:  int              = 0
    final_score:  float            = 0.0
    success:      bool             = False
    conversation: list[dict]       = []   # multi-turn memory for the LLM

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # ── Reset environment ──────────────────────────────────────────────
        resp = http.post(f"{ENV_URL}/reset", json={"task_name": task_name})
        resp.raise_for_status()
        observation = resp.json()

        # Initialise conversation with the system prompt (once per episode)
        conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step in range(1, MAX_STEPS + 1):
            if observation.get("done", False):
                break

            # ── Build prompt and append as user turn ───────────────────────
            user_prompt = build_user_prompt(observation)
            conversation.append({"role": "user", "content": user_prompt})

            # ── Call LLM (with retry logic) ────────────────────────────────
            action, llm_error = call_llm(client, conversation)

            # Append the assistant's logical reply to conversation history
            conversation.append({
                "role": "assistant",
                "content": json.dumps(action),
            })

            # ── Format action string for logging ──────────────────────────
            action_str = action.get("action_type", "unknown")
            if action.get("email_id"):
                action_str += f"({action['email_id']})"

            # ── Step the environment ───────────────────────────────────────
            resp = http.post(f"{ENV_URL}/step", json=action)
            resp.raise_for_status()
            result = resp.json()

            observation    = result.get("observation", {})
            reward_obj     = result.get("reward", {})
            done           = result.get("done", False)

            # Prefer cumulative/final score field; fall back to step score
            reward = (
                reward_obj.get("final_score")
                or reward_obj.get("cumulative_score")
                or reward_obj.get("score", 0.01)
            )
            reward = float(reward)
            # Clamp to strictly between 0 and 1
            reward = max(0.01, min(0.99, reward))
            rewards.append(reward)
            steps_taken = step

            # env_error is the authoritative error field per spec
            env_error = observation.get("last_action_error") or None

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=env_error,
            )

            if done:
                final_score = reward
                break

        # If episode ended without a done=true, use the last reward as score
        if not final_score and rewards:
            final_score = rewards[-1]

        # Clamp final_score to strictly between 0 and 1
        final_score = max(0.01, min(0.99, final_score))
        success = final_score >= SUCCESS_THRESHOLD

    except Exception as exc:
        # Always emit a [STEP] for the crash so evaluators see it
        log_step(
            step=steps_taken + 1,
            action="error",
            reward=0.01,
            done=True,
            error=str(exc),
        )
        rewards.append(0.01)
        final_score = 0.01  # Ensure final_score is set on exception
        print(f"[ERROR] task={task_name} {exc}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return final_score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN environment variable is required")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    http   = httpx.Client(timeout=60.0)

    # ── Verify environment is reachable ────────────────────────────────────
    try:
        health = http.get(f"{ENV_URL}/health")
        health.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Cannot reach environment at {ENV_URL}: {exc}") from exc

    # ── Run all three difficulty tasks ─────────────────────────────────────
    tasks  = ["easy", "medium", "hard"]
    scores: dict[str, float] = {}
    for task_name in tasks:
        scores[task_name] = run_task(client, http, task_name)

    # ── Print summary to stderr only (keeps stdout clean for evaluators) ───
    avg = sum(scores.values()) / len(scores)
    print("--- SUMMARY ---", file=sys.stderr)
    for task_name, s in scores.items():
        print(f"  {task_name}: {s:.2f}", file=sys.stderr)
    print(f"  AVERAGE: {avg:.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()
