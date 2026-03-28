"""Task registry for the Email Triage environment.

Defines three difficulty levels — easy, medium, hard — each with its own
task description, max steps, and grading configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class TaskConfig:
    """Configuration for a single triage task."""

    name: str
    difficulty: str
    description: str
    max_steps: int
    email_count: int
    seed: int
    grader_weights: Dict[str, float] = field(default_factory=dict)


TASK_REGISTRY: Dict[str, TaskConfig] = {
    "easy": TaskConfig(
        name="easy",
        difficulty="easy",
        description=(
            "Label the Urgent Email — Your inbox contains 5 emails. Exactly "
            "one of them is genuinely urgent (time-sensitive financial, legal, "
            "or security matter). Open emails to read them, identify the urgent "
            "one, and apply the label 'urgent' to it. Do NOT label non-urgent "
            "emails as urgent."
        ),
        max_steps=20,
        email_count=5,
        seed=42,
    ),
    "medium": TaskConfig(
        name="medium",
        difficulty="medium",
        description=(
            "Sort and Prioritise Inbox — Your inbox contains 10 emails of "
            "mixed types (meeting requests, support tickets, newsletters, "
            "personal messages). Assign a priority to every email: 'urgent', "
            "'normal', or 'low'. Your ranking will be compared against the "
            "gold-standard ordering using Kendall-tau correlation."
        ),
        max_steps=30,
        email_count=10,
        seed=123,
    ),
    "hard": TaskConfig(
        name="hard",
        difficulty="hard",
        description=(
            "Triage, Reply, and Archive — Your inbox contains 15 emails "
            "across 5 categories: urgent, reply-needed, FYI, spam, and "
            "newsletter. You must: (1) Label every email with its correct "
            "category, (2) Reply to all 'reply-needed' emails with a helpful "
            "and professional response, and (3) Archive all spam and newsletter "
            "emails. Scoring: label accuracy 40%, reply quality 40%, "
            "archive correctness 20%."
        ),
        max_steps=40,
        email_count=15,
        seed=256,
        grader_weights={
            "label_accuracy": 0.4,
            "reply_quality": 0.4,
            "archive_correctness": 0.2,
        },
    ),
}


def get_task(name: str) -> TaskConfig:
    """Return the TaskConfig for the given task name.

    Raises:
        KeyError: If the task name is not in the registry.
    """
    if name not in TASK_REGISTRY:
        raise KeyError(
            f"Unknown task '{name}'. Available tasks: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[name]


def list_tasks() -> list[dict]:
    """Return a list of task summaries suitable for the /tasks endpoint."""
    return [
        {
            "name": t.name,
            "difficulty": t.difficulty,
            "description": t.description,
            "max_steps": t.max_steps,
            "email_count": t.email_count,
        }
        for t in TASK_REGISTRY.values()
    ]
