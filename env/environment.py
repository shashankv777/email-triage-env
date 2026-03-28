"""Core EmailTriageEnv class — implements reset / step / state for all tasks."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from env.data import generate_inbox
from env.graders import grade_easy, grade_hard, grade_medium, reset_reply_cache
from env.models import Email, EmailAction, EmailObservation, EmailReward
from env.tasks import TaskConfig, get_task

_VALID_ACTIONS = {"open", "label", "prioritise", "reply", "archive", "skip", "done"}


class EmailTriageEnv:
    """OpenEnv-compatible email triage environment.

    Manages episode state, validates actions, delegates grading, and enforces
    step limits and loop penalties.
    """

    def __init__(self) -> None:
        """Initialise with empty state — call reset() to start an episode."""
        self._task: Optional[TaskConfig] = None
        self._emails: List[Email] = []
        self._gold_labels: Dict[str, Dict[str, str]] = {}
        self._current_email: Optional[Email] = None
        self._action_history: List[str] = []
        self._step_count: int = 0
        self._done: bool = True
        self._replies: Dict[str, str] = {}
        self._archived_ids: set[str] = set()
        self._opened_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_name: str = "easy") -> EmailObservation:
        """Start a fresh episode for the given task.

        Args:
            task_name: One of 'easy', 'medium', 'hard'.

        Returns:
            The initial observation.
        """
        self._task = get_task(task_name)
        emails, metadata = generate_inbox(
            task_difficulty=self._task.difficulty,
            seed=self._task.seed,
        )
        self._emails = emails
        self._gold_labels = metadata["gold_labels"]
        self._current_email = None
        self._action_history = []
        self._step_count = 0
        self._done = False
        self._replies = {}
        self._archived_ids = set()
        self._opened_ids = set()
        reset_reply_cache()

        return self._observation()

    def step(
        self, action: EmailAction
    ) -> Tuple[EmailObservation, EmailReward, bool, Dict[str, Any]]:
        """Execute one action and return (observation, reward, done, info).

        Args:
            action: The agent's action.

        Returns:
            A 4-tuple of (observation, reward, done, info dict).

        Raises:
            RuntimeError: If the environment has not been reset.
        """
        if self._task is None or self._done:
            raise RuntimeError(
                "Environment is not active. Call reset() before step()."
            )

        self._step_count += 1
        info: Dict[str, Any] = {}

        # Validate action type
        if action.action_type not in _VALID_ACTIONS:
            info["error"] = f"Invalid action_type '{action.action_type}'"
            return self._observation(), self._grade(), self._done, info

        # Record action for history / loop detection
        action_key = self._action_key(action)
        self._action_history.append(action_key)

        # Dispatch
        match action.action_type:
            case "open":
                info.update(self._handle_open(action))
            case "label":
                info.update(self._handle_label(action))
            case "prioritise":
                info.update(self._handle_prioritise(action))
            case "reply":
                info.update(self._handle_reply(action))
            case "archive":
                info.update(self._handle_archive(action))
            case "skip":
                info["message"] = "Skipped."
            case "done":
                info.update(self._handle_done())

        # Check max steps
        if self._step_count >= self._task.max_steps and not self._done:
            self._done = True
            info["terminated_reason"] = "max_steps_reached"

        reward = self._grade()
        obs = self._observation()
        return obs, reward, self._done, info

    def state(self) -> Dict[str, Any]:
        """Return a full serialisable snapshot of the current state."""
        return {
            "task": self._task.name if self._task else None,
            "step_count": self._step_count,
            "done": self._done,
            "emails": [e.model_dump() for e in self._emails],
            "current_email": self._current_email.model_dump() if self._current_email else None,
            "action_history": list(self._action_history),
            "replies": dict(self._replies),
            "archived_ids": list(self._archived_ids),
            "opened_ids": list(self._opened_ids),
            "gold_labels": self._gold_labels,
        }

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_open(self, action: EmailAction) -> Dict[str, Any]:
        """Open an email by id."""
        email = self._find_email(action.email_id)
        if email is None:
            return {"error": f"Email '{action.email_id}' not found."}
        email.is_read = True
        self._current_email = email
        self._opened_ids.add(email.id)
        return {"message": f"Opened email '{email.id}': {email.subject}"}

    def _handle_label(self, action: EmailAction) -> Dict[str, Any]:
        """Apply a label to an email."""
        email = self._find_email(action.email_id)
        if email is None:
            return {"error": f"Email '{action.email_id}' not found."}
        if not action.label:
            return {"error": "Label action requires a 'label' field."}
        if action.label not in email.labels:
            email.labels.append(action.label)
        return {"message": f"Labelled email '{email.id}' with '{action.label}'."}

    def _handle_prioritise(self, action: EmailAction) -> Dict[str, Any]:
        """Assign a priority to an email."""
        email = self._find_email(action.email_id)
        if email is None:
            return {"error": f"Email '{action.email_id}' not found."}
        if action.priority not in ("urgent", "normal", "low"):
            return {"error": f"Invalid priority '{action.priority}'. Use urgent/normal/low."}
        email.priority = action.priority
        return {"message": f"Set priority of email '{email.id}' to '{action.priority}'."}

    def _handle_reply(self, action: EmailAction) -> Dict[str, Any]:
        """Submit a reply to an email."""
        email = self._find_email(action.email_id)
        if email is None:
            return {"error": f"Email '{action.email_id}' not found."}
        if not action.reply_text or len(action.reply_text.strip()) == 0:
            return {"error": "Reply action requires non-empty 'reply_text'."}
        self._replies[email.id] = action.reply_text
        return {"message": f"Reply sent to email '{email.id}'."}

    def _handle_archive(self, action: EmailAction) -> Dict[str, Any]:
        """Archive an email."""
        email = self._find_email(action.email_id)
        if email is None:
            return {"error": f"Email '{action.email_id}' not found."}
        self._archived_ids.add(email.id)
        return {"message": f"Archived email '{email.id}'."}

    def _handle_done(self) -> Dict[str, Any]:
        """Agent signals it is finished."""
        # Penalise early done: check minimum progress
        task_name = self._task.name if self._task else "easy"
        min_progress = self._check_minimum_progress(task_name)
        if not min_progress:
            self._done = True
            return {"message": "Done called early — minimum progress not met."}
        self._done = True
        return {"message": "Episode complete."}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_email(self, email_id: Optional[str]) -> Optional[Email]:
        """Look up an email by id."""
        if not email_id:
            return None
        for email in self._emails:
            if email.id == email_id:
                return email
        return None

    def _action_key(self, action: EmailAction) -> str:
        """Create a string key for loop detection."""
        parts = [action.action_type]
        if action.email_id:
            parts.append(action.email_id)
        if action.label:
            parts.append(action.label)
        if action.priority:
            parts.append(action.priority)
        return ":".join(parts)

    def _check_minimum_progress(self, task_name: str) -> bool:
        """Check whether the agent has made minimum progress before calling done."""
        match task_name:
            case "easy":
                return len(self._opened_ids) >= 1
            case "medium":
                prioritised = sum(1 for e in self._emails if e.priority is not None)
                return prioritised >= 3
            case "hard":
                labelled = sum(1 for e in self._emails if len(e.labels) > 0)
                return labelled >= 3
            case _:
                return True

    def _observation(self) -> EmailObservation:
        """Build the current observation."""
        return EmailObservation(
            inbox=deepcopy(self._emails),
            current_email=deepcopy(self._current_email),
            action_history=list(self._action_history),
            step_count=self._step_count,
            task_description=self._task.description if self._task else "",
            done=self._done,
        )

    def _grade(self) -> EmailReward:
        """Delegate grading to the appropriate task grader."""
        if self._task is None:
            return EmailReward(score=0.0, partial_score=0.0, breakdown={}, feedback="No task loaded.")

        match self._task.name:
            case "easy":
                return grade_easy(
                    emails=self._emails,
                    gold_labels=self._gold_labels,
                    action_history=self._action_history,
                    step_count=self._step_count,
                )
            case "medium":
                return grade_medium(
                    emails=self._emails,
                    gold_labels=self._gold_labels,
                    action_history=self._action_history,
                    step_count=self._step_count,
                )
            case "hard":
                return grade_hard(
                    emails=self._emails,
                    gold_labels=self._gold_labels,
                    action_history=self._action_history,
                    replies=self._replies,
                    archived_ids=self._archived_ids,
                    step_count=self._step_count,
                )
            case _:
                return EmailReward(
                    score=0.0,
                    partial_score=0.0,
                    breakdown={},
                    feedback=f"Unknown task '{self._task.name}'.",
                )
