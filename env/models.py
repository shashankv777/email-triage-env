"""Pydantic v2 models for the Email Triage OpenEnv environment."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Email(BaseModel):
    """A single email message in the inbox."""

    model_config = {"frozen": False}

    id: str = Field(..., description="Unique email identifier")
    sender: str = Field(..., description="Sender email address")
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Full email body text")
    timestamp: str = Field(..., description="ISO-8601 timestamp")
    labels: List[str] = Field(default_factory=list, description="Mutable labels")
    is_read: bool = Field(default=False, description="Whether the email has been read")
    priority: Optional[str] = Field(
        default=None,
        description='Priority level: "urgent" | "normal" | "low" | None',
    )


class EmailObservation(BaseModel):
    """Observation returned by the environment on reset / step."""

    model_config = {"frozen": False}

    inbox: List[Email] = Field(..., description="List of emails in the inbox")
    current_email: Optional[Email] = Field(
        default=None, description="Currently open email, if any"
    )
    action_history: List[str] = Field(
        default_factory=list, description="Past actions taken this episode"
    )
    step_count: int = Field(default=0, description="Number of steps taken so far")
    task_description: str = Field(
        ..., description="Natural-language description of the task"
    )
    done: bool = Field(default=False, description="Whether the episode has ended")


class EmailAction(BaseModel):
    """Action submitted by the agent."""

    action_type: str = Field(
        ...,
        description='One of: "open", "label", "prioritise", "reply", "archive", "skip", "done"',
    )
    email_id: Optional[str] = Field(
        default=None, description="Target email ID (required for most actions)"
    )
    label: Optional[str] = Field(
        default=None, description="Label to apply (for label action)"
    )
    priority: Optional[str] = Field(
        default=None,
        description='Priority to assign: "urgent" | "normal" | "low" (for prioritise action)',
    )
    reply_text: Optional[str] = Field(
        default=None, description="Reply body text (for reply action)"
    )


class EmailReward(BaseModel):
    """Reward signal returned after each step."""

    score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall score for the episode so far"
    )
    partial_score: float = Field(
        ..., ge=0.0, le=1.0, description="Progress score so far"
    )
    breakdown: dict = Field(
        default_factory=dict, description="Per-criterion score breakdown"
    )
    feedback: str = Field(
        default="", description="Human-readable explanation of the reward"
    )
