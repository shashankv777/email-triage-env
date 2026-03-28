"""Root-level models for OpenEnv compatibility.

Re-exports the models from env.models for the openenv push command.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import List, Optional


class EmailAction(Action):
    """Action for the Email Triage environment."""

    action_type: str = Field(
        ...,
        description="Type of action: 'open', 'label', 'prioritise', 'reply', 'archive', 'skip', 'done'",
    )
    email_id: Optional[str] = Field(default=None, description="ID of the email to act on")
    label: Optional[str] = Field(default=None, description="Label to apply (urgent, reply-needed, fyi, spam, newsletter)")
    priority: Optional[str] = Field(default=None, description="Priority level (urgent, normal, low)")
    reply_text: Optional[str] = Field(default=None, description="Reply text for reply action")


class EmailObservation(Observation):
    """Observation from the Email Triage environment."""

    inbox: List[dict] = Field(default_factory=list, description="List of emails in the inbox")
    current_email: Optional[dict] = Field(default=None, description="Currently open email")
    task_description: str = Field(default="", description="Description of the current task")
    step_count: int = Field(default=0, description="Current step number")
    max_steps: int = Field(default=20, description="Maximum steps allowed")
    done: bool = Field(default=False, description="Whether the episode is done")
