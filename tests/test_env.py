"""Unit tests for the EmailTriageEnv — reset, step, and state."""

from __future__ import annotations

import pytest

from env.environment import EmailTriageEnv
from env.models import EmailAction, EmailObservation, EmailReward


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def env() -> EmailTriageEnv:
    """Return a fresh environment instance."""
    return EmailTriageEnv()


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_easy_returns_observation(self, env: EmailTriageEnv) -> None:
        obs = env.reset("easy")
        assert isinstance(obs, EmailObservation)
        assert len(obs.inbox) == 5
        assert obs.step_count == 0
        assert obs.done is False
        assert obs.current_email is None
        assert obs.action_history == []

    def test_reset_medium_returns_10_emails(self, env: EmailTriageEnv) -> None:
        obs = env.reset("medium")
        assert len(obs.inbox) == 10

    def test_reset_hard_returns_15_emails(self, env: EmailTriageEnv) -> None:
        obs = env.reset("hard")
        assert len(obs.inbox) == 15

    def test_reset_unknown_task_raises(self, env: EmailTriageEnv) -> None:
        with pytest.raises(KeyError):
            env.reset("nonexistent")

    def test_reset_clears_previous_state(self, env: EmailTriageEnv) -> None:
        env.reset("easy")
        env.step(EmailAction(action_type="skip"))
        obs = env.reset("easy")
        assert obs.step_count == 0
        assert obs.action_history == []

    def test_reset_is_reproducible(self, env: EmailTriageEnv) -> None:
        obs1 = env.reset("easy")
        obs2 = env.reset("easy")
        ids1 = [e.id for e in obs1.inbox]
        ids2 = [e.id for e in obs2.inbox]
        assert ids1 == ids2


# ---------------------------------------------------------------------------
# Step tests
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_without_reset_raises(self, env: EmailTriageEnv) -> None:
        with pytest.raises(RuntimeError):
            env.step(EmailAction(action_type="skip"))

    def test_skip_action(self, env: EmailTriageEnv) -> None:
        env.reset("easy")
        obs, reward, done, info = env.step(EmailAction(action_type="skip"))
        assert isinstance(obs, EmailObservation)
        assert isinstance(reward, EmailReward)
        assert obs.step_count == 1
        assert done is False

    def test_open_action_sets_current_email(self, env: EmailTriageEnv) -> None:
        obs = env.reset("easy")
        eid = obs.inbox[0].id
        obs2, _, _, _ = env.step(EmailAction(action_type="open", email_id=eid))
        assert obs2.current_email is not None
        assert obs2.current_email.id == eid
        assert obs2.current_email.is_read is True

    def test_open_invalid_email_id(self, env: EmailTriageEnv) -> None:
        env.reset("easy")
        _, _, _, info = env.step(EmailAction(action_type="open", email_id="bogus"))
        assert "error" in info

    def test_label_action(self, env: EmailTriageEnv) -> None:
        obs = env.reset("easy")
        eid = obs.inbox[0].id
        obs2, _, _, _ = env.step(
            EmailAction(action_type="label", email_id=eid, label="urgent")
        )
        labelled = [e for e in obs2.inbox if e.id == eid][0]
        assert "urgent" in labelled.labels

    def test_prioritise_action(self, env: EmailTriageEnv) -> None:
        obs = env.reset("medium")
        eid = obs.inbox[0].id
        obs2, _, _, _ = env.step(
            EmailAction(action_type="prioritise", email_id=eid, priority="normal")
        )
        updated = [e for e in obs2.inbox if e.id == eid][0]
        assert updated.priority == "normal"

    def test_reply_action(self, env: EmailTriageEnv) -> None:
        obs = env.reset("hard")
        eid = obs.inbox[0].id
        env.step(EmailAction(action_type="open", email_id=eid))
        _, _, _, info = env.step(
            EmailAction(action_type="reply", email_id=eid, reply_text="Thanks for your message.")
        )
        assert "error" not in info

    def test_archive_action(self, env: EmailTriageEnv) -> None:
        obs = env.reset("hard")
        eid = obs.inbox[0].id
        _, _, _, info = env.step(
            EmailAction(action_type="archive", email_id=eid)
        )
        assert "error" not in info

    def test_done_action_ends_episode(self, env: EmailTriageEnv) -> None:
        obs = env.reset("easy")
        # Open at least one email to meet minimum progress
        eid = obs.inbox[0].id
        env.step(EmailAction(action_type="open", email_id=eid))
        _, _, done, _ = env.step(EmailAction(action_type="done"))
        assert done is True

    def test_max_steps_terminates(self, env: EmailTriageEnv) -> None:
        env.reset("easy")
        # Easy max steps = 20
        for _ in range(20):
            obs, _, done, _ = env.step(EmailAction(action_type="skip"))
            if done:
                break
        assert done is True

    def test_invalid_action_type(self, env: EmailTriageEnv) -> None:
        env.reset("easy")
        _, _, _, info = env.step(EmailAction(action_type="fly"))
        assert "error" in info


# ---------------------------------------------------------------------------
# State tests
# ---------------------------------------------------------------------------

class TestState:
    def test_state_after_reset(self, env: EmailTriageEnv) -> None:
        env.reset("easy")
        s = env.state()
        assert s["task"] == "easy"
        assert s["step_count"] == 0
        assert s["done"] is False
        assert len(s["emails"]) == 5

    def test_state_reflects_actions(self, env: EmailTriageEnv) -> None:
        obs = env.reset("easy")
        eid = obs.inbox[0].id
        env.step(EmailAction(action_type="open", email_id=eid))
        s = env.state()
        assert s["step_count"] == 1
        assert eid in s["opened_ids"]
        assert len(s["action_history"]) == 1

    def test_state_contains_gold_labels(self, env: EmailTriageEnv) -> None:
        env.reset("easy")
        s = env.state()
        assert "gold_labels" in s
        assert len(s["gold_labels"]) == 5
