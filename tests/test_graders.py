"""Tests for grader determinism and range — all graders must return [0.0, 1.0]."""

from __future__ import annotations

import pytest

from env.data import generate_inbox
from env.graders import grade_easy, grade_hard, grade_medium, reset_reply_cache
from env.models import Email, EmailReward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_easy_data() -> tuple[list[Email], dict]:
    emails, meta = generate_inbox("easy", seed=42)
    return emails, meta["gold_labels"]


def _get_medium_data() -> tuple[list[Email], dict]:
    emails, meta = generate_inbox("medium", seed=123)
    return emails, meta["gold_labels"]


def _get_hard_data() -> tuple[list[Email], dict]:
    emails, meta = generate_inbox("hard", seed=256)
    return emails, meta["gold_labels"]


def _assert_valid_reward(reward: EmailReward) -> None:
    assert isinstance(reward, EmailReward)
    assert 0.0 <= reward.score <= 1.0, f"score out of range: {reward.score}"
    assert 0.0 <= reward.partial_score <= 1.0, f"partial_score out of range: {reward.partial_score}"
    assert isinstance(reward.breakdown, dict)
    assert isinstance(reward.feedback, str)


# ---------------------------------------------------------------------------
# Easy grader tests
# ---------------------------------------------------------------------------

class TestEasyGrader:
    def test_no_actions_score_zero(self) -> None:
        emails, gold = _get_easy_data()
        reward = grade_easy(emails, gold, [], 0)
        _assert_valid_reward(reward)
        assert reward.score == 0.0

    def test_open_all_gives_partial_credit(self) -> None:
        emails, gold = _get_easy_data()
        history = [f"open:{e.id}" for e in emails]
        for e in emails:
            e.is_read = True
        reward = grade_easy(emails, gold, history, len(history))
        _assert_valid_reward(reward)
        assert reward.score >= 0.4  # 5 * 0.1 = 0.5

    def test_correct_label_gives_full_score(self) -> None:
        emails, gold = _get_easy_data()
        # Find the urgent email
        urgent_id = None
        for eid, meta in gold.items():
            if meta["category"] == "urgent":
                urgent_id = eid
                break
        # Open all and label the correct one
        history = [f"open:{e.id}" for e in emails]
        for e in emails:
            e.is_read = True
            if e.id == urgent_id:
                e.labels.append("urgent")
        history.append(f"label:{urgent_id}:urgent")
        reward = grade_easy(emails, gold, history, len(history))
        _assert_valid_reward(reward)
        assert reward.score == 1.0

    def test_wrong_label_penalised(self) -> None:
        emails, gold = _get_easy_data()
        # Label a non-urgent email as urgent
        urgent_id = None
        for eid, meta in gold.items():
            if meta["category"] == "urgent":
                urgent_id = eid
                break
        for e in emails:
            if e.id != urgent_id:
                e.labels.append("urgent")
                break
        reward = grade_easy(emails, gold, [], 1)
        _assert_valid_reward(reward)
        # Should be 0.0 because no opens and wrong label penalty
        assert reward.score == 0.0

    def test_deterministic(self) -> None:
        emails1, gold1 = _get_easy_data()
        emails2, gold2 = _get_easy_data()
        r1 = grade_easy(emails1, gold1, ["open:email-001"], 1)
        r2 = grade_easy(emails2, gold2, ["open:email-001"], 1)
        assert r1.score == r2.score
        assert r1.breakdown == r2.breakdown

    def test_loop_penalty(self) -> None:
        emails, gold = _get_easy_data()
        # Same action repeated 5 times
        history = ["skip"] * 5
        reward = grade_easy(emails, gold, history, 5)
        _assert_valid_reward(reward)
        assert reward.breakdown.get("loop_penalty", 0) > 0


# ---------------------------------------------------------------------------
# Medium grader tests
# ---------------------------------------------------------------------------

class TestMediumGrader:
    def test_no_priorities_score_zero(self) -> None:
        emails, gold = _get_medium_data()
        reward = grade_medium(emails, gold, [], 0)
        _assert_valid_reward(reward)
        assert reward.score == 0.0

    def test_perfect_priorities(self) -> None:
        emails, gold = _get_medium_data()
        for e in emails:
            e.priority = gold[e.id]["gold_priority"]
        history = [f"prioritise:{e.id}:{e.priority}" for e in emails]
        reward = grade_medium(emails, gold, history, len(history))
        _assert_valid_reward(reward)
        assert reward.score >= 0.9  # Should be 1.0 with bonus

    def test_partial_priorities(self) -> None:
        emails, gold = _get_medium_data()
        # Assign correct priority to only half
        for i, e in enumerate(emails):
            if i < 5:
                e.priority = gold[e.id]["gold_priority"]
        reward = grade_medium(emails, gold, [], 5)
        _assert_valid_reward(reward)
        assert 0.0 < reward.score < 1.0

    def test_deterministic(self) -> None:
        emails1, gold1 = _get_medium_data()
        emails2, gold2 = _get_medium_data()
        for e in emails1:
            e.priority = "normal"
        for e in emails2:
            e.priority = "normal"
        r1 = grade_medium(emails1, gold1, [], 10)
        r2 = grade_medium(emails2, gold2, [], 10)
        assert r1.score == r2.score


# ---------------------------------------------------------------------------
# Hard grader tests
# ---------------------------------------------------------------------------

class TestHardGrader:
    def test_no_actions_score_zero(self) -> None:
        reset_reply_cache()
        emails, gold = _get_hard_data()
        reward = grade_hard(emails, gold, [], {}, set(), 0)
        _assert_valid_reward(reward)
        assert reward.score == 0.0

    def test_perfect_labels_partial_score(self) -> None:
        reset_reply_cache()
        emails, gold = _get_hard_data()
        for e in emails:
            cat = gold[e.id]["category"]
            e.labels.append(cat)
        reward = grade_hard(emails, gold, [], {}, set(), 15)
        _assert_valid_reward(reward)
        # Label accuracy = 1.0, reply = 0.0, archive = 0.0
        # 0.4 * 1.0 + 0.4 * 0.0 + 0.2 * 0.0 = 0.4
        assert reward.score >= 0.35

    def test_perfect_archive(self) -> None:
        reset_reply_cache()
        emails, gold = _get_hard_data()
        archived = set()
        for e in emails:
            if gold[e.id]["category"] in ("spam", "newsletter"):
                archived.add(e.id)
        reward = grade_hard(emails, gold, [], {}, archived, 10)
        _assert_valid_reward(reward)
        assert reward.breakdown.get("archive_correctness", 0) > 0.9

    def test_score_range(self) -> None:
        reset_reply_cache()
        emails, gold = _get_hard_data()
        reward = grade_hard(emails, gold, ["skip"] * 5, {}, set(), 5)
        _assert_valid_reward(reward)

    def test_deterministic(self) -> None:
        reset_reply_cache()
        emails1, gold1 = _get_hard_data()
        emails2, gold2 = _get_hard_data()
        r1 = grade_hard(emails1, gold1, [], {}, set(), 0)
        reset_reply_cache()
        r2 = grade_hard(emails2, gold2, [], {}, set(), 0)
        assert r1.score == r2.score
