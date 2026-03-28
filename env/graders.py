"""Programmatic graders for all three email triage tasks.

Each grader returns a float in [0.0, 1.0] and a breakdown dict.
Graders are deterministic given the same inputs, except for the LLM-based
reply quality scorer in the hard task (which is called once and cached).
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional

from env.models import Email, EmailReward


def _kendall_tau(x: List[int], y: List[int]) -> float:
    """Compute Kendall-tau correlation coefficient between two rankings.

    Returns a float in [-1.0, 1.0]. Returns 0.0 for degenerate inputs.
    """
    n = len(x)
    if n < 2:
        return 0.0
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            product = dx * dy
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    total_pairs = concordant + discordant
    if total_pairs == 0:
        return 0.0
    tau = (concordant - discordant) / total_pairs
    return tau


# ---------------------------------------------------------------------------
# Easy grader — "Label the Urgent Email"
# ---------------------------------------------------------------------------

def grade_easy(
    emails: List[Email],
    gold_labels: Dict[str, Dict[str, str]],
    action_history: List[str],
    step_count: int,
) -> EmailReward:
    """Grade the easy task: identify and label the single urgent email.

    Scoring:
        +0.1 per unique email opened (up to 0.5)
        +0.5 for labelling the correct email as 'urgent'
        -0.1 for each wrong email labelled 'urgent'
        -0.05 per repeated action (3+ consecutive identical)
    Final score clamped to [0.0, 1.0].
    """
    # Find the gold-urgent email id
    urgent_id: Optional[str] = None
    for eid, meta in gold_labels.items():
        if meta["category"] == "urgent":
            urgent_id = eid
            break

    # Count unique emails opened
    opened_ids: set[str] = set()
    for entry in action_history:
        if entry.startswith("open:"):
            opened_ids.add(entry.split(":", 1)[1])
    open_score = min(len(opened_ids) * 0.1, 0.5)

    # Check labelling
    correct_label = 0.0
    wrong_penalty = 0.0
    for email in emails:
        if "urgent" in email.labels:
            if email.id == urgent_id:
                correct_label = 0.5
            else:
                wrong_penalty += 0.1

    # Loop penalty
    loop_penalty = _compute_loop_penalty(action_history)

    raw = open_score + correct_label - wrong_penalty - loop_penalty
    score = max(0.0, min(1.0, raw))

    breakdown = {
        "emails_opened": len(opened_ids),
        "open_score": round(open_score, 3),
        "correct_label": round(correct_label, 3),
        "wrong_label_penalty": round(wrong_penalty, 3),
        "loop_penalty": round(loop_penalty, 3),
        "raw_score": round(raw, 3),
    }

    feedback_parts: list[str] = []
    feedback_parts.append(f"Opened {len(opened_ids)} emails (+{open_score:.1f}).")
    if correct_label > 0:
        feedback_parts.append("Correctly labelled the urgent email (+0.5).")
    else:
        feedback_parts.append("Did not label the correct email as urgent.")
    if wrong_penalty > 0:
        feedback_parts.append(f"Wrong 'urgent' labels applied (-{wrong_penalty:.1f}).")
    if loop_penalty > 0:
        feedback_parts.append(f"Loop penalty: -{loop_penalty:.2f}.")

    return EmailReward(
        score=round(score, 4),
        partial_score=round(score, 4),
        breakdown=breakdown,
        feedback=" ".join(feedback_parts),
    )


# ---------------------------------------------------------------------------
# Medium grader — "Sort and Prioritise Inbox"
# ---------------------------------------------------------------------------

_PRIORITY_RANK = {"urgent": 3, "normal": 2, "low": 1}


def grade_medium(
    emails: List[Email],
    gold_labels: Dict[str, Dict[str, str]],
    action_history: List[str],
    step_count: int,
) -> EmailReward:
    """Grade the medium task: assign correct priorities to all 10 emails.

    Scoring:
        Kendall-tau between agent priority ranking and gold ranking.
        tau >= 0.7 → 1.0; below that scaled linearly.
        Bonus +0.1 if all emails prioritised before max steps.
    """
    agent_ranks: list[int] = []
    gold_ranks: list[int] = []
    prioritised_count = 0
    total = len(emails)

    for email in emails:
        gold_priority = gold_labels.get(email.id, {}).get("gold_priority", "low")
        gold_rank = _PRIORITY_RANK.get(gold_priority, 1)
        gold_ranks.append(gold_rank)

        if email.priority is not None:
            agent_rank = _PRIORITY_RANK.get(email.priority, 1)
            prioritised_count += 1
        else:
            agent_rank = 0  # not yet assigned
        agent_ranks.append(agent_rank)

    # Compute Kendall-tau only if at least 2 emails are prioritised
    if prioritised_count >= 2:
        tau = _kendall_tau(agent_ranks, gold_ranks)
        if math.isnan(tau):  # NaN guard
            tau = 0.0
    else:
        tau = 0.0

    # Scale: tau >= 0.7 → 1.0, below scaled linearly from 0
    if tau >= 0.7:
        tau_score = 1.0
    elif tau > 0.0:
        tau_score = tau / 0.7
    else:
        tau_score = 0.0

    # Bonus for completing all
    bonus = 0.1 if prioritised_count == total else 0.0

    loop_penalty = _compute_loop_penalty(action_history)
    raw = tau_score + bonus - loop_penalty
    score = max(0.0, min(1.0, raw))

    breakdown = {
        "kendall_tau": round(tau, 4),
        "tau_score": round(tau_score, 4),
        "prioritised_count": prioritised_count,
        "total_emails": total,
        "completion_bonus": round(bonus, 3),
        "loop_penalty": round(loop_penalty, 3),
    }

    feedback = (
        f"Prioritised {prioritised_count}/{total} emails. "
        f"Kendall-tau = {tau:.3f} (score: {tau_score:.3f})."
    )
    if bonus > 0:
        feedback += " Completion bonus +0.1."
    if loop_penalty > 0:
        feedback += f" Loop penalty: -{loop_penalty:.2f}."

    return EmailReward(
        score=round(score, 4),
        partial_score=round(score, 4),
        breakdown=breakdown,
        feedback=feedback,
    )


# ---------------------------------------------------------------------------
# Hard grader — "Triage, Reply, and Archive"
# ---------------------------------------------------------------------------

# Cache for LLM reply quality scores within an episode
_reply_quality_cache: Dict[str, float] = {}


def reset_reply_cache() -> None:
    """Clear the reply quality cache (call on episode reset)."""
    _reply_quality_cache.clear()


def _score_reply_with_llm(
    original_email: Email,
    reply_text: str,
) -> float:
    """Score a reply using an LLM rubric. Returns 0.0–1.0.

    Calls the LLM once per (email_id, reply_text) pair and caches the result.
    Falls back to a heuristic score if the LLM call fails.
    """
    cache_key = f"{original_email.id}::{reply_text}"
    if cache_key in _reply_quality_cache:
        return _reply_quality_cache[cache_key]

    # Attempt LLM-based scoring
    try:
        from openai import OpenAI

        api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        token = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
        model = os.getenv("MODEL_NAME", "")

        if not token or not model:
            raise ValueError("Missing HF_TOKEN or MODEL_NAME for LLM grading")

        client = OpenAI(base_url=api_base, api_key=token)

        prompt = (
            "You are an email reply quality evaluator. Score the following reply "
            "on three criteria (0-10 each):\n"
            "1. Relevance — does the reply address the original email's content?\n"
            "2. Tone — is the reply professional and appropriate?\n"
            "3. Completeness — does the reply answer all questions / cover all points?\n\n"
            f"Original email subject: {original_email.subject}\n"
            f"Original email body:\n{original_email.body}\n\n"
            f"Reply:\n{reply_text}\n\n"
            "Respond ONLY with a JSON object: "
            '{"relevance": <0-10>, "tone": <0-10>, "completeness": <0-10>}'
        )

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )

        content = response.choices[0].message.content or ""
        # Try to extract JSON from the response
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        scores = json.loads(content)
        relevance = float(scores.get("relevance", 5)) / 10.0
        tone = float(scores.get("tone", 5)) / 10.0
        completeness = float(scores.get("completeness", 5)) / 10.0
        quality = (relevance + tone + completeness) / 3.0
        quality = max(0.0, min(1.0, quality))

    except Exception:
        # Heuristic fallback: score based on reply length and basic checks
        quality = _heuristic_reply_score(original_email, reply_text)

    _reply_quality_cache[cache_key] = quality
    return quality


def _heuristic_reply_score(original_email: Email, reply_text: str) -> float:
    """Fallback heuristic for reply quality when LLM is unavailable."""
    if not reply_text or len(reply_text.strip()) < 10:
        return 0.1

    score = 0.3  # base score for having some text

    # Length bonus
    words = reply_text.split()
    if len(words) >= 20:
        score += 0.2
    elif len(words) >= 10:
        score += 0.1

    # Relevance: check if reply references keywords from original
    original_words = set(original_email.subject.lower().split())
    reply_lower = reply_text.lower()
    overlap = sum(1 for w in original_words if w in reply_lower and len(w) > 3)
    if overlap >= 2:
        score += 0.2
    elif overlap >= 1:
        score += 0.1

    # Tone: check for greeting/closing
    if any(g in reply_lower for g in ["hi", "hello", "dear", "thanks", "regards"]):
        score += 0.1

    return max(0.0, min(1.0, score))


def grade_hard(
    emails: List[Email],
    gold_labels: Dict[str, Dict[str, str]],
    action_history: List[str],
    replies: Dict[str, str],
    archived_ids: set[str],
    step_count: int,
) -> EmailReward:
    """Grade the hard task: label, reply, and archive.

    Weighted: 0.4 * label_accuracy + 0.4 * reply_quality + 0.2 * archive_correctness
    """
    total = len(emails)

    # --- Label accuracy (40%) ---
    correct_labels = 0
    labelled_count = 0
    for email in emails:
        gold_cat = gold_labels.get(email.id, {}).get("category", "")
        if email.labels:
            labelled_count += 1
            if gold_cat in email.labels:
                correct_labels += 1
    label_score = correct_labels / total if total > 0 else 0.0

    # --- Reply quality (40%) ---
    reply_needed_ids: list[str] = []
    for email in emails:
        if gold_labels.get(email.id, {}).get("category") == "reply-needed":
            reply_needed_ids.append(email.id)

    if reply_needed_ids:
        reply_scores: list[float] = []
        for eid in reply_needed_ids:
            if eid in replies:
                email_obj = next((e for e in emails if e.id == eid), None)
                if email_obj:
                    rs = _score_reply_with_llm(email_obj, replies[eid])
                    reply_scores.append(rs)
                else:
                    reply_scores.append(0.0)
            else:
                reply_scores.append(0.0)
        reply_score = sum(reply_scores) / len(reply_scores)
    else:
        reply_score = 1.0  # no replies needed

    # --- Archive correctness (20%) ---
    should_archive: set[str] = set()
    should_not_archive: set[str] = set()
    for email in emails:
        cat = gold_labels.get(email.id, {}).get("category", "")
        if cat in ("spam", "newsletter"):
            should_archive.add(email.id)
        else:
            should_not_archive.add(email.id)

    if should_archive:
        correctly_archived = len(archived_ids & should_archive)
        wrongly_archived = len(archived_ids & should_not_archive)
        archive_score = correctly_archived / len(should_archive)
        # Penalise wrong archives
        if wrongly_archived > 0:
            archive_score = max(0.0, archive_score - wrongly_archived * 0.1)
    else:
        archive_score = 1.0

    # Weighted sum
    weights = {"label_accuracy": 0.4, "reply_quality": 0.4, "archive_correctness": 0.2}
    weighted = (
        weights["label_accuracy"] * label_score
        + weights["reply_quality"] * reply_score
        + weights["archive_correctness"] * archive_score
    )

    loop_penalty = _compute_loop_penalty(action_history)
    raw = weighted - loop_penalty
    score = max(0.0, min(1.0, raw))

    breakdown = {
        "label_accuracy": round(label_score, 4),
        "labelled_count": labelled_count,
        "reply_quality": round(reply_score, 4),
        "replies_submitted": len([r for r in reply_needed_ids if r in replies]),
        "replies_needed": len(reply_needed_ids),
        "archive_correctness": round(archive_score, 4),
        "correctly_archived": len(archived_ids & should_archive) if should_archive else 0,
        "should_archive": len(should_archive),
        "loop_penalty": round(loop_penalty, 3),
        "weights": weights,
    }

    feedback = (
        f"Labels: {correct_labels}/{total} correct ({label_score:.2f}). "
        f"Replies: {len([r for r in reply_needed_ids if r in replies])}/{len(reply_needed_ids)} "
        f"submitted (avg quality {reply_score:.2f}). "
        f"Archives: {len(archived_ids & should_archive) if should_archive else 0}/{len(should_archive)} correct."
    )
    if loop_penalty > 0:
        feedback += f" Loop penalty: -{loop_penalty:.2f}."

    return EmailReward(
        score=round(score, 4),
        partial_score=round(score, 4),
        breakdown=breakdown,
        feedback=feedback,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _compute_loop_penalty(action_history: List[str]) -> float:
    """Compute penalty for repeated consecutive actions (3+ repeats).

    Returns a non-negative penalty value.
    """
    if len(action_history) < 3:
        return 0.0

    penalty = 0.0
    consecutive = 1
    for i in range(1, len(action_history)):
        if action_history[i] == action_history[i - 1]:
            consecutive += 1
            if consecutive >= 3:
                penalty += 0.05
        else:
            consecutive = 1

    return penalty
