"""Synthetic email dataset generator for the Email Triage environment.

Provides a pool of 30+ realistic email templates and functions to sample
reproducible inboxes for each task difficulty level.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import List

from env.models import Email

# ---------------------------------------------------------------------------
# Email template pool — 35 unique templates
# Each tuple: (sender, subject, body, category, gold_priority)
#   category in {"urgent", "reply-needed", "fyi", "spam", "newsletter"}
#   gold_priority in {"urgent", "normal", "low"}
# ---------------------------------------------------------------------------

EMAIL_TEMPLATES: list[tuple[str, str, str, str, str]] = [
    # ── urgent (financial / legal / time-critical) ──────────────────────
    (
        "payments@acmefinancials.io",
        "URGENT: Wire transfer requires your approval within 2 hours",
        (
            "Hi,\n\nA wire transfer of $47,500 to vendor Globex Ltd is pending your "
            "approval. Per compliance policy the transfer will be cancelled if not "
            "approved by 3 PM EST today. Please log into the treasury portal and "
            "confirm.\n\nRegards,\nTreasury Operations"
        ),
        "urgent",
        "urgent",
    ),
    (
        "legal@partnerlaw.co",
        "Action Required: NDA execution deadline tomorrow",
        (
            "Dear Counsel,\n\nThe mutual NDA between your organisation and Initech "
            "Corp must be counter-signed by end of business tomorrow. Failure to "
            "execute will delay the due-diligence phase of the acquisition.\n\n"
            "Please review the attached redline and confirm acceptance.\n\nBest,\n"
            "Sarah Lin, Partner"
        ),
        "urgent",
        "urgent",
    ),
    (
        "alerts@cloudmonitor.dev",
        "CRITICAL: Production database CPU at 98%",
        (
            "Alert triggered at 14:32 UTC.\n\nService: primary-postgres\n"
            "Metric: CPU utilisation 98.2% (threshold 85%)\n"
            "Duration: 12 minutes\n\nImmediate investigation recommended. "
            "Auto-scaling has been attempted but reached instance limits.\n\n"
            "— CloudMonitor Alerts"
        ),
        "urgent",
        "urgent",
    ),
    (
        "cfo@northstarventures.biz",
        "Board meeting moved to TODAY 4 PM — revised agenda attached",
        (
            "Team,\n\nDue to scheduling conflicts the quarterly board meeting has "
            "been moved to today at 4 PM in Conference Room A. Please review the "
            "revised agenda and have your departmental reports ready.\n\n"
            "Thanks,\nMichael Torres, CFO"
        ),
        "urgent",
        "urgent",
    ),
    (
        "security@idshieldpro.net",
        "Suspicious login detected on your account",
        (
            "We detected a login to your account from an unrecognised device in "
            "São Paulo, Brazil at 03:17 UTC. If this was not you, please reset "
            "your password immediately and enable two-factor authentication.\n\n"
            "— IDShield Security Team"
        ),
        "urgent",
        "urgent",
    ),
    # ── reply-needed (colleague/client asking a question) ──────────────
    (
        "j.kumar@teamsync.io",
        "Quick question about the Q3 roadmap",
        (
            "Hey,\n\nI'm putting together the Q3 planning deck and wanted to "
            "confirm whether the data-pipeline migration is still targeted for "
            "July or if it's slipping to August. Can you let me know by Thursday?\n\n"
            "Cheers,\nJai"
        ),
        "reply-needed",
        "normal",
    ),
    (
        "m.chen@clientcorp.com",
        "Re: Proposal feedback",
        (
            "Hi,\n\nThanks for sending over the proposal. We have a few questions:\n"
            "1. Can the implementation timeline be shortened to 8 weeks?\n"
            "2. Is 24/7 support included in Tier 2?\n"
            "3. What are the penalties for SLA breaches?\n\n"
            "Looking forward to your answers.\n\nBest,\nMei Chen"
        ),
        "reply-needed",
        "normal",
    ),
    (
        "r.okonkwo@designhub.co",
        "Need your input on homepage mockups",
        (
            "Hi,\n\nI've uploaded three homepage variants to the shared drive. "
            "Could you review them and let me know which direction you prefer? "
            "We need to lock the design by Friday to stay on schedule.\n\n"
            "Thanks!\nRita"
        ),
        "reply-needed",
        "normal",
    ),
    (
        "a.petrov@vendorsupply.com",
        "Invoice #8842 — clarification needed",
        (
            "Hello,\n\nInvoice #8842 lists 150 units of SKU-X200 at $12.50/unit "
            "but our PO was for 120 units at $11.75. Could you confirm the correct "
            "figures so we can process payment?\n\nRegards,\nAlexei Petrov"
        ),
        "reply-needed",
        "normal",
    ),
    (
        "hr@internalcomms.org",
        "Please confirm your emergency contact details",
        (
            "Hi,\n\nAs part of our annual records update, please reply with your "
            "current emergency contact name, phone number, and relationship. "
            "Deadline: end of this week.\n\nThank you,\nHR Department"
        ),
        "reply-needed",
        "low",
    ),
    (
        "t.williams@partnerfirm.co",
        "Follow-up: integration testing schedule",
        (
            "Hi,\n\nJust circling back on the integration testing window. Are we "
            "still looking at the second week of next month? Our QA team needs at "
            "least two weeks' notice to allocate resources.\n\nThanks,\nTyler"
        ),
        "reply-needed",
        "normal",
    ),
    (
        "s.garcia@salesdept.io",
        "Can you join the client call on Wednesday?",
        (
            "Hey,\n\nWe have a discovery call with Pinnacle Corp on Wednesday at "
            "11 AM. They specifically asked for a technical walkthrough. Could you "
            "join for 30 minutes to cover the architecture section?\n\nLet me know.\n"
            "Sofia"
        ),
        "reply-needed",
        "normal",
    ),
    # ── fyi (informational, no action needed) ──────────────────────────
    (
        "updates@projecttracker.app",
        "Sprint 14 retrospective summary",
        (
            "Team,\n\nHere's the retro summary:\n- Went well: deployment automation, "
            "test coverage improvements\n- Improve: cross-team communication, "
            "documentation updates\n- Action items assigned in Jira.\n\nNo reply "
            "needed — just FYI.\n\n— PM Bot"
        ),
        "fyi",
        "low",
    ),
    (
        "ceo@northstarventures.biz",
        "Company all-hands recording available",
        (
            "Hi everyone,\n\nThe recording of yesterday's all-hands is now available "
            "on the intranet under Videos > All-Hands > Q2-2025. Key highlights: "
            "revenue up 18% QoQ, new office opening in Austin, and updated PTO "
            "policy effective July 1.\n\nBest,\nDana Park, CEO"
        ),
        "fyi",
        "low",
    ),
    (
        "devops@internaltools.dev",
        "Scheduled maintenance: CI/CD pipeline — Saturday 2 AM",
        (
            "Hi team,\n\nRoutine maintenance on the CI/CD pipeline is scheduled for "
            "Saturday 2–4 AM UTC. Expect intermittent build failures during that "
            "window. No action needed on your part.\n\n— DevOps"
        ),
        "fyi",
        "low",
    ),
    (
        "analytics@dashboardhq.io",
        "Weekly metrics digest — W23",
        (
            "Your weekly digest:\n\n- DAU: 34,200 (+5.1%)\n- Conversion rate: "
            "3.8% (flat)\n- Churn: 1.2% (-0.3%)\n- NPS: 62 (+4)\n\n"
            "Full dashboard: https://dashboardhq.io/w23\n\n— Analytics Bot"
        ),
        "fyi",
        "low",
    ),
    (
        "facilities@officemgmt.com",
        "Kitchen renovation complete — 3rd floor",
        (
            "Good news! The 3rd-floor kitchen renovation is complete. New espresso "
            "machine, filtered water tap, and additional fridge space are now "
            "available. Enjoy!\n\n— Facilities Management"
        ),
        "fyi",
        "low",
    ),
    # ── spam ───────────────────────────────────────────────────────────
    (
        "winner@lotteryprize.xyz",
        "Congratulations! You've won $5,000,000!!!",
        (
            "Dear Lucky Winner,\n\nYou have been selected as the grand prize winner "
            "of the International Email Lottery! To claim your $5,000,000 prize, "
            "simply reply with your full name, bank account number, and routing "
            "number.\n\nAct NOW before your prize expires!\n\n— Lottery Claims Dept"
        ),
        "spam",
        "low",
    ),
    (
        "deals@cheapmeds-online.xyz",
        "80% OFF — Limited time pharmacy deals!",
        (
            "Get prescription medications at 80% off retail prices! No prescription "
            "needed! Fast worldwide shipping! Click here: http://totallylegit.xyz/"
            "meds\n\nUnsubscribe? Never!"
        ),
        "spam",
        "low",
    ),
    (
        "prince@royaltreasury.ng",
        "Confidential Business Proposal",
        (
            "Dear Friend,\n\nI am Prince Emeka, son of the late King of a small "
            "but wealthy nation. I need your help to transfer $12.7 million USD. "
            "You will receive 30% for your assistance. Please reply urgently.\n\n"
            "Yours faithfully,\nPrince Emeka"
        ),
        "spam",
        "low",
    ),
    (
        "support@amaz0n-verify.co",
        "Your account has been locked — verify immediately",
        (
            "Dear Customer,\n\nWe have detected unusual activity on your account. "
            "Your account has been temporarily locked. Click the link below to "
            "verify your identity: http://amaz0n-verify.co/login\n\n"
            "If you do not verify within 24 hours your account will be deleted."
        ),
        "spam",
        "low",
    ),
    (
        "noreply@crypto-moonshot.io",
        "🚀 This coin is about to 1000x — get in NOW",
        (
            "Insider tip: $MOONCOIN is about to explode. Our analysts predict a "
            "1000x return in the next 48 hours. Don't miss out — buy now at "
            "http://crypto-moonshot.io/buy\n\nDisclaimer: not financial advice lol"
        ),
        "spam",
        "low",
    ),
    # ── newsletter ────────────────────────────────────────────────────
    (
        "digest@techweekly.io",
        "Tech Weekly #142: AI agents, Rust in production, and more",
        (
            "This week in tech:\n\n1. OpenAI announces GPT-5 turbo\n2. Rust adoption "
            "hits all-time high in backend services\n3. EU passes new AI regulation "
            "framework\n4. Interview: Building reliable multi-agent systems\n\n"
            "Read more at techweekly.io\n\nUnsubscribe: techweekly.io/unsub"
        ),
        "newsletter",
        "low",
    ),
    (
        "news@startupdaily.com",
        "Startup Daily: $200M Series C for DevTools startup",
        (
            "Top stories:\n\n• CodeForge raises $200M Series C at $2B valuation\n"
            "• Remote-work platform LayerDesk acquires ChatBot.ai\n"
            "• 5 tips for surviving a down round\n\n"
            "Read the full newsletter: startupdaily.com/issue/487"
        ),
        "newsletter",
        "low",
    ),
    (
        "weekly@pythondigest.org",
        "Python Digest #301 — New PEPs, FastAPI 1.0, and tutorials",
        (
            "This week:\n- PEP 750: Template Strings accepted\n"
            "- FastAPI 1.0 released with major performance improvements\n"
            "- Tutorial: Building async pipelines with asyncio\n"
            "- Library spotlight: Polars 0.20\n\n"
            "Read online: pythondigest.org/301"
        ),
        "newsletter",
        "low",
    ),
    (
        "curated@designinspo.co",
        "Design Inspiration Weekly — Issue 88",
        (
            "This week's curated picks:\n\n1. Minimalist dashboard UI by @artisan\n"
            "2. Bold typography trends for 2025\n3. Case study: Redesigning a "
            "banking app\n4. Free icon pack: 500+ line icons\n\n"
            "View gallery: designinspo.co/88"
        ),
        "newsletter",
        "low",
    ),
    (
        "updates@cloudblog.dev",
        "Cloud Blog: Serverless best practices 2025",
        (
            "New on the blog:\n\n- Serverless anti-patterns to avoid\n"
            "- Cost optimisation for Lambda at scale\n- Terraform vs Pulumi: "
            "2025 comparison\n\nRead: cloudblog.dev/serverless-2025\n\nUnsubscribe: "
            "cloudblog.dev/unsub"
        ),
        "newsletter",
        "low",
    ),
    # ── ambiguous emails (look urgent in subject but are not) ──────────
    (
        "events@conferencehub.com",
        "URGENT: Early-bird registration closing soon!",
        (
            "Don't miss out! Early-bird pricing for DevCon 2025 ends this Friday. "
            "Save $200 on your conference pass. Regular pricing applies after that.\n\n"
            "Register: conferencehub.com/devcon2025\n\nThis is a marketing email."
        ),
        "newsletter",
        "low",
    ),
    (
        "promo@saastools.io",
        "URGENT — Your free trial expires in 3 days",
        (
            "Hi,\n\nYour 14-day free trial of SaaSTools Pro is expiring in 3 days. "
            "Upgrade now to keep your data and unlock premium features.\n\n"
            "Upgrade: saastools.io/upgrade\n\nNo hard feelings if you don't — "
            "your data will be retained for 30 days after expiry."
        ),
        "spam",
        "low",
    ),
    # ── more reply-needed to fill hard task ────────────────────────────
    (
        "d.nakamura@engineering.co",
        "Code review request — PR #1247",
        (
            "Hi,\n\nI've opened PR #1247 which refactors the authentication "
            "middleware. Could you review it when you get a chance? There are a few "
            "design decisions I'd love your input on, especially around the token "
            "refresh logic.\n\nThanks,\nDaisuke"
        ),
        "reply-needed",
        "normal",
    ),
    (
        "l.santos@customersuccess.io",
        "Escalation: Enterprise client unhappy with response time",
        (
            "Hi,\n\nPinnacle Corp (Enterprise tier) has filed a formal complaint "
            "about response times on their support tickets. Average first-response "
            "is 6 hours vs our 2-hour SLA. Can we discuss remediation steps today?\n\n"
            "Thanks,\nLucia"
        ),
        "reply-needed",
        "urgent",
    ),
    (
        "p.wright@legalops.co",
        "Re: Data processing agreement — redline review",
        (
            "Hi,\n\nAttached is the redlined DPA from Initech's legal team. They've "
            "pushed back on clauses 4.2 (data retention) and 7.1 (liability cap). "
            "Can you review and suggest our counter-position by Wednesday?\n\n"
            "Best,\nPaul Wright"
        ),
        "reply-needed",
        "normal",
    ),
    # ── fyi extras ─────────────────────────────────────────────────────
    (
        "system@hrsuite.app",
        "Your PTO balance has been updated",
        (
            "Hi,\n\nYour PTO balance has been updated for the current quarter:\n"
            "- Vacation: 12 days remaining\n- Sick leave: 5 days remaining\n"
            "- Personal: 2 days remaining\n\nView details: hrsuite.app/pto\n\n"
            "— HR Suite"
        ),
        "fyi",
        "low",
    ),
    (
        "notifications@githubnotifier.dev",
        "New release: your-library v2.4.0",
        (
            "A new release has been published for your-library:\n\n"
            "v2.4.0 — Changelog:\n- Added async support for all endpoints\n"
            "- Fixed memory leak in connection pool\n- Deprecated legacy auth module\n\n"
            "View release: github.com/org/your-library/releases/v2.4.0"
        ),
        "fyi",
        "low",
    ),
]


def _generate_timestamp(rng: random.Random, base_time: datetime) -> str:
    """Generate a random timestamp within the last 7 days."""
    offset_seconds = rng.randint(0, 7 * 24 * 3600)
    ts = base_time - timedelta(seconds=offset_seconds)
    return ts.isoformat()


def generate_inbox(
    task_difficulty: str,
    seed: int = 42,
) -> tuple[list[Email], dict]:
    """Generate a reproducible inbox for the given task difficulty.

    Returns:
        A tuple of (list of Email objects, metadata dict with gold labels).
    """
    rng = random.Random(seed)
    base_time = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    templates = list(EMAIL_TEMPLATES)
    rng.shuffle(templates)

    match task_difficulty:
        case "easy":
            count = 5
            # Ensure exactly 1 urgent email
            urgent = [t for t in templates if t[3] == "urgent"]
            non_urgent = [t for t in templates if t[3] != "urgent"]
            selected_urgent = rng.sample(urgent, 1)
            selected_others = rng.sample(non_urgent, count - 1)
            selected = selected_urgent + selected_others
            rng.shuffle(selected)
        case "medium":
            count = 10
            # Mix of categories for priority assignment
            urgent = [t for t in templates if t[4] == "urgent"]
            normal = [t for t in templates if t[4] == "normal"]
            low = [t for t in templates if t[4] == "low"]
            selected = (
                rng.sample(urgent, min(2, len(urgent)))
                + rng.sample(normal, min(4, len(normal)))
                + rng.sample(low, min(4, len(low)))
            )
            rng.shuffle(selected)
            selected = selected[:count]
        case "hard":
            count = 15
            # Ensure representation across all 5 categories
            by_cat: dict[str, list] = {}
            for t in templates:
                by_cat.setdefault(t[3], []).append(t)
            selected = []
            for cat, items in by_cat.items():
                n = {"urgent": 3, "reply-needed": 5, "fyi": 3, "spam": 2, "newsletter": 2}.get(cat, 2)
                selected.extend(rng.sample(items, min(n, len(items))))
            rng.shuffle(selected)
            selected = selected[:count]
        case _:
            raise ValueError(f"Unknown task difficulty: {task_difficulty}")

    emails: list[Email] = []
    gold_labels: dict[str, dict] = {}

    for i, (sender, subject, body, category, priority) in enumerate(selected):
        email_id = f"email-{i + 1:03d}"
        ts = _generate_timestamp(rng, base_time)
        email = Email(
            id=email_id,
            sender=sender,
            subject=subject,
            body=body,
            timestamp=ts,
            labels=[],
            is_read=False,
            priority=None,
        )
        emails.append(email)
        gold_labels[email_id] = {
            "category": category,
            "gold_priority": priority,
        }

    metadata = {
        "gold_labels": gold_labels,
        "task_difficulty": task_difficulty,
        "email_count": len(emails),
        "seed": seed,
    }

    return emails, metadata
