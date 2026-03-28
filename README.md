---
title: Email Triage Agent Environment
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# Email Triage Agent Environment

An [OpenEnv](https://huggingface.co/spaces/openenv/openenv)-compatible environment for training and evaluating AI agents on **real-world email triage tasks**. Agents must classify, prioritise, and respond to emails across three difficulty levels.

---

## 1. Overview

Knowledge workers spend hours every day triaging email — reading, classifying, prioritising, replying, and archiving messages. This environment distils that workflow into a structured, programmatically-graded task that AI agents can learn and be evaluated on.

**Why it matters for agent training:**

- **Real-world relevance** — email triage is one of the most common productivity tasks humans perform daily.
- **Dense reward signal** — the grading function provides partial credit at every step, enabling meaningful learning.
- **Scalable difficulty** — three task tiers (easy → medium → hard) test increasingly complex reasoning and action composition.
- **Deterministic evaluation** — fixed random seeds ensure reproducible benchmarks.

---

## 2. Environment Description

The environment simulates an email inbox containing synthetic but realistic messages spanning five categories:

| Category       | Description                                                  |
|----------------|--------------------------------------------------------------|
| **urgent**     | Time-sensitive financial, legal, or security matters         |
| **reply-needed** | Questions or requests from colleagues/clients requiring a response |
| **fyi**        | Informational updates — no action needed                     |
| **spam**       | Phishing, scams, unsolicited offers                          |
| **newsletter** | Subscribed digests and marketing emails                      |

Some emails are **deliberately ambiguous** (e.g. "URGENT" in the subject but marketing content in the body) to test genuine comprehension.

---

## 3. Observation Space

Each observation is a JSON object with the following fields:

| Field              | Type                | Description                                      |
|--------------------|---------------------|--------------------------------------------------|
| `inbox`            | `List[Email]`       | All emails in the current inbox                  |
| `current_email`    | `Email \| null`     | The email currently "open" (if any)              |
| `action_history`   | `List[str]`         | String keys of all actions taken this episode     |
| `step_count`       | `int`               | Number of steps taken so far                     |
| `task_description` | `str`               | Natural-language description of the current task |
| `done`             | `bool`              | Whether the episode has ended                    |

### Email Object

| Field       | Type            | Description                              |
|-------------|-----------------|------------------------------------------|
| `id`        | `str`           | Unique identifier (e.g. `email-001`)     |
| `sender`    | `str`           | Sender email address                     |
| `subject`   | `str`           | Subject line                             |
| `body`      | `str`           | Full body text                           |
| `timestamp` | `str`           | ISO-8601 timestamp                       |
| `labels`    | `List[str]`     | Mutable list of applied labels           |
| `is_read`   | `bool`          | Whether the email has been opened        |
| `priority`  | `str \| null`   | `"urgent"`, `"normal"`, `"low"`, or null |

---

## 4. Action Space

The agent submits one action per step. All actions are JSON objects with the following schema:

| Action Type   | Required Fields            | Description                                   |
|---------------|----------------------------|-----------------------------------------------|
| `open`        | `email_id`                 | Open an email to read its contents             |
| `label`       | `email_id`, `label`        | Apply a text label to an email                 |
| `prioritise`  | `email_id`, `priority`     | Set priority: `"urgent"`, `"normal"`, `"low"` |
| `reply`       | `email_id`, `reply_text`   | Send a reply to the email                      |
| `archive`     | `email_id`                 | Archive an email (remove from active inbox)    |
| `skip`        | *(none)*                   | Skip this step (do nothing)                    |
| `done`        | *(none)*                   | Signal the agent is finished                   |

---

## 5. Tasks

### Easy — "Label the Urgent Email"

- **Inbox size:** 5 emails (exactly 1 is genuinely urgent)
- **Goal:** Open emails, identify the urgent one, label it `"urgent"`
- **Max steps:** 20
- **Grader:** +0.1 per email opened, +0.5 for correct label, −0.1 per wrong label

### Medium — "Sort and Prioritise Inbox"

- **Inbox size:** 10 emails of mixed types
- **Goal:** Assign correct priority (`urgent` / `normal` / `low`) to all 10
- **Max steps:** 30
- **Grader:** Kendall-tau correlation vs gold ordering; τ ≥ 0.7 → 1.0, scaled linearly below. +0.1 bonus if all done.

### Hard — "Triage, Reply, and Archive"

- **Inbox size:** 15 emails across all 5 categories
- **Goal:** Label all emails, reply to reply-needed emails, archive spam/newsletters
- **Max steps:** 40
- **Grader:** Weighted: 40% label accuracy + 40% reply quality (LLM-graded) + 20% archive correctness

---

## 6. Reward Function

Rewards provide **dense signal** on every step — not just at the end.

### Penalties

- **Loop penalty:** −0.05 for each repeated consecutive action (3+ identical in a row)
- **Early done:** calling `"done"` before minimum progress yields a lower score
- **Wrong labels:** −0.1 per incorrectly labelled email (easy task)

### Score normalisation

All final scores are clamped to **[0.0, 1.0]**.

---

## 7. Setup and Usage

### Local installation

```bash
# Clone
git clone https://huggingface.co/spaces/<your-username>/email-triage-env
cd email-triage-env

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
```

### Running inference

```bash
export HF_TOKEN="hf_..."
export MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
export ENV_URL="http://localhost:7860"

python inference.py
```

### Running tests

```bash
pytest tests/ -v
```

---

## 8. Baseline Scores

| Task   | Baseline Model | Score |
|--------|---------------|-------|
| easy   | [MODEL_NAME]  | TBD   |
| medium | [MODEL_NAME]  | TBD   |
| hard   | [MODEL_NAME]  | TBD   |

---

## 9. Deployment

### Hugging Face Spaces

1. Create a new Space on Hugging Face (Docker SDK).
2. Push this repository to the Space.
3. Set the following Secrets in the Space settings:
   - `HF_TOKEN` — your Hugging Face API token
   - `MODEL_NAME` — the model to use for LLM-based grading
4. The Space will build the Docker image and serve on port 7860.
5. Verify: `curl https://<space-url>/health` → `{"status": "ok"}`

### Validation

```bash
# Run the openenv validator against your deployed Space
openenv validate https://<space-url>
```

---

## 10. License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
