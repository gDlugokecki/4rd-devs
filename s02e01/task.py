"""
Categorize task — classify 10 items as DNG or NEU using a ≤100-token prompt.

Reactor-related items must always be classified as NEU (even if dangerous).

Strategy:
- Start with a compact prompt template
- For each item, substitute actual {code}/{description} values and send to hub
- The hub's internal model sees the filled-in prompt and must output DNG or NEU
- On misclassification or budget error → reset + use Claude to refine prompt
- Loop until all 10 correct and flag is received
"""

import csv
import io
import json
import logging
import re
import time

import httpx
from anthropic.types import TextBlock
from dotenv import load_dotenv

from ai_client import AIClient
from send_answer import API_KEY, send_answer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TASK = "categorize"
CSV_URL = f"https://hub.ag3nts.org/data/{API_KEY}/categorize.csv"
MAX_ATTEMPTS = 20


# ──────────────────────────────────────────────────────────────
# Hub helpers
# ──────────────────────────────────────────────────────────────

def fetch_items() -> list[dict]:
    resp = httpx.get(CSV_URL, timeout=15)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    items = list(reader)
    log.info("Fetched %d items from CSV (columns: %s)", len(items), list(items[0].keys()) if items else [])
    for it in items:
        log.info("  %s: %s", it.get("code", "?"), it.get("description", "?"))
    return items


def reset_session() -> dict:
    data = send_answer(TASK, {"prompt": "reset"})
    log.info("Reset: %s", data)
    return data


def classify_item(prompt_template: str, code: str, description: str) -> dict:
    """Fill in the template with actual values and send to hub."""
    filled = prompt_template.replace("{code}", code).replace("{description}", description)
    data = send_answer(TASK, {"prompt": filled})
    log.info("  [%s] → %s", code, data)
    return data


def extract_flag(data: dict | str) -> str | None:
    text = json.dumps(data) if isinstance(data, dict) else str(data)
    m = re.search(r"\{FLG:[^}]+\}", text)
    return m.group(0) if m else None


# ──────────────────────────────────────────────────────────────
# One full classification run
# ──────────────────────────────────────────────────────────────

def run_classification(prompt_template: str) -> tuple[list[dict], str | None]:
    """
    Returns (results, flag_or_None).
    Stops early if budget is exhausted.
    """
    items = fetch_items()
    results = []

    for item in items:
        code = item.get("code", "")
        description = item.get("description", "")
        response = classify_item(prompt_template, code, description)
        results.append({"item": item, "response": response})

        flag = extract_flag(response)
        if flag:
            return results, flag

        code_val = response.get("code", 0)
        if code_val == -890:
            log.warning("Wrong classification for %s — stopping run", code)
            break
        if code_val == -910:
            log.warning("Budget exhausted — stopping run early")
            break

    return results, None


# ──────────────────────────────────────────────────────────────
# Prompt engineer (Claude as meta-agent)
# ──────────────────────────────────────────────────────────────

ENGINEER_SYSTEM = """You are a prompt engineer optimizing a classifier prompt for a legacy system.

CONSTRAINTS:
- The internal model has a 100-token context window (input + output combined).
- Your prompt template uses {code} and {description} as placeholders — you substitute real values before sending.
- The model must output exactly DNG (dangerous) or NEU (neutral) — nothing else.
- Reactor-related items (fuel cassettes, reactor parts, nuclear materials) must ALWAYS output NEU, even if dangerous. This is intentional.
- Other dangerous items (weapons, explosives, radioactive non-reactor) must output DNG.
- Write the prompt in English for token efficiency.
- Keep the static instruction part as short as possible — descriptions can be long.
- Do NOT include XML tags, markdown, or explanation — output ONLY the raw prompt template text.

Given feedback from previous attempts, suggest an improved prompt template."""


def ask_engineer(ai: AIClient, current_prompt: str, feedback: str) -> str:
    user_msg = f"""Current prompt template:
---
{current_prompt}
---

Feedback from the hub:
{feedback}

Provide an improved prompt template (raw text only, no explanation)."""

    response = ai.client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        system=ENGINEER_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    block = response.content[0]
    if not isinstance(block, TextBlock):
        raise ValueError(f"Unexpected block: {type(block)}")
    new_prompt = block.text.strip()
    log.info("Engineer suggested prompt:\n%s", new_prompt)
    return new_prompt


# ──────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────

INITIAL_PROMPT = (
    "DNG=weapon/firearm/explosive. NEU=tool/part/equipment/material. "
    "Reactor items always NEU.\n"
    "{code}: {description}\n"
    "DNG or NEU:"
)


def main():
    ai = AIClient(system=ENGINEER_SYSTEM, model="claude-sonnet-4-6")
    prompt = INITIAL_PROMPT
    feedback = "No previous attempt — first run."

    for attempt in range(1, MAX_ATTEMPTS + 1):
        log.info("=" * 60)
        log.info("Attempt %d", attempt)
        log.info("Prompt template:\n%s", prompt)

        if attempt > 1:
            reset_session()
            time.sleep(1)

        results, flag = run_classification(prompt)

        if flag:
            log.info("FLAG OBTAINED: %s", flag)
            print(f"\n{'=' * 60}\nFLAG: {flag}\n{'=' * 60}")
            return flag

        feedback_lines = [f"Attempt {attempt} results:"]
        for r in results:
            item = r["item"]
            resp = r["response"]
            feedback_lines.append(
                f"  code={item.get('code')} desc={item.get('description')!r} → {resp}"
            )
        feedback = "\n".join(feedback_lines)
        log.info("Feedback:\n%s", feedback)

        prompt = ask_engineer(ai, prompt, feedback)
        time.sleep(1)

    log.error("Reached max attempts without flag")
    return None


if __name__ == "__main__":
    main()
