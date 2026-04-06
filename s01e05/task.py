"""
Railway task — AI agent that activates route X-01.

The agent is given a single tool: `railway_action`.
It starts by calling help, reads the self-documented API,
then follows the described sequence to activate the route.
503 retries and rate-limit waits are handled transparently in the tool.
"""

import json
import logging
import os
import re
import time
import httpx
from datetime import datetime, timezone

from dotenv import load_dotenv

from ai_client import AIClient
from send_answer import send_answer, API_URL, API_KEY

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TASK = "railway"
MODEL = "claude-sonnet-4-6"
MAX_AGENT_ITERATIONS = 30

SYSTEM_PROMPT = """You are an autonomous agent tasked with activating railway route X-01 via a self-documenting API.

Your only tool is `railway_action`. Follow this process strictly:

1. Call `railway_action` with `{"action": "help"}` to get the full API documentation.
2. Read the documentation carefully — it lists all available actions, their parameters, and the exact sequence required.
3. Execute the documented sequence step by step, using exactly the action names and parameter names from the docs.
4. After each call, check the response for a flag matching the pattern {FLG:...}. If found, report it immediately.
5. If a step fails, read the error message carefully — it usually tells you exactly what went wrong.

Do NOT guess action names. Do NOT skip steps. Follow the documentation precisely."""


TOOL_DEFINITION = {
    "name": "railway_action",
    "description": (
        "Calls the railway API with the given action payload. "
        "Handles 503 retries and rate-limit waits automatically. "
        "Returns the JSON response from the API."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "payload": {
                "type": "object",
                "description": (
                    'The action payload, e.g. {"action": "help"} or '
                    '{"action": "some_action", "param": "value"}.'
                ),
            }
        },
        "required": ["payload"],
    },
}


def _backoff(attempt: int) -> float:
    return min(2**attempt, 60)


def _rate_limit_wait(headers: dict) -> float:
    for key in ("x-ratelimit-reset", "ratelimit-reset", "retry-after", "x-retry-after"):
        value = headers.get(key)
        if value is None:
            continue
        try:
            reset_ts = float(value)
            now = datetime.now(timezone.utc).timestamp()
            wait = max(reset_ts - now, 0) + 1
            log.info("Rate-limit reset in %.1fs (header %s=%s)", wait, key, value)
            return wait
        except ValueError:
            pass
    return 5.0


def _log_rl_headers(headers: dict):
    rl = {
        k: v
        for k, v in headers.items()
        if any(x in k.lower() for x in ("ratelimit", "retry", "x-rate"))
    }
    if rl:
        log.info("Rate-limit headers: %s", rl)


def call_railway_api(action_payload: dict, max_retries: int = 25) -> dict:
    body = {"apikey": API_KEY, "task": TASK, "answer": action_payload}

    for attempt in range(1, max_retries + 1):
        try:
            resp = httpx.post(API_URL, json=body, timeout=30)
            headers = dict(resp.headers)
            _log_rl_headers(headers)

            if resp.status_code == 503:
                wait = _backoff(attempt)
                time.sleep(wait)
                continue

            if resp.status_code == 429:
                wait = _rate_limit_wait(headers)
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            remaining = headers.get("x-ratelimit-remaining") or headers.get(
                "ratelimit-remaining"
            )
            if remaining is not None:
                try:
                    if int(remaining) <= 1:
                        wait = _rate_limit_wait(headers)
                        log.warning("Rate limit almost gone — waiting %.1fs", wait)
                        time.sleep(wait)
                except ValueError:
                    pass

            return data

        except httpx.HTTPStatusError as exc:
            log.error("HTTP error: %s", exc)
            time.sleep(_backoff(attempt))
        except httpx.RequestError as exc:
            log.error("Request error: %s", exc)
            time.sleep(_backoff(attempt))

    raise RuntimeError(f"API call failed after {max_retries} attempts")


def execute_tool(name: str, inputs: dict) -> str:
    if name != "railway_action":
        return json.dumps({"error": f"Unknown tool: {name}"})
    result = call_railway_api(inputs["payload"])
    return json.dumps(result, ensure_ascii=False)


def check_for_flag(text: str) -> str | None:
    m = re.search(r"\{FLG:[^}]+\}", text)
    return m.group(0) if m else None


def run_agent():
    ai = AIClient(system=SYSTEM_PROMPT, model=MODEL)

    messages = [
        {
            "role": "user",
            "content": (
                "Activate railway route X-01. "
                "Start by calling the help action to read the API documentation, "
                "then follow the documented sequence exactly."
            ),
        }
    ]

    for iteration in range(MAX_AGENT_ITERATIONS):
        log.info("=== Agent iteration %d ===", iteration + 1)

        response = ai.client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=[TOOL_DEFINITION],
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        # Check for flag in any text block
        for block in response.content:
            if hasattr(block, "text"):
                flag = check_for_flag(block.text)
                if flag:
                    log.info("FLAG FOUND: %s", flag)
                    result = send_answer(TASK, flag)
                    print(
                        f"\n{'=' * 60}\nFLAG: {flag}\nSubmit result: {result}\n{'=' * 60}"
                    )
                    return flag

        if response.stop_reason == "end_turn":
            text = next((b.text for b in response.content if hasattr(b, "text")), "")
            log.info("Agent finished: %s", text)
            print(f"\nAgent done:\n{text}")
            return None

        if response.stop_reason != "tool_use":
            log.warning("Unexpected stop_reason: %s", response.stop_reason)
            break

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            log.info("Tool call: %s(%s)", block.name, json.dumps(block.input))
            result = execute_tool(block.name, block.input)

            flag = check_for_flag(result)
            if flag:
                log.info("FLAG in tool result: %s", flag)
                submit_result = send_answer(TASK, flag)
                print(
                    f"\n{'=' * 60}\nFLAG: {flag}\nSubmit result: {submit_result}\n{'=' * 60}"
                )

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
            )

        messages.append({"role": "user", "content": tool_results})

    log.error("Max agent iterations reached without finding flag")
    return None


def poll():
    body = {"apikey": API_KEY, "task": TASK, "answer": {"action": "help"}}
    resp = httpx.post(API_URL, json=body, timeout=30)
    log.info(resp)


if __name__ == "__main__":
    # run_agent()
    poll()
