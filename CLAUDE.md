# 4rd-devs

Solutions to AI_devs 4 course tasks.

## Tasks

Each task lives in its own directory (`s01e01/`, `s02e01/`, etc.) and interacts with `https://hub.ag3nts.org/verify` via POST:

```json
{"apikey": "<key>", "task": "<task-name>", "answer": <answer>}
```

Use `send_answer(task, answer)` from `send_answer.py` to submit.
Use `AIClient` from `ai_client.py` for LLM calls.

Secret/mystery tasks: find a flag `{FLG:...}` — hints may be cryptic, don't take them literally.

## Utilities

- `ai_client.py` — `AIClient(system, model)` with `.ask(prompt)`; use `.client` directly for agentic loops
- `send_answer.py` — `send_answer(task, answer)`; also exports `API_URL`, `API_KEY`

## Code style

- No comments unless logic is non-obvious
- Prefer concise, readable code over verbose explanations
- Run tasks with `poetry run python -m sXXeYY.task` from the project root
