# 4rd-devs

This repo contains solutions to AI_devs 4 course tasks.

## What this repo is about

### 1. Regular tasks
Each task lives in its own directory (`s01e01/`, `s01e02/`, etc.) and requires building an agent-based solution that interacts with the course API at `https://hub.ag3nts.org/verify`.

All requests are POST with JSON body:
```json
{
  "apikey": "<key>",
  "task": "<task-name>",
  "answer": <answer>
}
```

Use `send_answer(task, answer)` from `send_answer.py` to submit final answers.
Use `AIClient` from `ai_client.py` for LLM calls (wraps Anthropic SDK).

### 2. Secret/mystery tasks
Hidden tasks discovered through hints. The goal is to figure out what the task expects and find a flag in the format `{FLG:...}`. Hints may be cryptic, ironic, or joke-based — don't take them literally.

## Shared utilities

- `ai_client.py` — `AIClient(system, model)` with `.ask(prompt)` for simple calls; use `.client` directly for tool-use agentic loops
- `send_answer.py` — `send_answer(task, answer)` submits to the grading API; also exports `API_URL` and `API_KEY`

## Patterns used

- Simple LLM classification: `AIClient.ask()` in a loop (see `s01e01/`)
- Agentic tool loop: `ai.client.messages.create()` with tools + message history (see `s01e02/`, `s01e05/`)
- FastAPI server acting as a conversational proxy (see `s01e03/`)
