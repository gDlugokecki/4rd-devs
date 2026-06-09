import base64
import io
import json
import os
import struct
from itertools import combinations

import httpx
import numpy as np
from PIL import Image
from anthropic import Anthropic
from anthropic.types import ToolUseBlock
from dotenv import load_dotenv

from send_answer import API_KEY

load_dotenv()

VERIFY_URL = "https://hub.ag3nts.org/verify"
BOARD_URL = f"https://hub.ag3nts.org/data/{API_KEY}/electricity.png"
REF_URL = "https://hub.ag3nts.org/i/solved_electricity.png"

CELLS = [f"{r}x{c}" for r in range(1, 4) for c in range(1, 4)]
_CW = {"T": "R", "R": "B", "B": "L", "L": "T"}


def fetch_image(url: str) -> Image.Image:
    resp = httpx.get(url, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def _cluster_centers(density: np.ndarray, threshold: float, gap: int = 5) -> list[int]:
    peaks = np.where(density > threshold)[0]
    if len(peaks) == 0:
        return []
    centers, run = [], [int(peaks[0])]
    for p in peaks[1:]:
        if p - run[-1] <= gap:
            run.append(int(p))
        else:
            centers.append(int(np.mean(run)))
            run = [int(p)]
    centers.append(int(np.mean(run)))
    return centers


def _find_4_borders(centers: list[int], min_cell: int = 60) -> list[int] | None:
    if len(centers) < 4:
        return None
    best, best_var = None, float("inf")
    for combo in combinations(centers, 4):
        diffs = [combo[i + 1] - combo[i] for i in range(3)]
        if min(diffs) < min_cell:
            continue
        var = float(np.var(diffs))
        if var < best_var:
            best_var, best = var, combo
    return list(best) if best else None


def detect_grid(img: Image.Image) -> tuple[int, int, int, int]:
    arr = np.array(img.convert("L"))
    dark = arr < 90
    rows = _find_4_borders(_cluster_centers(dark.mean(axis=1), 0.25), min_cell=60)
    cols = _find_4_borders(_cluster_centers(dark.mean(axis=0), 0.25), min_cell=60)
    if not rows or not cols:
        h, w = arr.shape
        return w // 4, h // 8, 3 * w // 4, 7 * h // 8
    return cols[0], rows[0], cols[-1], rows[-1]


def crop_cells(img: Image.Image) -> dict[str, Image.Image]:
    x1, y1, x2, y2 = detect_grid(img)
    cw = (x2 - x1) // 3
    ch = (y2 - y1) // 3
    cells = {}
    for row in range(1, 4):
        for col in range(1, 4):
            cx = x1 + (col - 1) * cw
            cy = y1 + (row - 1) * ch
            cells[f"{row}x{col}"] = img.crop((cx, cy, cx + cw, cy + ch))
    return cells


def upscale(cell: Image.Image, scale: int = 4) -> Image.Image:
    return cell.resize((cell.width * scale, cell.height * scale), Image.NEAREST)


def rotate_connections(conns: frozenset[str], n: int) -> frozenset[str]:
    result = set(conns)
    for _ in range(n % 4):
        result = {_CW[c] for c in result}
    return frozenset(result)


def find_rotation(current: frozenset[str], target: frozenset[str]) -> int:
    for n in range(4):
        if rotate_connections(current, n) == target:
            return n
    return -1


DESCRIBE_PROMPT = """\
Puzzle tile. The black rectangle along all 4 edges is the BORDER FRAME — ignore it.
Inside: one cable shape (thick black lines).

Visual shapes:
  CORNER   = L or J shape  → exits 2 adjacent edges
  STRAIGHT = full bar       → exits 2 opposite edges (top+bottom or left+right)
  T-SHAPE  = T or ⊢ shape  → exits 3 edges
  CROSS    = + shape        → exits all 4 edges

Analyze the image. On the very last line write ONLY the exit-edge letters with no other text.
Use: T=top  R=right  B=bottom  L=left  (example last lines: "TR" or "BLT" or "LR")"""


def describe_cell(client: Anthropic, cell: Image.Image, label: str) -> frozenset[str]:
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_to_b64(upscale(cell))}},
                {"type": "text", "text": DESCRIBE_PROMPT},
            ],
        }],
    )
    text = resp.content[0].text.strip().upper()
    last_line = next((ln.strip() for ln in reversed(text.splitlines()) if ln.strip()), "")
    conns = frozenset(c for c in last_line if c in "TRBL")
    print(f"    {label}: last_line={last_line!r}  → {sorted(conns)}")
    return conns


_ref_connections: dict[str, frozenset[str]] | None = None


def get_ref_connections(client: Anthropic) -> dict[str, frozenset[str]]:
    global _ref_connections
    if _ref_connections is None:
        print("  Describing reference board (cached after first run)...")
        ref_cells = crop_cells(fetch_image(REF_URL))
        _ref_connections = {name: describe_cell(client, ref_cells[name], f"ref {name}") for name in CELLS}
    return _ref_connections


def compute_rotation_plan(client: Anthropic) -> dict[str, int]:
    current_cells = crop_cells(fetch_image(BOARD_URL))
    ref_connections = get_ref_connections(client)

    plan: dict[str, int] = {}
    for name in CELLS:
        cur = describe_cell(client, current_cells[name], f"cur {name}")
        ref = ref_connections[name]
        n = find_rotation(cur, ref)
        if n == -1:
            print(f"  WARNING {name}: {sorted(cur)} → {sorted(ref)} — no match, skipping")
            n = 0
        else:
            print(f"  {name}: {sorted(cur)} → {sorted(ref)} = {n} rotation(s)")
        plan[name] = n

    return plan


TOOLS = [
    {
        "name": "analyze_board",
        "description": (
            "Describes each cell's cable connections individually using vision, "
            "then computes the exact rotation count per cell using math. "
            "Returns a JSON rotation plan."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "rotate",
        "description": "Rotates one cell 90° clockwise. Cell: 'RowxCol' e.g. '2x3'. Returns API response (flag if solved).",
        "input_schema": {
            "type": "object",
            "properties": {"cell": {"type": "string"}},
            "required": ["cell"],
        },
    },
    {
        "name": "reset",
        "description": "Resets the board to its original state.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


def handle_tool(name: str, inp: dict, client: Anthropic) -> tuple[list[dict], str | None]:
    flag = None

    if name == "analyze_board":
        plan = compute_rotation_plan(client)
        content = [{"type": "text", "text": json.dumps(plan)}]

    elif name == "rotate":
        cell = inp["cell"]
        resp = httpx.post(
            VERIFY_URL,
            json={"apikey": API_KEY, "task": "electricity", "answer": {"rotate": cell}},
            timeout=30,
        )
        result = resp.json()
        print(f"  rotate {cell} → {result}")
        text = json.dumps(result)
        if "FLG:" in text:
            flag = text
        content = [{"type": "text", "text": text}]

    elif name == "reset":
        httpx.get(f"{BOARD_URL}?reset=1", timeout=30)
        print("  Board reset.")
        content = [{"type": "text", "text": "Board reset to initial state."}]

    else:
        content = [{"type": "text", "text": f"Unknown tool: {name}"}]

    return content, flag


SYSTEM = """You are solving a 3×3 electrical circuit puzzle.

CELL ADDRESSING (RowxCol):
  1x1 | 1x2 | 1x3   ← Row 1 (top)
  2x1 | 2x2 | 2x3   ← Row 2 (middle)
  3x1 | 3x2 | 3x3   ← Row 3 (bottom)

WORKFLOW:
1. Call analyze_board — returns JSON with rotation count per cell (0–3)
2. For each cell with count > 0, call rotate() that many times
3. Call analyze_board again to verify — all values should be 0
4. If rotate() response contains 'FLG:', report it and stop

Call reset() only if the board is badly wrong."""


def main():
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    httpx.get(f"{BOARD_URL}?reset=1", timeout=30)
    print("Board reset.\n")

    messages = [{"role": "user", "content": "Solve the electricity puzzle."}]

    while True:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "end_turn":
            final = next((b.text for b in reversed(resp.content) if hasattr(b, "text")), "")
            print(f"\nAgent: {final}")
            break

        tool_results = []
        flag = None

        for block in resp.content:
            if isinstance(block, ToolUseBlock):
                print(f"\n[{block.name}] {block.input}")
                content, f = handle_tool(block.name, block.input, client)
                if f:
                    flag = f
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": content}
                )

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        if flag:
            print(f"\nFLAG: {flag}")
            break


if __name__ == "__main__":
    main()
