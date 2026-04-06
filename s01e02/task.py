import json
from pathlib import Path

from dotenv import load_dotenv

from ai_client import AIClient

from . import tools

load_dotenv()

FILTERED_PEOPLE_PATH = Path(__file__).parent.parent / "s01e01" / "filtered_people.json"


INVESTIGATION_TOOLS = [t for t in tools.DEFINITIONS if t["name"] != "submit_answer"]


def investigate_person(
    client: AIClient, name: str, surname: str, birth_year: int
) -> dict | None:
    """Investigate one person. Returns hit data if found near a power plant, else None."""
    system = (
        "You are an investigative agent checking if a person was seen near a power plant.\n"
        "1. Call get_location to get their coordinates.\n"
        "2. For each coordinate call find_nearby_power_plant.\n"
        "3. If any hit is found, call get_access_level (use the birthYear provided).\n"
        '4. Respond with JSON only: {"found": true, "powerPlant": "PWR...", "accessLevel": ...} '
        'or {"found": false} — no extra text.'
    )
    messages = [
        {
            "role": "user",
            "content": f"Investigate: {name} {surname}, birthYear: {birth_year}",
        }
    ]

    for _ in range(10):
        response = client.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            tools=INVESTIGATION_TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    try:
                        data = json.loads(block.text)
                        return data if data.get("found") else None
                    except json.JSONDecodeError:
                        return None
            return None

        if response.stop_reason != "tool_use":
            print(f"  [{name} {surname}] unexpected stop: {response.stop_reason}")
            return None

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = tools.process(block.name, block.input)
            print(f"  [{name} {surname}] {block.name}({block.input}) -> {result}")
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
            )
        messages.append({"role": "user", "content": tool_results})

    print(f"  [{name} {surname}] max iterations reached")
    return None


def s01e02():
    client = AIClient()

    with open(FILTERED_PEOPLE_PATH) as f:
        people = json.load(f)

    hits = []
    for person in people:
        name, surname, birth_year = person["name"], person["surname"], person["born"]
        print(f"Investigating {name} {surname}...")
        result = investigate_person(client, name, surname, birth_year)
        if result:
            hits.append({"name": name, "surname": surname, **result})
            print(f"  HIT: {name} {surname} -> {result}")

    print(f"\nInvestigation complete. {len(hits)} suspect(s) found.")

    for hit in hits:
        response = tools.submit_answer(
            hit["name"], hit["surname"], hit["accessLevel"], hit["powerPlant"]
        )
        print(f"Submitted {hit['name']} {hit['surname']}: {response}")


if __name__ == "__main__":
    s01e02()
