import csv
import os
import json
from pprint import pprint

from dotenv import load_dotenv
from pathlib import Path

from pydantic.type_adapter import R
from ai_client import AIClient
from send_answer import send_answer


load_dotenv()

API_KEY = os.environ["AI_DEVS_4_API_KEY"]


CSV_PATH = Path(__file__).parent / "people.csv"
OUTPUT_CSV_PATH = Path(__file__).parent / "output.csv"
FILTERED_JSON_PATH = Path(__file__).parent / "filtered_people.json"

ALLOWED_TAGS = [
    "IT",
    "transport",
    "edukacja",
    "medycyna",
    "praca z ludźmi",
    "praca z pojazdami",
    "praca fizyczna",
]

system_prompt = f"""You are a tag classifier. Given a description, respond with a JSON array of matching tags. No explanation, no extra text — only valid JSON.

Allowed tags: {", ".join(ALLOWED_TAGS)}

Example: ["tag1", "tag2"]"""


def s01e01():
    client = AIClient(system_prompt)

    output_path = Path(__file__).parent / "output.csv"

    with open(CSV_PATH) as f, open(output_path, "w", newline="") as out:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or []) + ["tags"]
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(reader):
            if row["gender"] not in ["M"]:
                continue

            birth_year = int(row["birthDate"][:4])
            if birth_year < 1986 or birth_year > 2006:
                continue

            if row["birthPlace"] != "Grudziądz":
                continue

            answer = client.ask(
                f"Based on the description assign suitable tags. Description is:\n{row['job']}"
            )

            tags = json.loads(answer)
            if not tags:
                continue
            row["tags"] = ";".join(tags)
            pprint(
                f"{i}, {tags}",
            )
            writer.writerow(row)


def build_answer():
    with open(OUTPUT_CSV_PATH) as f:
        reader = csv.DictReader(f)
        people = []
        for row in reader:
            birth_year = int(row["birthDate"][:4])
            tags = row["tags"].split(";")
            if "transport" in tags:
                person = {
                    "name": row["name"],
                    "surname": row["surname"],
                    "born": birth_year,
                    "gender": row["gender"],
                    "city": row["birthPlace"],
                    "tags": row["tags"].split(";"),
                }
                people.append(person)
    with open(FILTERED_JSON_PATH, "w") as f:
        json.dump(people, f, indent=2, ensure_ascii=False)
    response = send_answer(task="people", answer=people)
    pprint(f"{people}, {len(people)}, {response}")


if __name__ == "__main__":
    # s01e01()
    build_answer()
