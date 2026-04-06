import httpx
import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

API_URL = os.environ["AI_DEVS_4_API_URL"]
API_KEY = os.environ["AI_DEVS_4_API_KEY"]


class AnswerPayload(BaseModel):
    task: str
    apikey: str = API_KEY
    answer: object


def send_answer(task: str, answer: object) -> dict:
    payload = AnswerPayload(task=task, answer=answer)
    response = httpx.post(url=API_URL, json=payload.model_dump())
    return response.json()
