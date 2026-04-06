from pprint import pprint

from send_answer import send_answer


def answer():
    payload = {
        "url": "https://c570-88-156-142-239.ngrok-free.app",
        "sessionID": "abc123",
    }
    response = send_answer(task="proxy", answer=payload)
    pprint(response)


if __name__ == "__main__":
    answer()
