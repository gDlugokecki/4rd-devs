import os

from anthropic import Anthropic
from anthropic.types import TextBlock
from dotenv import load_dotenv

load_dotenv()


class AIClient:
    def __init__(
        self,
        system: str = "You are a helpful assistant",
        model: str = "claude-sonnet-4-6",
    ):
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.system = system
        self.model = model

    def ask(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=self.system,
            messages=[{"role": "user", "content": prompt}],
        )

        block = response.content[0]
        if not isinstance(block, TextBlock):
            raise ValueError(f"Unexpected block type: {type(block)}")
        return block.text
