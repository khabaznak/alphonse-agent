import os

import requests


class OpenAIClient:
    def __init__(
        self,
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        timeout=60,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = timeout

    def complete(self, system_prompt, user_prompt):
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key in {self.api_key_env}.")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]
