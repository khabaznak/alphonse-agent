import requests


class OllamaClient:
    def __init__(
        self,
        base_url="http://localhost:11434",
        model="mistral:7b-instruct",
        timeout=120,
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    def complete(self, system_prompt, user_prompt):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        return response.json()["message"]["content"]
