import requests


class LocalAIClient:
    def __init__(
        self,
        base_url="http://localhost:8080",
        model="naturallm-7b-instruct",
        temperature=0.7,
        timeout=120,
    ):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, system_prompt, user_prompt, context=None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]
