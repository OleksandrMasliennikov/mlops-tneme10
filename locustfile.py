"""Load test a local-only Ollama route through LiteLLM on port 4000."""

import random

from locust import HttpUser, between, task


PROMPTS = (
    "Explain what a Linux process is in two sentences.",
    "Describe a Python list in two sentences.",
    "What does a database index do? Answer in two sentences.",
    "Explain gradient descent in two sentences.",
)


class LocalLLMUser(HttpUser):
    host = "http://127.0.0.1:4000"
    wait_time = between(1, 2)

    @task
    def chat(self):
        payload = {
            "model": "chat-local",
            "messages": [{"role": "user", "content": random.choice(PROMPTS)}],
            "max_tokens": 32,
            "stream": False,
        }

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            name="POST /v1/chat/completions (chat-local)",
            timeout=180,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:160]}")
                return
            try:
                answer = response.json()["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError):
                response.failure("Invalid chat response")
                return
            if not answer:
                response.failure("Empty model response")
