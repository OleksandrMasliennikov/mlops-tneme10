# pip install openai
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"          # довільне значення – Ollama не потребує ключа
)

response = client.chat.completions.create(
    model="mistral:7b-instruct-q4_K_M",
    messages=[
        {"role": "system", "content": "Ти корисний асистент."},
        {"role": "user",   "content": "Поясни що таке KV-cache в LLM за 3 речення."}
    ],
    temperature=0.7,
    max_tokens=300
)

print(response.choices[0].message.content)
print(f"Tokens used: {response.usage.total_tokens}")