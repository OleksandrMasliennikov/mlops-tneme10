# AI Inference Infrastructure & Optimization — Code Examples

Усі приклади коду з презентації, зібрані в одному файлі в порядку появи на слайдах.

---

## Slide 14 — GGUF і Ollama: встановлення та базовий запит

```bash
# Встановлення Ollama (Linux/macOS)
curl -fsSL https://ollama.com/install.sh | sh

# Завантаження та запуск Mistral 7B Q4_K_M (~4.1 GB)
ollama pull mistral:7b-instruct-q4_K_M
ollama run mistral:7b-instruct-q4_K_M

# OpenAI-compatible API (для LiteLLM proxy)
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mistral","messages":[{"role":"user","content":"Hello"}]}'
```

---

## Slide 24 — LiteLLM: docker-compose для unified gateway

```yaml
# docker-compose.yml
services:
  litellm:
    image: ghcr.io/berriai/litellm:main-latest
    ports: ["4000:4000"]
    environment:
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
    command: >
      --model ollama/mistral
      --model openai/gpt-5-mini
      --fallbacks '[{"ollama/mistral": ["gpt-5-mini"]}]'
      --port 4000
```

---

## Slide 26 — OpenRouter: базовий Python-клієнт

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)
response = client.chat.completions.create(
    model="openrouter/free",  # авто-вибір безкоштовної моделі
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## Slide 31 — Ollama: встановлення та CLI-команди

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows – завантажити інсталятор
# https://ollama.com/download/windows

# Docker
docker run -d -p 11434:11434 \
  --name ollama ollama/ollama
```

```bash
$ ollama pull mistral:7b-instruct-q4_K_M
  # завантажити модель (~4.1 GB)
$ ollama run mistral:7b-instruct-q4_K_M
  # інтерактивний чат у терміналі
$ ollama list
  # список завантажених моделей
$ ollama show mistral --modelfile
  # показати Modelfile
$ ollama rm mistral:7b
  # видалити модель
$ ollama serve
  # запустити API daemon явно
$ OLLAMA_NUM_THREADS=8 ollama serve
  # вказати кількість CPU потоків
```

---

## Slide 33 — Python: базовий запит до Ollama

**Варіант A — через OpenAI SDK (рекомендовано)**

```python
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
```

**Варіант B — Ollama Python SDK**

```python
# pip install ollama
import ollama

resp = ollama.chat(
    model="mistral:7b-instruct-q4_K_M",
    messages=[{
        "role": "user",
        "content": "Що таке PagedAttention?"
    }]
)
print(resp["message"]["content"])

# Inspect metadata
print(resp["eval_count"])       # output tokens
print(resp["eval_duration"])    # ns
tps = resp["eval_count"] / (resp["eval_duration"] / 1e9)
print(f"TPS: {tps:.1f}")
```

---

## Slide 34 — Python: Streaming та вимірювання метрик (TTFT/TPS)

```python
import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def measure_llm(model: str, prompt: str) -> dict:
    """Вимірює TTFT, TPS і загальну latency."""
    t_start = time.perf_counter()
    first_token_time = None
    full_text = []

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=200
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            full_text.append(delta)
            print(delta, end="", flush=True)   # stream to console

    t_end = time.perf_counter()
    total_tokens = len("".join(full_text).split())

    return {
        "TTFT_ms":    round((first_token_time - t_start) * 1000, 1),
        "total_s":    round(t_end - t_start, 2),
        "TPS_approx": round(total_tokens / (t_end - (first_token_time or t_start)), 1),
    }

result = measure_llm("mistral:7b-instruct-q4_K_M", "Explain gradient descent in 5 sentences.")
print(f"\n\nTTFT: {result['TTFT_ms']} ms  |  TPS: {result['TPS_approx']}  |  Total: {result['total_s']}s")

# Output: TTFT: 2340.7 ms  |  TPS: 7.2  |  Total: 14.8s     ← типово для Intel i7, CPU-only
```

---

## Slide 35 — REST API: curl та JavaScript

**curl — chat/completions (OpenAI-format)**

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":  "mistral:7b-instruct-q4_K_M",
    "stream": false,
    "messages": [
      {"role": "system", "content": "Відповідай лише українською."},
      {"role": "user",   "content": "Що таке квантизація моделі?"}
    ]
  }'
```

**JavaScript / Node.js — streaming**

```javascript
// npm install openai
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:11434/v1",
  apiKey:  "ollama",
});
const stream = await client.chat.completions.create({
  model:    "mistral:7b-instruct-q4_K_M",
  messages: [{ role: "user", content: "Explain batching in LLMs." }],
  stream:   true,
  max_tokens: 200,
});

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
}
```

---

## Slide 36 — Modelfile: кастомізація системного промпту

```dockerfile
# Modelfile — описує кастомну модель
FROM mistral:7b-instruct-q4_K_M

# Системний промпт – встановлюється раз, не повторюється у кожному запиті
SYSTEM """
Ти – MLOps-асистент для курсу. Відповідай чітко і з прикладами коду.
Завжди вказуй джерела якщо посилаєшся на факти.
"""

# Параметри генерації
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_predict 1024     # max output tokens
PARAMETER num_ctx 4096         # context window

# Опис для ollama list
LICENSE Apache 2.0
```

```bash
# Створити кастомну модель
ollama create mlops-assistant \
  -f ./Modelfile

# Перевірити
ollama list

# Запустити
ollama run mlops-assistant

# Використати через API
curl localhost:11434/v1/chat/completions \
  -d '{
    "model": "mlops-assistant",
    "messages": [{
      "role": "user",
      "content": "Поясни vLLM"
    }]
  }'
```

---

## Slide 37 — Ollama у Docker (локально, CPU)

```yaml
# docker-compose.yml — локальна розробка (CPU)
services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama    # кешуємо моделі між рестартами
    environment:
      - OLLAMA_NUM_THREADS=8         # кількість CPU потоків
      - OLLAMA_MAX_LOADED_MODELS=1   # тільки одна модель в пам'яті
    restart: unless-stopped

  # Опціонально: Open WebUI (Llama-like chat UI)
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    environment:
      - OLLAMA_API_BASE_URL=http://ollama:11434
    depends_on:
      - ollama

volumes:
  ollama_data:
```

```bash
# Запустити стек
docker compose up -d

# Завантажити модель у контейнер
docker exec ollama \
  ollama pull mistral:7b-instruct-q4_K_M

# Перевірити API (з хоста)
curl http://localhost:11434/api/tags

# Переглянути логи
docker compose logs -f ollama

# Open WebUI → http://localhost:3000
```

---

## Slide 40 — Docker Compose: Ollama + GPU на хмаровому сервері

```yaml
# docker-compose.gpu.yml
services:
  ollama:
    image: ollama/ollama:latest
    runtime: nvidia                      # потребує nvidia-container-toolkit
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_NUM_GPU=999               # використовувати всі GPU-шари
    restart: unless-stopped

  nginx:                                 # reverse proxy + auth
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
      - ./certs:/etc/nginx/certs
    depends_on:
      - ollama

volumes:
  ollama_data:
```

**nginx.conf — базовий захист API токеном**

```nginx
server {
  listen 443 ssl;
  server_name your.domain.com;

  ssl_certificate     /etc/nginx/certs/cert.pem;
  ssl_certificate_key /etc/nginx/certs/key.pem;

  location / {
    # Перевірка Bearer token
    if ($http_authorization != "Bearer YOUR_SECRET_TOKEN") {
      return 401;
    }
    proxy_pass http://ollama:11434;
    proxy_read_timeout 300s;
  }
}
```

```bash
# Запуск:
docker compose -f docker-compose.gpu.yml up -d
docker exec ollama ollama pull mistral:7b-instruct
```

---

## Slide 41 — Python: підключення до хмарового Ollama

```python
import os
from openai import OpenAI

# Замість localhost – ваш хмаровий ендпойнт
OLLAMA_ENDPOINT = os.getenv("OLLAMA_URL", "https://your.domain.com")
OLLAMA_TOKEN    = os.getenv("OLLAMA_TOKEN", "")

client = OpenAI(
    base_url=f"{OLLAMA_ENDPOINT}/v1",
    api_key=OLLAMA_TOKEN or "ollama"     # якщо nginx вимагає token
)

# Перевірка доступних моделей
models = client.models.list()
print([m.id for m in models.data])

# Chat запит (повністю ідентичний локальному!)
response = client.chat.completions.create(
    model="mistral:7b-instruct-q4_K_M",
    messages=[
        {"role": "system",  "content": "You are an MLOps expert."},
        {"role": "user",    "content": "Compare vLLM and TGI in 3 points."}
    ],
    temperature=0.5,
    max_tokens=400
)
print(response.choices[0].message.content)
```

**.env файл для конфігурації**

```bash
# .env (не комітити в git!)
OLLAMA_URL=https://your.domain.com
OLLAMA_TOKEN=your_secret_token_here

# Локальна розробка
# OLLAMA_URL=http://localhost:11434
# OLLAMA_TOKEN=
```

**Переключення local ↔ cloud однією змінною**

```python
import os

IS_LOCAL = os.getenv("ENV") == "local"

client = OpenAI(
    base_url=(
        "http://localhost:11434/v1"
        if IS_LOCAL else
        os.getenv("OLLAMA_URL") + "/v1"
    ),
    api_key=os.getenv("OLLAMA_TOKEN", "ollama")
)
```

---

## Slide 46 — Python: базовий запит до OpenRouter

```python
import os
from openai import OpenAI

# Єдина точка змін порівняно з Ollama!
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

response = client.chat.completions.create(
    model="openrouter/free",   # або платна, напр. openai/gpt-5-mini
    messages=[
        {"role": "system", "content": "You are a helpful MLOps assistant."},
        {"role": "user",   "content": "What is continuous batching in LLMs?"}
    ],
    extra_headers={
        # Відображається у usage-дашборді OpenRouter
        "X-Title": "MLOps Course Demo",
        # HTTP-Referer ідентифікує ваш застосунок
        "HTTP-Referer": "https://your-app.com",
    },
    temperature=0.5,
    max_tokens=500,
)

print(response.choices[0].message.content)

# Метадані
usage = response.usage
print(f"In: {usage.prompt_tokens} | Out: {usage.completion_tokens}")
```

---

## Slide 47 — OpenRouter: Streaming та Tool Calling

**Streaming**

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

stream = client.chat.completions.create(
    model="anthropic/claude-haiku-4.5",
    messages=[{
        "role": "user",
        "content": "Write a Python class for a simple LRU cache."
    }],
    stream=True
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
```

**Tool / Function Calling**

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_model_info",
        "description": "Get LLM model details",
        "parameters": {
            "type": "object",
            "properties": {
                "model_name": {"type": "string"}
            },
            "required": ["model_name"]
        }
    }
}]

response = client.chat.completions.create(
    model="openai/gpt-5-mini",
    messages=[{
        "role": "user",
        "content": "Tell me about mistral-7b"
    }],
    tools=tools,
    tool_choice="auto"
)

msg = response.choices[0].message
if msg.tool_calls:
    for call in msg.tool_calls:
        print(call.function.name)
        print(call.function.arguments)
```

---

## Slide 48 — OpenRouter: Fallback та Provider Routing

```python
from openai import OpenAI
import os
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)
# ── Fallback: список моделей від пріоритетної до запасної ──────────────────
response = client.chat.completions.create(
    model="openai/gpt-5.6",           # перша спроба
    messages=[{"role": "user", "content": "Summarize PagedAttention."}],
    extra_body={
        # Якщо gpt-5.6 недоступний або повертає помилку → спробуємо по черзі:
        "models": [
            "openai/gpt-5.6",
            "anthropic/claude-sonnet-5",
            "mistralai/mistral-large",
        ],
        # ── Provider Routing ────────────────────────────────────────────────
        "provider": {
            "order":         ["Anthropic", "OpenAI", "Together"],  # пріоритет провайдерів
            "allow_fallbacks": True,     # дозволити перехід між провайдерами
            "data_collection": "deny",   # заборонити зберігання даних (де підтримується)
        },
        # ── Route: cheapest / fastest / latency ─────────────────────────────
        "route": "fallback",             # або "cheapest" | "fastest" | "latency"
    }
)
print(response.choices[0].message.content)
# Який провайдер відповів:
print(response.model)
```

*Значення `route`: `fallback` — спробуй наступну модель при помилці/таймауті (default); `cheapest` — вибрати найдешевшого провайдера для цієї моделі; `fastest` — вибрати провайдера з найнижчою latency (за останні 5 хв).*

---

## Slide 49 — Async: паралельні запити до OpenRouter

```python
import asyncio, os
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)
MODEL = "mistralai/mistral-7b-instruct"   # безкоштовна модель
QUESTIONS = [
    "What is KV-cache in LLMs?",
    "Explain quantization in 2 sentences.",
    "What is continuous batching?",
    "Compare vLLM and TGI.",
    "What is TTFT metric?",
]

async def ask(question: str, idx: int) -> dict:
    """Один async запит."""
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
        max_tokens=150,
    )
    return {"q": question, "a": resp.choices[0].message.content, "idx": idx}

async def main():
    # Запускаємо всі 5 запитів паралельно
    tasks = [ask(q, i) for i, q in enumerate(QUESTIONS)]
    results = await asyncio.gather(*tasks)
    for r in sorted(results, key=lambda x: x["idx"]):
        print(f"Q{r['idx']+1}: {r['q']}")
        print(f"A:  {r['a'][:120]}...\n")

asyncio.run(main())
# Час: ~2-3 секунди замість ~10-15 при послідовних запитах
```

*`asyncio.gather()` пускає всі запити одночасно. Rate limit OpenRouter: ~200 req/min для більшості моделей. Використовуй `asyncio.Semaphore` для контролю.*

---

## Slide 50 — Порівняння відповідей кількох моделей

```python
import asyncio, os
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

MODELS = [
    "openai/gpt-5-mini",
    "anthropic/claude-haiku-4.5",
    "openrouter/free",   # авто-вибір безкоштовної моделі
    "mistralai/mistral-small",         # дешева
]

async def compare(prompt: str):
    async def query(model):
        import time
        t = time.perf_counter()
        r = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return {
            "model":   model.split("/")[-1],
            "answer":  r.choices[0].message.content,
            "latency": round((time.perf_counter() - t) * 1000),
            "tokens":  r.usage.completion_tokens,
        }
    results = await asyncio.gather(*[query(m) for m in MODELS])
    for r in results:
        print(f"[{r['model']}] {r['latency']}ms / {r['tokens']} tokens")
        print(r['answer'][:200], "\n")

asyncio.run(compare("Explain PagedAttention in 3 sentences."))
```

---

## Slide 52 — LiteLLM Proxy: єдиний endpoint для Ollama + OpenRouter

```yaml
# config.yaml – LiteLLM Proxy
model_list:
  # Локальна Ollama (пріоритет)
  - model_name: "chat"
    litellm_params:
      model: ollama/mistral:7b-instruct-q4_K_M
      api_base: http://localhost:11434

  # Хмарний Ollama (backup)
  - model_name: "chat"
    litellm_params:
      model: ollama/mistral:7b-instruct-q4_K_M
      api_base: https://your.domain.com/v1
      api_key: os.environ/OLLAMA_TOKEN

  # OpenRouter fallback
  - model_name: "chat"
    litellm_params:
      model: openrouter/mistralai/mistral-small
      api_key: os.environ/OPENROUTER_API_KEY

  # Дорожча модель для складних запитів
  - model_name: "smart"
    litellm_params:
      model: openrouter/openai/gpt-5-mini
      api_key: os.environ/OPENROUTER_API_KEY

router_settings:
  routing_strategy: "latency-based-routing"
  fallbacks:
    - "ollama/mistral:7b-instruct-q4_K_M":
        ["openrouter/mistralai/mistral-small"]

litellm_settings:
  cache: true
  cache_params:
    type: "redis"
    host: "localhost"
    port: 6379
```

```bash
# pip install 'litellm[proxy]'
litellm --config config.yaml \
        --port 4000 --debug

# Тест через curl
curl http://localhost:4000/chat/completions \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chat",
    "messages": [{
      "role": "user",
      "content": "Hello from LiteLLM!"
    }]
  }'
```

```python
# Той самий OpenAI client!
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:4000",
    api_key="sk-1234"
)
r = client.chat.completions.create(
    model="chat",
    messages=[{
      "role":"user",
      "content":"Test fallback"
    }]
)
```

---

## Slide 53 — Python: єдиний клієнт для всіх бекендів

```python
import os
from openai import OpenAI
from enum import Enum

class LLMBackend(Enum):
    LOCAL_OLLAMA  = "local_ollama"
    CLOUD_OLLAMA  = "cloud_ollama"
    OPENROUTER    = "openrouter"
    LITELLM_PROXY = "litellm"

def get_client(backend: LLMBackend) -> tuple[OpenAI, str]:
    """Повертає (client, model_name) для обраного бекенду."""
    configs = {
        LLMBackend.LOCAL_OLLAMA: {
            "base_url": "http://localhost:11434/v1",
            "api_key":  "ollama",
            "model":    "mistral:7b-instruct-q4_K_M",
        },
        LLMBackend.CLOUD_OLLAMA: {
            "base_url": os.getenv("OLLAMA_URL", "") + "/v1",
            "api_key":  os.getenv("OLLAMA_TOKEN", "ollama"),
            "model":    "mistral:7b-instruct-q4_K_M",
        },
        LLMBackend.OPENROUTER: {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key":  os.environ["OPENROUTER_API_KEY"],
            "model":    "mistralai/mistral-7b-instruct",
        },
        LLMBackend.LITELLM_PROXY: {
            "base_url": "http://localhost:4000",
            "api_key":  "sk-1234",
            "model":    "chat",
        },
    }
    cfg = configs[backend]
    return OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"]), cfg["model"]

# Використання – однаковий код для всіх бекендів!
BACKEND = LLMBackend(os.getenv("LLM_BACKEND", "local_ollama"))
client, model = get_client(BACKEND)

response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "What is inference optimization?"}],
    max_tokens=300,
)
print(response.choices[0].message.content)
```

---

## Slide 58 — Налаштування середовища (перед лабораторними)

```bash
# Ollama
curl -fsSL https://ollama.com/install.sh | sh

# LiteLLM
pip install litellm 'litellm[proxy]'

# Locust
pip install locust
```

---

## Slide 59 — Lab 1: Ollama Deployment та baseline вимірювання

```bash
# Крок 1: Завантажити модель (перший раз ~4.1 GB)
ollama pull mistral:7b-instruct-q4_K_M

# Крок 2: Тест у CLI
ollama run mistral:7b-instruct-q4_K_M "Explain KV-cache in 3 sentences"

# Крок 3: Перевірити REST API
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral:7b-instruct-q4_K_M",
    "messages": [{"role":"user","content":"Hello, world!"}],
    "stream": false
  }'

# Крок 4: Вимірювання TPS через ollama ps та /api/generate
ollama run mistral "Generate 200 words about MLOps" --verbose
```

---

## Slide 60 — Lab 2: Quantization Lab — порівняння моделей

```bash
# Завантажити різні кванти одної моделі
ollama pull mistral:7b-instruct-q4_K_M    # 4.1 GB
ollama pull mistral:7b-instruct-q8_0      # 7.7 GB (тільки якщо RAM дозволяє)
```

```python
# Python скрипт для порівняння (зберегти як compare_quants.py)
import requests, time, json

PROMPTS = [
    "What is the capital of France?",
    "Explain gradient descent in simple terms.",
    "Write a Python function to reverse a string.",
    # ... 7 more prompts
]

def query_ollama(model, prompt):
    start = time.perf_counter()
    resp = requests.post("http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False})
    elapsed = time.perf_counter() - start
    data = resp.json()
    tps = data.get("eval_count", 0) / max(data.get("eval_duration", 1) / 1e9, 0.001)
    return {"response": data["response"][:200], "tps": round(tps, 1), "time_s": round(elapsed, 2)}
```

---

## Slide 61 — Lab 3: LiteLLM Gateway — fallback логіка

```yaml
# config.yaml для LiteLLM proxy
model_list:
  - model_name: "chat"
    litellm_params:
      model: ollama/mistral:7b-instruct-q4_K_M
      api_base: http://localhost:11434
  - model_name: "chat"
    litellm_params:
      model: openrouter/mistralai/mistral-7b-instruct
      api_key: os.environ/OPENROUTER_API_KEY

router_settings:
  routing_strategy: "latency-based-routing"
  fallbacks: [{"ollama/mistral": ["openrouter/mistralai/mistral-7b-instruct"]}]
```

```bash
# Запуск proxy
litellm --config config.yaml --port 4000 --debug

# Тест: відправляємо запит через LiteLLM proxy (OpenAI-compatible)
curl http://localhost:4000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{"model":"chat","messages":[{"role":"user","content":"Hello!"}]}'
```

---

## Slide 62 — Lab 4: Load Testing з Locust

```python
# locustfile.py
from locust import HttpUser, task, between
import json, random

PROMPTS = ["Explain MLOps briefly.", "What is LLM quantization?",
           "How does batching improve throughput?"]

class LLMUser(HttpUser):
    wait_time = between(1, 3)
    host = "http://localhost:4000"   # LiteLLM proxy

    @task
    def chat_request(self):
        payload = {
            "model": "chat",
            "messages": [{"role": "user", "content": random.choice(PROMPTS)}],
            "max_tokens": 150
        }
        with self.client.post("/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer sk-1234"},
            catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")

# Запуск: locust -f locustfile.py --users 10 --spawn-rate 2
```

---

## Slide 65 — vLLM на GPU: production docker-compose

```yaml
# Docker-compose для vLLM (потребує NVIDIA GPU + CUDA 12+)
services:
  vllm:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
    ports:
      - "8000:8000"
    command: >
      --model mistralai/Mistral-7B-Instruct-v0.3
      --quantization awq
      --max-model-len 4096
      --max-num-seqs 256
      --enable-prefix-caching
      --tensor-parallel-size 1
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
```

```bash
# Тест
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mistralai/Mistral-7B-Instruct-v0.3","messages":[{"role":"user","content":"Hi"}]}'
```

*Ключові прапорці: `--quantization awq` — AWQ int4, зменшує VRAM з 14GB до ~5GB; `--enable-prefix-caching` — кешує спільні prefix (system prompt) між запитами; `--max-num-seqs` — max паралельних sequences у batch (continuous batching); `--tensor-parallel-size` — розподіл на кілька GPU (1 = один GPU).*

---

## Slide 66 — TGI (Text Generation Inference)

```bash
# TGI через Docker (потребує GPU)
docker run --gpus all -p 8080:80 \
  -v ~/.cache/huggingface:/data \
  -e HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id mistralai/Mistral-7B-Instruct-v0.3 \
  --quantize gptq \
  --max-concurrent-requests 128 \
  --max-batch-prefill-tokens 4096

# Streaming запит
curl http://localhost:8080/generate_stream \
  -H 'Content-Type: application/json' \
  -d '{"inputs":"What is MLOps?","parameters":{"max_new_tokens":200}}'
```

---

## Slide 70 — BentoML: LLM Service

```python
# service.py — BentoML LLM Service
import bentoml
from transformers import pipeline

@bentoml.service(
    resources={"memory": "8Gi"},
    traffic={"timeout": 60}
)
class LLMService:
    def __init__(self):
        self.model = pipeline(
            "text-generation",
            model="mistralai/Mistral-7B-Instruct-v0.3",
            device_map="auto"
        )

    @bentoml.api
    def generate(self, prompt: str) -> str:
        result = self.model(prompt, max_new_tokens=200, do_sample=False)
        return result[0]["generated_text"]

# Запуск: bentoml serve service:LLMService
# Build: bentoml build → створює Bento (Docker image з усім)
# Deploy: bentoml deploy → BentoCloud або Kubernetes
```
