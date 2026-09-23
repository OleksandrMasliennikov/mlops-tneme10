"""Measure Ollama TTFT and generation speed on CPU.

Run: python ttft_tps.py
Requires only Python's standard library and a running Ollama server.
"""

import json
import time
from urllib.request import Request, urlopen


URL = "http://127.0.0.1:11434/api/generate"
MODEL = "mistral:7b-instruct-q4_K_M"


def measure(prompt: str, output_limit: int) -> dict:
    """Read Ollama's JSON-lines stream and use its exact token counters."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": True,
        "keep_alive": "10m",
        "options": {"num_gpu": 0, "num_predict": output_limit, "temperature": 0},
    }
    request = Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    first_text_at = None
    final = None
    with urlopen(request, timeout=600) as response:
        for line in response:
            event = json.loads(line)
            if "error" in event:
                raise RuntimeError(event["error"])
            if event.get("response") and first_text_at is None:
                first_text_at = time.perf_counter()
            if event.get("done"):
                final = event

    if first_text_at is None or final is None:
        raise RuntimeError("No output text or final Ollama statistics received")
    count = final.get("eval_count", 0)
    duration_ns = final.get("eval_duration", 0)
    return {
        "input_tokens": final.get("prompt_eval_count", 0),
        "output_tokens": count,
        "ttft_ms": (first_text_at - started) * 1000,
        "tps": count * 1e9 / duration_ns if duration_ns else 0,
        "load_ms": final.get("load_duration", 0) / 1e6,
    }


def show_row(label: str, result: dict) -> None:
    print(
        f"| {label} | {result['input_tokens']} | {result['output_tokens']} "
        f"| {result['ttft_ms']:.1f} | {result['tps']:.2f} "
        f"| {result['load_ms']:.1f} |",
        flush=True,
    )


def main() -> None:
    # Load the model before measurements so a cold start does not skew TTFT.
    measure("Say hello.", 1)

    print("\nInput-length experiment: fixed output limit (120 tokens)")
    print("| Target input | Actual input tokens | Output tokens | TTFT ms | TPS | Load ms |")
    print("|---|---:|---:|---:|---:|---:|")
    for target in (50, 200, 500):
        # Targets are approximate; the tokenizer's actual count is printed.
        context = " ".join(f"item{i}: blue" for i in range(target // 4))
        prompt = (
            "Read this list, then write a detailed paragraph about the color blue. "
            "Keep writing until you reach the response limit. List: " + context
        )
        show_row(f"~{target}", measure(prompt, 120))

    print("\nOutput-length experiment: fixed input prompt")
    print("| Output limit | Input tokens | Actual output tokens | TTFT ms | TPS | Load ms |")
    print("|---|---:|---:|---:|---:|---:|")
    prompt = (
        "Write a continuous detailed explanation of how gradient descent works. "
        "Include practical examples and keep writing until the response limit."
    )
    for limit in (50, 200, 500):
        show_row(str(limit), measure(prompt, limit))


if __name__ == "__main__":
    main()
