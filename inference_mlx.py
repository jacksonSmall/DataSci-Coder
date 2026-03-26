#!/usr/bin/env python3
"""Inference using MLX on Apple Silicon.

Usage:
    python inference_mlx.py --mode cli                    # Interactive CLI
    python inference_mlx.py --mode server                 # FastAPI on port 8000
    python inference_mlx.py --adapter outputs/mlx_adapter # Custom adapter path
"""

import argparse
import json
from pathlib import Path

from mlx_lm import load, generate

BASE_DIR = Path(__file__).parent
DEFAULT_ADAPTER = BASE_DIR / "outputs" / "mlx_coder_adapter"
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

SYSTEM_PROMPT = (
    "You are an expert data science coding assistant. "
    "Return clean, runnable Python code with inline comments. "
    "No explanations outside code blocks unless asked."
)


def parse_args():
    parser = argparse.ArgumentParser(description="MLX inference")
    parser.add_argument("--mode", choices=["cli", "server"], default="cli")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=str, default=str(DEFAULT_ADAPTER))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def load_model(model_name: str, adapter_path: str):
    """Load model with LoRA adapter via mlx-lm."""
    if adapter_path is not None:
        adapter = Path(adapter_path)
        if not adapter.exists():
            print(f"Warning: adapter path {adapter_path} not found, loading base model only")
            adapter_path = None

    print(f"Loading model {model_name}...")
    if adapter_path:
        print(f"  with adapter: {adapter_path}")
    model, tokenizer = load(model_name, adapter_path=adapter_path)
    print("Model loaded!")
    return model, tokenizer


def make_sampler(temperature: float = 0.7, top_p: float = 0.9):
    """Create a sampler function for mlx-lm generate."""
    from mlx_lm.sample_utils import make_sampler as _make_sampler
    return _make_sampler(temp=temperature, top_p=top_p)


def make_logits_processor(repetition_penalty: float = 1.25):
    """Create a repetition penalty logits processor."""
    from mlx_lm.sample_utils import make_logits_processors
    return make_logits_processors(repetition_penalty=repetition_penalty)


def generate_response(
    model, tokenizer, prompt: str,
    max_tokens: int = 1024, temperature: float = 0.7,
) -> str:
    """Generate a response from the model."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    sampler = make_sampler(temperature=temperature)
    logits_processors = make_logits_processor(repetition_penalty=1.25)

    response = generate(
        model, tokenizer,
        prompt=input_text,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        verbose=False,
    )

    return response.strip()


def run_cli(model, tokenizer, max_tokens, temperature):
    """Interactive CLI mode."""
    print("\nData Science Assistant — MLX (type 'quit' to exit)")
    print("-" * 50)

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not prompt:
            continue

        response = generate_response(
            model, tokenizer, prompt, max_tokens, temperature
        )
        print(f"\nAssistant: {response}")


def run_server(model, tokenizer, max_tokens, temperature, port):
    """FastAPI HTTP server mode."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from fastapi import FastAPI
    from pydantic import BaseModel as PydanticModel
    import uvicorn

    app = FastAPI(title="Data Science Assistant API (MLX)")
    executor = ThreadPoolExecutor(max_workers=1)

    class Query(PydanticModel):
        prompt: str
        max_tokens: int = max_tokens
        temperature: float = temperature

    class Response(PydanticModel):
        response: str

    @app.post("/generate", response_model=Response)
    async def gen(query: Query):
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    generate_response,
                    model, tokenizer, query.prompt, query.max_tokens, query.temperature,
                ),
                timeout=120,
            )
        except asyncio.TimeoutError:
            return Response(response="Error: generation timed out.")
        except Exception as e:
            return Response(response=f"Error: {type(e).__name__}")
        return Response(response=result)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    print(f"\nStarting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)


def main():
    args = parse_args()
    model, tokenizer = load_model(args.model, args.adapter)

    if args.mode == "cli":
        run_cli(model, tokenizer, args.max_tokens, args.temperature)
    else:
        run_server(model, tokenizer, args.max_tokens, args.temperature, args.port)


if __name__ == "__main__":
    main()
