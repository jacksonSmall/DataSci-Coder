#!/usr/bin/env python3
"""Step 4: Model serving — CLI and FastAPI HTTP server.

Run on jarch. Access from Mac: ssh -L 8000:localhost:8000 jarch

Usage:
    python inference.py --mode cli      # Interactive CLI
    python inference.py --mode server   # FastAPI on port 8000
"""

import argparse
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
DEFAULT_ADAPTER = BASE_DIR / "outputs" / "lora_adapter"

# Must match the system prompt used during training (format_training_data.py)
SYSTEM_PROMPT = (
    "You are a data science assistant with deep expertise in machine learning, "
    "deep learning, NLP, computer vision, MLOps, and data engineering. "
    "You give clear, accurate explanations with practical examples and code "
    "when relevant. You cite best practices and explain trade-offs."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Model inference")
    parser.add_argument("--mode", choices=["cli", "server"], default="cli")
    parser.add_argument("--adapter", type=str, default=str(DEFAULT_ADAPTER))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Max new tokens (defaults to seq_len from training config)")
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def load_model(adapter_path: str):
    """Load 4-bit base model + LoRA adapter."""
    from unsloth import FastLanguageModel

    config_path = Path(adapter_path) / "training_config.json"
    seq_len = 512  # safe default matching training
    if config_path.exists():
        config = json.loads(config_path.read_text())
        seq_len = config.get("seq_length", 512)

    print(f"Loading model from {adapter_path} (seq_len={seq_len})...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print("Model loaded!")
    return model, tokenizer, seq_len


def generate_response(
    model, tokenizer, prompt: str,
    max_tokens: int = 512, temperature: float = 0.7,
) -> str:
    """Generate a response from the model."""
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
            repetition_penalty=1.25,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def run_cli(model, tokenizer, max_tokens, temperature):
    """Interactive CLI mode."""
    print("\nData Science Assistant (type 'quit' to exit)")
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
    from pydantic import BaseModel
    import uvicorn

    app = FastAPI(title="Data Science Assistant API")
    executor = ThreadPoolExecutor(max_workers=1)  # serialize GPU calls

    class Query(BaseModel):
        prompt: str
        max_tokens: int = max_tokens
        temperature: float = temperature

    class Response(BaseModel):
        response: str

    @app.post("/generate", response_model=Response)
    async def generate(query: Query):
        import logging
        logger = logging.getLogger("inference")

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
            logger.error("Generation timed out for prompt: %s", query.prompt[:100])
            return Response(response="Error: generation timed out. Try a shorter prompt or lower max_tokens.")
        except Exception as e:
            logger.error("Generation failed: %s", e)
            return Response(response=f"Error: generation failed ({type(e).__name__})")

        logger.info("Generated %d chars for prompt: %s", len(result), query.prompt[:80])
        return Response(response=result)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    print(f"\nStarting server on port {port}...")
    print(f"Access from Mac: ssh -L {port}:localhost:{port} jarch")
    uvicorn.run(app, host="0.0.0.0", port=port)


def main():
    args = parse_args()
    model, tokenizer, seq_len = load_model(args.adapter)

    # Default max tokens to training seq_len (don't generate beyond training range)
    max_tokens = args.max_tokens if args.max_tokens else seq_len

    if args.mode == "cli":
        run_cli(model, tokenizer, max_tokens, args.temperature)
    else:
        run_server(model, tokenizer, max_tokens, args.temperature, args.port)


if __name__ == "__main__":
    main()
