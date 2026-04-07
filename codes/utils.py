import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def _call_llm(prompt, model, temperature=None):
    """Shared LLM inference logic with retry.

    Requires environment variables:
        OPENAI_API_KEY: API key for the OpenAI-compatible endpoint.
        OPENAI_BASE_URL: Base URL.
    """
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    if temperature is not None:
        kwargs["temperature"] = temperature

    max_retries = 10
    retry_delay = 1
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(**kwargs)
            return completion.choices[0].message.to_dict()
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise


def get_response(prompt):
    """LLM inference for detection / annotation tasks (env: DET_MODEL)."""
    model = os.getenv("DET_MODEL", "deepseek-reasoner")
    return _call_llm(prompt, model, temperature=0.0)


def get_response_gen(prompt):
    """LLM inference for text generation tasks (env: GEN_MODEL)."""
    model = os.getenv("GEN_MODEL", "gpt-4.1-mini")
    return _call_llm(prompt, model)


def write_jsonl_to_file(data, filename):
    """Write data to a JSONL file, auto-incrementing filename if it already exists."""
    base_filename, extension = os.path.splitext(filename)
    counter = 1

    new_filename = filename
    while os.path.exists(new_filename):
        new_filename = f"{base_filename}_{counter}{extension}"
        counter += 1

    with open(new_filename, 'w') as file:
        for d in data:
            file.write(json.dumps(d) + '\n')

    print(f"Data written to {new_filename}")