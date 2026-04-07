import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_ID = os.environ.get("BASE_ID", "google/gemma-4-E2B")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "/root/results_expanded_pass2")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "120"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.8"))
TOP_P = float(os.environ.get("TOP_P", "0.95"))
DO_SAMPLE = os.environ.get("DO_SAMPLE", "1") != "0"
PROMPT_MODE = os.environ.get("PROMPT_MODE", "plain").lower()
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "You are a helpful assistant.")
TURN_START = "<|turn>"
TURN_END = "<turn|>"
SYSTEM_CHAT_TEMPLATE = f"{TURN_START}system\n{{prompt}}{TURN_END}\n"
USER_CHAT_TEMPLATE = f"{TURN_START}user\n{{prompt}}{TURN_END}\n"
MODEL_CHAT_TEMPLATE = f"{TURN_START}model\n{{prompt}}{TURN_END}\n"
MODEL_TURN_PREFIX = f"{TURN_START}model\n"


def build_chat_prompt(prompt):
    parts = []
    if SYSTEM_PROMPT:
        parts.append(SYSTEM_CHAT_TEMPLATE.format(prompt=SYSTEM_PROMPT))
    parts.append(USER_CHAT_TEMPLATE.format(prompt=prompt))
    parts.append(MODEL_TURN_PREFIX)
    return "".join(parts)


def prepare_inputs(tokenizer, device, prompt):
    if PROMPT_MODE == "chat":
        if hasattr(tokenizer, "apply_chat_template"):
            messages = []
            if SYSTEM_PROMPT:
                messages.append({"role": "system", "content": SYSTEM_PROMPT})
            messages.append({"role": "user", "content": prompt})
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted_prompt = build_chat_prompt(prompt)
    else:
        formatted_prompt = prompt
    return tokenizer(formatted_prompt, return_tensors="pt").to(device)


def generate_text(model, tokenizer, prompt):
    inputs = prepare_inputs(tokenizer, model.device, prompt)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE if DO_SAMPLE else None,
            top_p=TOP_P if DO_SAMPLE else None,
        )

    generated_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def main():
    initial_prompt = sys.argv[1] if len(sys.argv) > 1 else None

    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_ID,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    print("Interactive inference ready.")
    print(f"Prompt mode: {PROMPT_MODE}")
    print("Type a prompt and press Enter.")
    print("Type 'exit' or 'quit' to stop.")

    if initial_prompt:
        print("\nPrompt:", initial_prompt)
        print("\nOutput:\n")
        print(generate_text(model, tokenizer, initial_prompt))
        print()

    while True:
        try:
            prompt = input("\nPrompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        print("\nOutput:\n")
        print(generate_text(model, tokenizer, prompt))


if __name__ == "__main__":
    main()
