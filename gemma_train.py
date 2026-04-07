import os
os.environ["TORCH_DISABLE_FOREACH"] = "1"

from itertools import chain
from datasets import load_dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, PeftModel, get_peft_model
import locale
locale.getpreferredencoding = lambda: "UTF-8"
import logging

# Configure logging
train_log_path = os.environ.get("TRAIN_LOG_PATH", "training_logs.txt")
logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(train_log_path)])

# === Configuration ===
model_name = "google/gemma-4-E2B"
device_map = {"": 0}  # Load model on GPU 0
output_dir = os.environ.get("OUTPUT_DIR", "./results_chunked_128")
block_size = int(os.environ.get("BLOCK_SIZE", "128"))
local_files_only = os.environ.get("LOCAL_FILES_ONLY", "0") == "1"
dataset_cache_dir = os.environ.get("DATASET_CACHE_DIR")
base_adapter_dir = os.environ.get("BASE_ADAPTER_DIR")
num_train_epochs = int(os.environ.get("NUM_TRAIN_EPOCHS", "3"))
learning_rate = float(os.environ.get("LEARNING_RATE", "2e-4"))

# Training Parameters
train_params = {
    "num_train_epochs": num_train_epochs,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "learning_rate": learning_rate,
    "weight_decay": 0.001,
    "max_grad_norm": 0.3,
    "max_steps": -1,
    "warmup_ratio": 0.03,
    "optim": "adamw_torch",
    "lr_scheduler_type": "constant",
    "save_steps": 1000,
    "logging_steps": 10,
    "fp16": False,
    "bf16": True,
    "gradient_checkpointing": True
}


# Load Dataset
train_dataset = load_dataset("text", data_files="gistfile1.txt", cache_dir=dataset_cache_dir)["train"]
train_dataset = train_dataset.filter(lambda example: bool(example["text"].strip()))

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    extra_special_tokens={},
    local_files_only=local_files_only,
)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    texts = [text + tokenizer.eos_token for text in examples["text"]]
    return tokenizer(texts, add_special_tokens=False)


def group_texts(examples):
    concatenated_examples = {
        key: list(chain.from_iterable(examples[key]))
        for key in examples.keys()
    }
    total_length = len(concatenated_examples["input_ids"])
    total_length = (total_length // block_size) * block_size
    result = {
        key: [values[i:i + block_size] for i in range(0, total_length, block_size)]
        for key, values in concatenated_examples.items()
    }
    result["labels"] = [input_ids.copy() for input_ids in result["input_ids"]]
    return result


tokenized_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_dataset = tokenized_dataset.map(group_texts, batched=True)

# Load model in bf16
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map=device_map,
    dtype=torch.bfloat16,
    attn_implementation='eager',
    local_files_only=local_files_only,
)
model.config.use_cache = False

LORA_TARGET_SUFFIXES = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
}


def get_language_model_lora_targets(model):
    return [
        name
        for name, module in model.named_modules()
        if name.startswith("model.language_model.layers.")
        and name.rsplit(".", 1)[-1] in LORA_TARGET_SUFFIXES
        and isinstance(module, torch.nn.Linear)
    ]


if base_adapter_dir:
    model = PeftModel.from_pretrained(model, base_adapter_dir, is_trainable=True)
else:
    lora_target_modules = get_language_model_lora_targets(model)
    if not lora_target_modules:
        raise ValueError("No language-model LoRA target modules were found.")

    # LoRA - train only ~1% of parameters to fit in 24GB
    peft_config = LoraConfig(
        lora_alpha=32, lora_dropout=0.1, r=32, bias="none",
        task_type="CAUSAL_LM", target_modules=lora_target_modules
    )
    model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Workaround for CUDA _foreach_norm bug with Gemma 4
import torch.nn.utils as torch_utils
_orig_clip = torch_utils.clip_grad_norm_
def _patched_clip(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
    return _orig_clip(parameters, max_norm, norm_type=norm_type, error_if_nonfinite=error_if_nonfinite, foreach=False)
torch_utils.clip_grad_norm_ = _patched_clip

# Train Model
trainer = Trainer(
    model=model, train_dataset=tokenized_dataset,
    processing_class=tokenizer, args=TrainingArguments(output_dir=output_dir, **train_params)
)
trainer.train()
trainer.save_model()
