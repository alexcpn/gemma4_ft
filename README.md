# Google Gemma 4 Fine tuning

LoRA fine-tuning of `google/gemma-4-E2B` as a **raw causal language model** on a
plain-text corpus. This is **next-token (continued pre-training) fine-tuning — not
instruction tuning**. There is no chat template, no prompt/response split, and no
loss masking: the model simply learns to continue the text in `gistfile1.txt`.

## What the training does

`gemma_train.py` performs the following:

1. Loads `gistfile1.txt` as a `datasets` text dataset (one line = one example)
   and drops empty lines.
2. Tokenizes each line with an appended `eos_token`, **without** adding any
   special/BOS tokens.
3. Concatenates all tokens and splits them into fixed-size contiguous chunks of
   `BLOCK_SIZE` tokens (default `128`). Each chunk becomes one training example
   with `labels = input_ids` — standard next-token prediction where the loss is
   computed on every token in the window.
4. Loads the base model in bf16 on GPU 0 with eager attention.
5. Attaches a LoRA adapter (`r=32`, `alpha=32`, `dropout=0.1`) to the language
   model's attention and MLP projections
   (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`),
   restricted to modules under `model.language_model.layers.*`. Only the adapter
   weights are trained. If `BASE_ADAPTER_DIR` is set, training resumes from an
   existing adapter instead of creating a new one.
6. Trains with the HuggingFace `Trainer`: bf16, gradient checkpointing,
   `adamw_torch`, constant LR schedule, warmup ratio `0.03`, weight decay
   `0.001`, grad clip `0.3`, batch size 1, no gradient accumulation.
7. A monkey-patch forces `clip_grad_norm_` to use `foreach=False` as a
   workaround for a CUDA `_foreach_norm` bug with Gemma 4.
8. Saves the final adapter to `OUTPUT_DIR`.

Because the data is unstructured narrative prose and the model sees only raw
text, the resulting adapter biases the base model toward the style and facts of
the corpus — it does **not** teach it to follow instructions or respond in a
chat format.

## Dataset

`gistfile1.txt` — a small collection of short fictional stories (e.g. the
Kingdom of Jarkell / Pentiagon tales). Each line is treated as a document,
tokenized independently, then concatenated and chunked.

## Files

- `gemma_train.py` — the raw next-token LoRA training script described above.
- `gemma_sft_train.py` — alternative SFT-style training script (separate entry
  point, not used by the raw pipeline).
- `gemma_eval.py` — interactive prompt loop that loads the base model and
  merges a saved LoRA adapter for generation.
- `infer_loop.py` — inference loop utility.
- `gistfile1.txt` — training corpus.
- `main.py` — small entry point.

## Running

Training was done in a RunPod Instance **GPU: RTX 3090 24GB**;

For google/gemma-4-E2B, the rough memory picture is:

- weights in bf16: about 10.3 GB
- gradients for full training: about another 10.3 GB
- AdamW optimizer state: often about 2 x fp32 per parameter, roughly 41 GB
- activations: extra on top, depending on sequence length and batch size

So full fine-tuning would be far beyond 24 GB. LoRA avoids that by freezing the base 5.17B parameters and only training a small adapter, which in the run was about 50.7M trainable params, or 0.98% of the model.

That is why it fit on a 3090 24GB:

- base model stays loaded
- only adapter params get gradients
- only adapter params get optimizer state


```bash
# Optional environment overrides
export OUTPUT_DIR=./results_chunked_128
export BLOCK_SIZE=128
export NUM_TRAIN_EPOCHS=3
export LEARNING_RATE=2e-4
# export BASE_ADAPTER_DIR=./results_chunked_128   # to continue from an adapter
# export LOCAL_FILES_ONLY=1                       # to use only cached weights
# export DATASET_CACHE_DIR=/path/to/hf_cache
# export TRAIN_LOG_PATH=training_logs.txt

python gemma_train.py
```

The script expects a single GPU on device 0 with enough memory for bf16
Gemma-4-E2B plus LoRA (designed to fit in ~24 GB).

## Evaluating

After training, point `gemma_eval.py`'s `output_dir` at your adapter checkpoint
and run it for an interactive completion loop. Note: prompts are fed to the
model as **raw text continuations**, not chat turns — ask for completions, not
instructions.

## Training runs

**Run 1 — first clean LoRA pass on the expanded corpus.**
Finished at `train_loss = 3.378` over 177 steps. The adapter was clearly
learning the story world (proper nouns, setting, tone), but generations were
still loose and frequently drifted into generic fantasy or generic
world-knowledge completions.

**Run 2 — continuation from Run 1's adapter.**
Started from the Run 1 adapter (`BASE_ADAPTER_DIR`), used a larger corpus and
more epochs, and finished at `train_loss = 0.8109` over 405 steps. Logged
training windows frequently dropped into roughly `0.2 – 0.7`, meaning the
model fit the training distribution much more strongly.

For in-domain prompts this helped a lot: story-style continuations using names
like **Elorna**, **Igodo**, **Visgar**, **Jarkell**, and **Pentiagon** became
much more consistent and stayed inside the corpus's narrative voice. However,
broad question-style prompts such as *"what did Visgar do?"* still fail,
because the model was trained as **plain next-token story continuation on raw
text, not as a chat or Q&A model**. The low loss means it learned the text
*format* of the corpus better — it does **not** mean it became generally good
at answering arbitrary questions about the story.

### Sample output

Running the inference loop against a trained adapter:

```text
root@99658f6b192d:/# PROMPT_MODE=plain TMPDIR=/dev/shm /workspace/venv/bin/python /root/infer_loop.py
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|█████████████████████| 2011/2011 [00:02<00:00, 948.82it/s]
Interactive inference ready.
Prompt mode: plain
Type a prompt and press Enter.
Type 'exit' or 'quit' to stop.

Prompt> Elorna, with her inherent curiosity

Output:

and dedication, pursued a degree in Biochemistry at the University of California, San Diego. Her passion for science led her to conduct research in renowned laboratories such as those of Dr. Kihara and Dr. Ouchi, where she gained valuable insights into the intricacies of cellular biology and the immune system.

After completing her studies, Elorna embarked on a journey of discovery and adventure. She explored ancient ruins, ventured into uncharted territories, and delved into the secrets of ancient civilizations. Her insatiable thirst for knowledge led her to study martial arts, martial arts, and eventually, the enigmatic realm of magic

Prompt> Elorna loved

Output:

everything about her life, except one thing. She never felt like she belonged anywhere. A recent promotion in the prestigious royal family brought her to the capital city of Pentiagon, where she found herself surrounded by people who seemed to know each other's secrets and histories. Worse, she had no sense of purpose or belonging.

Determined to find a way to integrate herself into this new world, Elorna embarked on a quest to uncover the secrets of the royal family. Her journey led her to the heart of the palace, where she stumbled upon an ancient scroll. It spoke of a prophecy

Prompt> "Time," Lysandra began,

Output:

"is a curious thing. It bends with purpose."

I blinked, not understanding what she meant.

"Think about it," Lysandra continued. "When you look at a flower, you see a beautiful flower. When you look at a clock, you see a machine that measures time. But when you look at a person, you see a collection of cells, tissues, and organs that have their own rhythms and purposes."

I tried to wrap my mind around this concept. How could a clock, a flower, and a person all be seen as objects with their own unique properties and

Prompt> The Trial of Draven

Output:

Wood, Part 3: The Final Verdict

After months of investigation, the Royal Commission found no evidence linking Draven Wood to the murder of Dr. Anthon Sylvan. However, they did uncover a disturbing pattern of secrecy and manipulation within the royal court.

The commission revealed that Draven Wood had been involved in a conspiracy to frame Dr. Sylvan for the crime. The evidence was overwhelming: a forged letter, a coerced confession, and a royal decree signed by Draven Wood himself.

The royal court, outraged by the revelations, demanded justice. Draven Wood was arrested and
```

Note how the model doesn't "answer" the question — it *continues* it as prose,picking up proper nouns (`Jarkell`, `Pentiagon`) from the training corpus. Many places it deviates wildly.This is the expected behavior of raw next-token fine-tuning.
Limitations could be due to fewer number of epochs and smaller data set. Instruction Fine Tuning results are usually much better.

Note: It took a lot of tries to get the Transformer version, Cuda Version  etc updated in the RunPod (using AI Agents itself tool a lot of tokens) to get the latest Gemma 4 training loop working. To even train a small model without LoRA you need about 40 GB or more of GPU RAM. So this path is not for the common use case. Much easier is to use RAG or just search with Agentic AI
