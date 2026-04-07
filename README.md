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
- `fetch_runpod_file.py` — helper for pulling files from a RunPod instance.
- `gistfile1.txt` — training corpus.
- `main.py` — small entry point.

## Running

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
PROMPT_MODE=plain TMPDIR=/dev/shm /workspace/venv/bin/python /root/infer_loop.py

Enter your text (Type 'E' on a new line to Enter or exit):
who was Visgar?
E
Infer output ['who was Visgar? He was a captivating figure, a combination of
the wisdom of an elder and the energy of a rejuvenated Jarkell. His name
echoed through the halls of Pentiagon, and his stories had an allure that was
hard to resist. But behind his tales hid a treacherous agenda. He sought to
sow discord, to disrupt the balance that had been so recently threatened.
And he had the power to do it. As he delved deeper, he uncovered the
underbelly of Pentiagon. The secrets of his past and the alliances that had
brought him here. And it was in this landscape that he found Draven. The
bond between the two was immediate and intense. Draven, with his vision of
a united world where science and magic coexisted, and with his charismatic
if misguided leadership, had quickly won the hearts of the people of
Pentiagon. And Visgar, with his tales of a vanished realm and the alliances
that had led to its destruction, had also captured their imagination.']
Enter your text (Type 'E' on a new line to Enter or exit):
```

Note how the model doesn't "answer" the question — it *continues* it as prose,
picking up proper nouns (`Jarkell`, `Pentiagon`) from the training corpus and
inventing new characters (`Visgar`, `Draven`) in the same narrative style.
This is the expected behavior of raw next-token fine-tuning.
