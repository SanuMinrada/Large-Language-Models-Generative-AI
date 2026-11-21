# Week 4 Notes — Prompt Engineering and Fine-Tuning

Source: [Week 4 lecture slides](resources/Week%204.pdf) — Prompt Engineering, Fine-Tuning an LLM, and Evaluating Generative Models.

## Key terms

- **Prompt** — the instruction or question given to an LLM to get a response.
- **Foundation model (FM)** — an AI model trained on vast, diverse datasets, adaptable to many downstream tasks without retraining from scratch.
- **Prompt engineering** — designing prompts (and prompt structure) to improve the quality of an LLM's generated output.
- **Instruction prompt** — a prompt made of an *instruction* plus the *data* the instruction applies to, often with an *output indicator* guiding the response format.
- **In-context learning** — showing the LLM examples of the task directly in the prompt instead of only describing it (zero-shot, one-shot, few-shot).
- **Chain prompting** — breaking a task into a sequence of connected prompts, each feeding into the next, rather than one large prompt.
- **Fine-tuning** — further training a pretrained model's weights on task/domain-specific labelled data, rather than only steering it via prompts.
- **Supervised Fine-Tuning (SFT)** — fine-tuning on labelled input→output pairs (e.g. instruction/question → answer).
- **PEFT (Parameter-Efficient Fine-Tuning)** — fine-tuning methods that update only a small subset of parameters instead of the whole model.
- **LoRA (Low-Rank Adaptation)** — a PEFT method that freezes the base model and trains small low-rank weight matrices added alongside selected layers (e.g. the Q/V projections).
- **QLoRA** — LoRA combined with quantizing the frozen base model's weights to a lower bit-width, cutting memory use further.
- **Perplexity / BLEU / BERTScore** — word-level metrics for evaluating generated text against a reference.

## Summary

### Prompt engineering

- A **prompt** is just the instruction/question given to an LLM (e.g. ChatGPT) to produce a response; **prompt engineering** is designing that prompt to get better output from a **foundation model** — a model pretrained on broad data and adaptable to many tasks without retraining.
- **Basic prompt:** input text directly continued by the model (e.g. `"The sky is"` → `"blue."`) — works, but doesn't target a specific task.
- **Instruction prompt:** splits the prompt into an **instruction** (what to do) and **data** (what to do it on) — e.g. *Instruction:* "Classify the text into negative or positive." *Data:* "This is a great movie!" This can be extended further with an **output indicator** (e.g. prefixing with `Text:` / `Sentiment:`) to guide the exact response format.
- **Instruction prompt use cases:** supervised classification, search, summarization, code generation, named entity recognition — i.e. most of the NLP task taxonomy from Week 1, now accessed through prompting instead of training a dedicated model.
- **Improving prompt output quality:**
  - **Specificity** — be precise about the task (e.g. "Write a description in less than two sentences, formal tone" rather than just "write a description").
  - **Hallucination mitigation** — explicitly ask the model to say "I don't know" rather than confidently inventing an answer when it isn't sure.
  - **Order** — LLMs weight the beginning (primacy effect) and end (recency effect) of a prompt more heavily, so key instructions should sit at the start or end, not buried in the middle.
- **Advanced prompt components:** Persona (who the LLM should act as), Instruction (what to do), Context (why/background), Format (how the answer should look), Audience (who it's for), Tone (how it should sound), Data (what information to use). A well-constructed prompt can combine all of these — e.g. "You are an academic research assistant... summarize this paper... for undergraduate students... in five headed sections... using the uploaded paper only, don't invent findings."
- **In-context learning:** instead of *describing* the task, *show* it — zero-shot (no examples), one-shot (one example), few-shot (multiple examples) prompting, each giving the model a worked pattern to imitate.
- **Chain prompting:** break a complex task into a sequence of smaller prompts, each one's output feeding the next (e.g. generate a product name → generate a slogan from that name → generate a sales pitch from both). Useful for response validation and multi-step generation tasks like story writing.

### Fine-tuning an LLM

- **Prompt engineering vs. fine-tuning:** prompt engineering is faster, cheaper, and fully reversible (just change the prompt), so it's the default first approach. **Fine-tuning** is worth it when prompt engineering *consistently* fails to produce the right output quality even with good examples, or when the task needs genuine domain-specific knowledge the base model doesn't have (e.g. a university's specific exam rules, or a shop's own product catalogue and policies).
- **Common fine-tuning methods:** the training pipeline runs **Untrained → Base model (self-supervised language modeling on unlabeled data) → Instruction-tuned (supervised fine-tuning on labelled data) → Preference-tuned (further tuning from human/AI preference signals)**.
- **Supervised Fine-Tuning (SFT) / full fine-tuning:** trains on labelled data (e.g. instruction → output pairs, like the `databricks-dolly-15k` dataset) and updates *all* of the model's weights — effective for learning domain-specific representations, but needs significant compute, time, and storage, which becomes a real problem once models reach GPT-3.5/GPT-4/LLaMA-3.1 scale (175B / ~1.8T / 405B parameters).
- **Parameter-Efficient Fine-Tuning (PEFT):** rather than updating every weight, PEFT methods train only a small number of added or selected parameters, keeping the pretrained weights **frozen**:
  - **Adapters** — small extra modules inserted after the attention layer and the feed-forward layer inside each Transformer block; only the adapter weights are trained.
  - **LoRA (Low-Rank Adaptation)** — instead of adding new layers, LoRA represents the *update* to a weight matrix as a product of two small low-rank matrices (e.g. a full-rank 10×10 matrix has 100 parameters, but a rank-1 decomposition needs only 20, rank-2 only 40). In practice this is often applied just to the Query/Value projection matrices in each Transformer layer. The original weights stay frozen; only this small low-rank update is trained and then added back in.
  - **QLoRA (Quantized LoRA)** — combines LoRA with **quantization** of the frozen base model's weights (e.g. representing them in fewer bits, trading numeric precision for memory). Naively quantizing can cause distinct nearby weight values to collapse onto the same lower-precision value, losing information — **blockwise quantization** addresses this by quantizing smaller blocks of weights separately with their own scale, so values that are "high" or "low" relative to their local block are preserved better. In short: **quantization reduces the memory needed to hold the pretrained model; LoRA reduces the number of parameters that need to be trained.**

### Evaluating generative models

Word-level metrics compare a reference text against generated text:

- **Perplexity** — measures how uncertain the model is when predicting the next token; it's essentially the gap between the model's predicted probability distribution and the actual next token. Lower perplexity = better prediction of the evaluation text, but it says nothing about factual correctness or answer quality (this directly matches the perplexity used in the Week 3 Lab — same metric, now framed as an evaluation tool for generative models in general).
- **BLEU (Bilingual Evaluation Understudy)** — measures word/phrase overlap between a generated answer and a reference answer; higher BLEU means more word-sequence overlap, but — like perplexity — it doesn't verify that the answer is factually correct, and it penalizes answers that are phrased differently even when they mean the same thing.
- **BERTScore** — measures *semantic* similarity using contextual embeddings from a pretrained model (e.g. BERT) rather than literal word overlap, so paraphrases like "The child is happy" vs. "The kid is joyful" score well despite sharing few exact words. Reports Precision, Recall, and F1. Like BLEU, it doesn't check factual correctness — only how semantically similar the generated text is to the reference.

## Code / examples

- Basic prompt: `"The sky is"` → `"blue."`
- Instruction prompt with output indicator: `Classify the text into negative or positive. Text: "This is a great movie!" Sentiment:` → `"Positive"`
- Few-shot prompt: showing 2–3 labelled `Text: ... / Sentiment: ...` examples before asking the model to label a new, unlabelled one.
- LoRA parameter count example: a 10×10 weight matrix has 100 parameters full-rank; a rank-1 LoRA update needs only 20, rank-2 needs 40 — a large reduction in trainable parameters relative to full fine-tuning.
- `fine_tune_generative_models.ipynb` in this week's resources (lecturer-provided, reference only) demonstrates the full pipeline end-to-end: load `Qwen/Qwen2.5-0.5B-Instruct`, build a small Q&A dataset, configure LoRA via `peft.LoraConfig`, fine-tune with `trl.SFTTrainer`, then compare the base vs. fine-tuned model's answers and score them with `sacrebleu`.

## Things to revisit

- Working through *why* blockwise quantization specifically fixes the "similar weights collapse together" problem, rather than just accepting the slide's diagram.
- Trying the same instruction prompt with and without the "say I don't know if unsure" hallucination-mitigation instruction, to see how much it actually changes behaviour in practice.
- Understanding exactly what LoRA's `target_modules` (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) correspond to inside a Transformer block, tying this back to the Week 2 attention/feed-forward notes.
