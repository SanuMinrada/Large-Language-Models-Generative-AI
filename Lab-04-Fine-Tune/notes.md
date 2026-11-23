# Week 4 Notes — Prompt Engineering and Fine-Tuning

## Key terms

- **Prompt** — the instruction or question given to an AI tool to get a response.
- **Foundation model (FM)** — a model trained on vast, diverse data that has learned general patterns and relationships, and can be adapted to many downstream tasks without retraining from scratch.
- **Prompt engineering** — designing prompts (or prompt templates) that guide a model's output; faster, cheaper, and reversible compared to fine-tuning.
- **Fine-tuning** — training a model further on labelled examples so it answers in a required style or performs a specialised task; worth it when prompt engineering keeps failing or the task needs domain-specific knowledge.
- **Adapters** — small extra modules inserted into a pretrained Transformer block (after the attention layer and the feed-forward network); the original weights stay frozen and only the adapter is trained.
- **LoRA (Low-Rank Adaptation)** — instead of adding adapter layers, LoRA fine-tunes a small low-rank subset of the base model's own weight matrices (e.g. just the Query and Value matrices), keeping everything else frozen.
- **QLoRA** — LoRA combined with quantization: the base model's weights are compressed to lower precision (e.g. 4-bit) to cut memory use, while LoRA still only trains a small set of extra parameters.
- **Perplexity** — how uncertain a model is when predicting the next token; lower is better, but it doesn't measure factual correctness.
- **BLEU** — compares generated text to a reference answer by word/sequence overlap; higher means more overlap, but it doesn't check whether the answer is actually correct.
- **BERTScore** — compares generated and reference text using contextual embeddings (precision/recall/F1), so it can catch paraphrases BLEU would miss, but still doesn't check factual correctness.

## Summary

### Prompt engineering basics
A basic prompt is just input text (e.g. "The sky is" → "blue"). An **instruction prompt** adds a separate instruction and data (e.g. "Classify the text into negative or positive." + "This is a great movie!"), and adding an **output indicator** (like "Sentiment:") helps guide the format of the answer. To improve prompt quality: be specific about what you want, tell the model to say "I don't know" rather than guess (reduces hallucination), and put the important instruction at the start or end of the prompt (models attend more to the beginning — primacy — or the end — recency — of a prompt).

### Advanced prompt components
A more complete prompt can specify: **persona** (who the LLM should act as), **instruction** (what to do), **context** (why the task matters), **format** (how the answer should look), **audience** (who it's for), **tone** (how it should sound), and **data** (what information to use).

### In-context learning and chaining
Instead of describing a task, you can just show examples: **zero-shot** (no examples), **one-shot** (one example), **few-shot** (multiple examples) — more examples usually means more consistent output. **Chain prompting** breaks a big task into a sequence of smaller prompts (e.g. generate a product name → generate a slogan from that name → generate a sales pitch from both), useful for response validation or multi-step writing tasks.

### Fine-tuning vs. prompt engineering
Prompt engineering is faster, cheaper, and reversible, but can plateau — e.g. an LLM given examples for classifying bank complaints may still label new complaints inconsistently. Fine-tuning is worth it when the task needs knowledge a general-purpose model doesn't have (e.g. a university's own exam rules).

### Supervised fine-tuning (SFT) and PEFT
Full fine-tuning updates all of a model's weights using labelled question-response data — effective, but needs a lot of compute, time, and storage. **Parameter-Efficient Fine-Tuning (PEFT)** avoids this: LoRA only trains a small low-rank subset of the weight matrices (e.g. rank 2 instead of full rank 10×10 cuts parameters from 100 to ~40), while the rest of the model stays frozen. QLoRA adds quantization on top, so the frozen base model also takes less memory — this is why LoRA/QLoRA make fine-tuning possible on much smaller hardware than full fine-tuning would need.

### Evaluating generative models
Classic word-level metrics compare generated text to a reference: perplexity (model's own confidence), BLEU (word/sequence overlap), and BERTScore (semantic similarity via embeddings). All three are useful signals, but none of them directly checks whether an answer is factually correct — that has to be judged separately (e.g. by checking the actual facts, as this lab's seen/unseen question comparison does).
