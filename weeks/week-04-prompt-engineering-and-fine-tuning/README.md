# Week 4 — Prompt Engineering and Fine-Tuning

From steering an LLM purely through prompts to actually updating its weights: prompt engineering
techniques, when fine-tuning is worth it over prompting, parameter-efficient fine-tuning methods
(LoRA/QLoRA), and how generated text is evaluated.

## Key concepts

- **Prompt engineering** — instruction prompts, output indicators, specificity/hallucination/order
  techniques, advanced prompt components (persona, context, format, audience, tone, data)
- **In-context learning** — zero-shot, one-shot, and few-shot prompting
- **Chain prompting** — breaking a task into a sequence of connected prompts
- **Fine-tuning vs. prompt engineering** — when fine-tuning is actually worth the cost
- **Supervised Fine-Tuning (SFT)** and the pretraining → SFT → preference-tuning pipeline
- **Parameter-Efficient Fine-Tuning (PEFT)** — Adapters, LoRA (low-rank weight updates), QLoRA
  (LoRA + quantization)
- **Evaluating generative models** — perplexity, BLEU, BERTScore, and what each does/doesn't measure

## Materials

- [Week 4 lecture slides](resources/Week%204.pdf)
- [Lab sheet 4 — Fine-Tuning an LLM using LoRA/QLoRA](resources/Lab%20sheet%204.pdf)
- [fine_tune_generative_models.ipynb — lecturer's reference notebook (reference only, not my work)](resources/fine_tune_generative_models.ipynb)

## My work

- [Notes](notes.md)
- [Lab work — fine-tuning a small LLM with LoRA](lab/lab4_fine_tuning.ipynb)
- [Reflection](reflection.md)
