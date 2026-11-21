# Lab 04 — Fine-Tuning an LLM

## Laboratory Session

Lab 04 — Fine-Tuning an LLM using LoRA/QLoRA

## Objective

Fine-tune a small, instruction-tuned causal language model on a custom domain Q&A dataset using
LoRA, and compare the base vs. fine-tuned model's answers, per
[Lab sheet 4](resources/Lab%20sheet%204.pdf).

## Tasks and Implementation

Full implementation: [notebooks/lab4_fine_tuning.ipynb](notebooks/lab4_fine_tuning.ipynb)

1. Checked GPU availability (none available — CPU only in this environment; the notebook
   documents what this changes vs. the lab sheet's GPU/QLoRA assumption).
2. Chose `HuggingFaceTB/SmolLM2-360M-Instruct` (362M parameters) as a CPU-feasible model.
3. Built an original 32-example Q&A dataset about a fictional computer repair shop, "ByteFix
   Computer Repair & Accessories" (prices, policies, services — all invented specifically for
   this exercise, so the base model has no way to already know them).
4. Configured LoRA via `peft.LoraConfig` (rank 8, targeting the Q/K/V/O and MLP projection
   matrices).
5. Fine-tuned with `trl.SFTTrainer` for 5 epochs (20 steps).
6. Saved the LoRA adapter.
7. Compared the base model vs. the fine-tuned model on both a seen and several genuinely unseen
   questions, and scored a held-out evaluation set with BLEU.

The lecturer's [fine_tune_generative_models.ipynb](resources/fine_tune_generative_models.ipynb)
was used only as a reference for the overall workflow shape — the model, domain, dataset, and
Q&A pairs are original.

Conceptual background: [notes.md](notes.md) (prompt engineering, LoRA/QLoRA, evaluation metrics).

## Results and Evidence

- `model.print_trainable_parameters()`: only ~1.2% of the model's 362M parameters were trainable
  (LoRA).
- Training loss fell from 3.18 to 2.17; token-prediction accuracy rose from 49% to 60% over 5
  epochs.
- **BLEU score:** 0.88 (original model) → 1.20 (fine-tuned model) on a held-out evaluation set.
- The fine-tuned model drops the base model's generic hedging ("As an AI, I don't have real-time
  access to current prices...") and answers directly and confidently in the shop's voice, though
  it still doesn't reliably recall the exact trained prices on genuinely unseen questions.

*(Screenshots to be added here.)*

## Observations

- A real bug was caught and fixed along the way: an early version of the model-comparison step
  reused the same `base_model` object that `PeftModel.from_pretrained` had already wrapped,
  making the "original" vs. "fine-tuned" comparison meaningless (both pointed at the same
  LoRA-modified model, producing identical outputs). A logit-level check confirmed the adapter
  genuinely was shifting predictions, and the fix (loading two fully independent model instances)
  produced a comparison that actually meant what it claimed to mean. The lecturer's own reference
  notebook has this same bug, with the same identical-output symptom.
- With the fix, the before/after difference is real: the fine-tuned model clearly picked up the
  shop assistant's confident *style* well before it picked up the specific *facts* — a concrete
  illustration of the notes' point that BLEU/perplexity "don't directly measure factual
  correctness."

Full reflection: [reflection.md](reflection.md).

## Learning Outcomes

- Fine-tuned a pretrained causal LLM with LoRA, end-to-end, on CPU only.
- Learned to read `print_trainable_parameters()` output and connect it to the "freeze the base
  model, train a small low-rank update" idea from the lecture material.
- Debugged a subtle PEFT/model-object-mutation bug using a logit-level comparison rather than
  trusting a plausible-looking (but wrong) result.
- Used BLEU to get a quantitative measure of fine-tuning progress, while directly observing its
  limitation: it tracks word-overlap/style, not factual correctness.
