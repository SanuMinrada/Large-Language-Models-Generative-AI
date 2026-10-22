# Week 4 Reflection — Prompt Engineering and Fine-Tuning

## What I learned

The [notes](notes.md)' point that "fine-tuning is worth it when prompt engineering consistently
fails or the task needs domain-specific knowledge" went from abstract to very concrete once I
actually ran [Lab 4](lab/lab4_fine_tuning.ipynb): the untouched base model hedges on *every*
ByteFix question ("As an AI, I don't have real-time access to current prices...") because it
genuinely has no way to know a fictional shop's prices — no amount of clever prompting would fix
that, only giving it the information would, which is exactly the "domain-specific knowledge"
case from the slides. LoRA's mechanics also clicked properly for the first time:
`model.print_trainable_parameters()` showing only ~1.2% of 362M parameters as trainable made the
"freeze the base model, train a small low-rank update" idea tangible rather than diagrammatic.

## What was challenging

The genuinely hard part of this week wasn't the concepts, it was debugging my own Lab 4 code.
My first pass produced byte-for-byte identical "original" and "fine-tuned" answers, which looked
like the fine-tuning had done nothing — training loss had clearly dropped (3.18 → 2.17) so
*something* was learning, but the comparison said otherwise. Tracking it down (comparing raw
logits with and without the adapter via `disable_adapter()`) showed the adapter was in fact
changing predictions; the actual bug was that `PeftModel.from_pretrained(base_model, ...)`
mutates `base_model` in place, so my "original" and "fine-tuned" variables ended up pointing at
the same LoRA-modified model. I only noticed because I went looking for *why* the numbers didn't
match the loss curve, rather than accepting a plausible-looking result at face value — a good
reminder that "the code ran without errors" and "the code is correct" are different things.

## How it connects to earlier weeks

This directly extends Week 3: fine-tuning *is* another round of the same training loop from the
Lab 3 Transformer (forward pass → cross-entropy loss → backward pass → weight update) — just
starting from a pretrained checkpoint instead of random weights, and updating a small LoRA
subset instead of every parameter. The evaluation metrics section also closes a loop from both
Week 1 and Week 3: perplexity was already used to judge the Lab 3 Transformer, and this week adds
BLEU and BERTScore alongside it with an explicit caveat — none of them measure factual
correctness — which is precisely what the Lab 4 results demonstrate directly: BLEU went from 0.88
to 1.20 (a real improvement) while the fine-tuned model was *still* inventing USD price ranges
instead of recalling the actual trained LKR figures on unseen questions. Higher BLEU tracked
"sounds more like ByteFix," not "is factually correct about ByteFix."

## Questions I still have

- How many more training steps (or examples) would it typically take before a model like this
  starts reliably recalling specific trained facts (like exact prices) rather than just adopting
  the right tone and confidence level?
- Since the lecturer's own reference notebook has the same `base_model`-mutation bug, would
  fixing it there change their reported BLEU results (1.12 vs. 1.12) the same way it changed mine
  (0.88 vs. 1.20)?
- How would QLoRA's quantization change these results if I had GPU access — would the 4-bit
  base model's answers before fine-tuning look noticeably worse than this full-precision CPU run?

## What I'd try next

- Train for more epochs/steps on the same 32-example dataset to see at what point factual recall
  (not just tone) starts showing up in the generated answers.
- Try a slightly larger or more repetitive dataset (e.g. the same fact phrased three different
  ways) to see whether that helps the model actually retain specific numbers rather than just
  adopting the shop's confident voice.
- Add a BERTScore comparison alongside BLEU, since the Week 4 notes describe it as measuring
  semantic similarity rather than literal word overlap — potentially a fairer measure here, since
  the fine-tuned answers are stylistically right but still state different specific numbers than
  the reference.
