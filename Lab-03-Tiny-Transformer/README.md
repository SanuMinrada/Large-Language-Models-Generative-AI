# Lab 03 — Tiny Transformer

## Laboratory Session

Lab 03 — Developing a Tiny Transformer Language Model

## Objective

Build, train, evaluate, and save a Transformer from scratch, per
[Lab sheet 3](resources/Lab%20sheet%203.pdf) — applying the attention mechanism and Transformer
architecture (Query/Key/Value, scaled dot-product and multi-head attention, positional encoding)
covered in the lecture material.

## Tasks and Implementation

Full implementation: [notebooks/lab3_tiny_transformer.ipynb](notebooks/lab3_tiny_transformer.ipynb)

1. Loaded the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories) (1,000 stories; 900 train / 100 validation).
2. Tokenized at the character level (75-character vocabulary) and verified encode/decode round-tripping.
3. Prepared input/target sequence pairs for next-character prediction (`block_size=64`).
4. Built a causal Transformer from scratch: token + positional embeddings, masked multi-head
   self-attention (causal mask for autoregressive generation), GELU feed-forward layer, residual
   connections, and LayerNorm.
5. Initialized the model (80,715 parameters), AdamW optimizer, cross-entropy loss.
6–7. Trained for 15 epochs (100 steps/epoch), recording train/validation loss and plotting the
   learning curve.
8. Evaluated validation loss and perplexity.
9. Generated sample stories from prompts (e.g. "Once upon a time").
10. Ran a capacity experiment: trained a smaller variant (1 layer, 1 head) for the same number of
    steps and compared loss/perplexity/generated text against the baseline (2 layers, 2 heads).
11. Saved the trained model checkpoint, and built (without launching) a Gradio chat interface.

Conceptual background: [attention-theory-notes.md](attention-theory-notes.md) (Query/Key/Value,
scaled dot-product and multi-head attention, the Vaswani et al. 2017 paper) and
[llm-internals-notes.md](llm-internals-notes.md) (tokenizer → Transformer blocks → LM head
forward pass, autoregressive generation, causal masking).

## Results and Evidence

- Validation loss fell from 4.46 (untrained) to 1.56 over 15 epochs; final validation perplexity: **4.77**.
- Capacity experiment: baseline (2 layers, 2 heads) reached validation loss 1.563, vs. 1.768 for
  the smaller (1 layer, 1 head) variant trained for the same number of steps — and its generated
  text was visibly less coherent.
- Full loss curves, generated story samples, and the capacity-comparison output are in the
  notebook's executed cells.

*(Screenshots / additional results to be added here.)*

## Observations

- Building and training the Transformer from scratch made the forward-pass pipeline (tokenizer →
  Transformer blocks → LM head → sampling) feel concrete rather than diagrammatic — watching
  validation loss fall from roughly `ln(75)` (uniform-guessing baseline for a 75-character
  vocabulary) down to 1.56 gave an actual number to attach to "the model is learning structure."
- Interpreting the perplexity number in context took some thought: knowing 4.77 is "good" requires
  knowing the ceiling (75, for random guessing over this vocabulary) and that it's character-level,
  not directly comparable to perplexities quoted for real word/sub-word-level LLMs.
- The capacity experiment (step 10) gave a direct, measurable answer to "does fewer heads/layers
  actually matter": yes — worse loss and visibly less coherent text for the smaller model under
  identical training budgets.

Full reflections: [reflection.md](reflection.md) (Lab 3 / LLM internals) and
[attention-theory-reflection.md](attention-theory-reflection.md) (attention mechanism theory).

## Learning Outcomes

- Implemented masked multi-head self-attention, positional encoding, and a GELU feed-forward
  block from scratch using `nn.TransformerEncoderLayer` with a causal mask.
- Ran a full training loop (forward pass → cross-entropy loss → backward pass → weight update)
  and learned to read loss curves and perplexity as concrete evidence of learning.
- Directly observed the effect of model capacity (layers/heads) on both quantitative loss and
  qualitative generation quality.
- Connected the "Attention Is All You Need" theory (Q/K/V, scaled dot-product attention, why
  scale by √dₖ) to a working implementation, rather than treating it as a slide-only formula.
