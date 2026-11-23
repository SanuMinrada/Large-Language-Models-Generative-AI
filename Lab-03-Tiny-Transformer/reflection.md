# Week 3 Reflection — LLM Internals and Pre-training

## What I learned

Building and training a tiny Transformer from scratch made the whole pipeline (tokenizer →
Transformer blocks → output) feel real instead of just a diagram. The validation loss dropped
from 4.46 to 1.56 over 15 epochs, and perplexity landed at 4.77, which gave me an actual number
to attach to "the model is learning." The generated text is still mostly gibberish, but it's
gibberish with word-shaped chunks and roughly sentence-like punctuation, which is clearly
different from the random output the untrained model gave. Loading the saved model into a Gradio
chat window and actually sending it a message made it feel like a finished thing rather than just
notebook output. The BERT vs GPT vs T5 comparison also made a lot more sense once I'd implemented
the causal mask myself — I could see exactly what "causal" means in code, not just as a label.

## What was challenging

Understanding what a "good" perplexity score actually means took some thought — 4.77 is only good
because the vocabulary is 75 characters, so I can't compare it directly to perplexity scores for
real word-level LLMs. I also assumed at first that an encoder-decoder model (like T5) would be
stronger than a decoder-only one since it has both parts, but it turns out decoder-only models
dominate mainly because they're cheaper and faster to run at scale (KV caching, no extra
cross-attention), not because they're inherently smarter.

## How it connects to earlier weeks

This lab reuses the Week 2 mechanisms directly: GELU, residual connections, LayerNorm, and
multi-head attention, just with a causal mask added on top to make it GPT-style instead of
bidirectional. It also ties back to Week 1's tokenization discussion — I used character-level
tokens here (the simplest option), and felt directly why sub-word tokenization exists: with only
75 characters in the vocabulary, every word takes many tokens to generate.

## Mechanisms that improve performance

The things already helping this tiny model: masked multi-head self-attention (so tokens can
relate to each other), residual connections + LayerNorm (keep gradients stable through the
layers), GELU (smoother gradients than ReLU), and AdamW (adaptive learning rates). The capacity
experiment proves this isn't just theory — removing a layer and a head made the loss and the
generated text noticeably worse.

If I scaled this up, a few things would help further: sub-word tokenization instead of
character-level, more training data and steps, a longer context window, a learning-rate
warmup/decay schedule instead of a flat rate, some dropout, and more heads/layers.

## Questions I still have

- Would a sub-word tokenizer actually speed up learning, or just produce different kinds of errors?
- Would the smaller (1-layer, 1-head) model eventually catch up with more training steps, or is there a hard capacity ceiling?
- How exactly does KV-caching work during generation? My `generate()` function recomputes everything each step, so I get *why* caching would help but not the exact mechanics.

## What I'd try next

- Swap in a small trained sub-word tokenizer and compare results directly.
- Increase `block_size` beyond 64 characters to see if more context improves story coherence.
