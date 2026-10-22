# Week 2 Reflection — Transformers and Attention

## What I learned

The neuron → MLP → backpropagation refresher in the [notes](notes.md) turned out to matter more than I expected going in — I'd seen the Transformer feed-forward block described before, but seeing explicitly that it's "exactly `MLP(GELU)` applied position-wise" tied it directly back to the plain neural-network fundamentals instead of treating the Transformer as some separate, unrelated architecture. The Query/Key/Value framing (Query = what I'm looking for, Key = what I represent, Value = what I carry) finally gave me a mental model for attention instead of just a formula — `softmax(QKᵀ/√dₖ)·V` makes a lot more sense once each term has a concrete "role."

## What was challenging

The exit-ticket question "why scale by √dₖ" is one I could recite the slide's answer to ("prevents saturated softmax / vanishing gradients") but couldn't derive myself — I understand *that* large dot products push softmax into a near-one-hot, low-gradient regime, but I'd like to work through why the variance of `Q·K` grows specifically with `dₖ` rather than just taking it as given.

## How it connects to earlier weeks

This builds directly on Week 1. Attention operates on the token embeddings Week 1 introduced, and positional encoding is explicitly the fix for the "no word order" limitation of Bag-of-Words that Week 1's notes flagged (`"dog bites man" = "man bites dog"` as BoW vectors). Multi-head attention is also the mechanism that finally gives a model the contextual word sense that Week 1's static Word2Vec embeddings couldn't — the "bank" (river) vs. "bank" (financial) ambiguity from Week 1's notes is exactly what Week 2's attention mechanism is built to resolve by letting a word's representation depend on its context.

## Questions I still have

- The paper notes self-attention is `O(n²·d)` compute vs. `O(n·d²)` for RNNs — at what sequence length does that trade-off actually flip in practice, and is that part of why context-length limits exist on real LLMs?
- How do the different attention heads *actually* end up specializing (syntax, coreference, local context, long-range) during training — is that something that emerges purely from gradient descent, or is there any structural bias nudging heads toward different roles?

## What I'd try next

- Work through a tiny worked numerical example of scaled dot-product attention by hand (small Q/K/V matrices) rather than only reading the formula, to build real intuition before Week 3's "looking inside an LLM" material.
- Try visualizing real attention weights on a short sentence (the Week 2 slides mention BertViz) to see head specialization directly rather than just reading about it.
