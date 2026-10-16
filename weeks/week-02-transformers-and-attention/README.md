# Week 2 — Transformers and Attention

Neural network foundations that underpin every LLM, then the Transformer architecture itself:
why self-attention replaced recurrence, and how the original "Attention Is All You Need"
paper put the pieces together.

## Key concepts

- **Neural network basics** — the neuron, activation functions (Sigmoid, Tanh, ReLU, GELU,
  SwiGLU) and why LLMs favour GELU/SwiGLU, MLP forward pass, backpropagation, the vanishing
  gradient problem
- **From MLP to Transformer** — why fixed-input MLPs and sequential RNNs don't scale, and how
  self-attention gives an O(1) path between any two tokens
- **Query, Key, Value** — the three roles in attention and what each represents
- **Scaled dot-product attention** — why scaling by √dₖ matters (prevents saturated softmax /
  vanishing gradients)
- **Multi-head attention** — how different heads learn different relationships (syntactic,
  coreference, local context, long-range)
- **Positional encoding** — why it's needed since self-attention is permutation-invariant
- **"Attention Is All You Need" (Vaswani et al., 2017)** — full architecture, BLEU results,
  and why self-attention was chosen over RNNs/CNNs

## Materials

- [Week 2 lecture slides](resources/Week%202.pdf)
- [Vaswani et al. (2017) — Attention Is All You Need](resources/1706.03762v7.pdf)
- [transformer_practice.py — hands-on toy Transformer build](resources/transformer_practice.py)
- [Lab sheet 2 — AI in Everyday Challenges (LLM product-development activity)](resources/Lab%20sheet%202.pdf)

## My work

- [Notes](notes.md)
- [Lab work](lab/)
- [Reflection](reflection.md)
