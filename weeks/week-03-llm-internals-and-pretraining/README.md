# Week 3 — LLM Internals and Pre-training

A look inside a working Transformer LLM — tokenizer, LM head, autoregressive generation,
caching — followed by how models are actually pre-trained at scale and how the major
architecture families (BERT, GPT, T5) and modern LLMs differ.

## Key concepts

- **Transformer LLM forward pass** — tokenizer, stack of Transformer blocks, LM head
- **Autoregressive generation** — one token at a time, each step appending the previous output
  back into the prompt
- **Decoding/sampling** — greedy decoding vs. random sampling from the output probability
  distribution
- **Context length & KV caching** — why caching avoids recomputation across generation steps
- **Attention vs. feedforward sub-layers** — what each part of a Transformer block does
- **Encoder vs. decoder** — BERT (encoder, bidirectional) vs. GPT (decoder, autoregressive)
- **Pre-training objectives** — Masked Language Modeling (BERT), Causal Language Modeling
  (GPT), Span Corruption (T5)
- **BERT vs. GPT vs. T5** — architecture, pre-training strategy, and best-fit tasks compared
- **The modern LLM family** — GPT-4, LLaMA-3, Mistral, Gemma, Qwen and why decoder-only
  architectures dominate
- **Parameter counts** — where parameters come from (embeddings, attention, feedforward,
  output layers) and why more parameters ≠ automatically better

## Materials

- [Week 3 lecture slides](resources/Week%203.pdf)
- [tiny_transformer_llm.py — Lab sheet 3 starter notebook (TinyStories)](resources/tiny_transformer_llm.py)
- [Lab sheet 3 — Developing a Tiny Transformer Language Model](resources/Lab%20sheet%203.pdf)

## My work

- [Notes](notes.md)
- [Lab work — tiny Transformer trained from scratch](lab/lab3_tiny_transformer.ipynb)
- [Reflection](reflection.md)
