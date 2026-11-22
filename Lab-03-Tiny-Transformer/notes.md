# Week 3 Notes — LLM Internals and Pre-training

## Key terms

- **Autoregressive model** — generates one token at a time, feeding each new token back in as input for the next.
- **Forward pass** — one pass of input through tokenizer → Transformer blocks → LM head to produce output.
- **LM head** — the final layer that turns the model's hidden state into a probability distribution over the next token.
- **Context length** — the maximum number of tokens a model can process at once.
- **KV cache** — storing earlier tokens' computations so they aren't recomputed at every generation step.
- **Encoder** — reads input and builds a representation of it (e.g. BERT). **Decoder** — generates text one token at a time (e.g. GPT).
- **Pre-training** — the large-scale initial training phase on raw text, before any task-specific fine-tuning.
- **Masked Language Modeling (MLM)** — predict hidden tokens using bidirectional context (BERT).
- **Causal Language Modeling (CLM)** — predict the next token using only preceding tokens (GPT).
- **Span corruption** — reconstruct a removed span of tokens (T5).

## Summary

### Looking inside an LLM
- An LLM takes in a prompt and generates text one token at a time: each token goes through tokenizer → Transformer block stack → LM head, then gets appended back onto the prompt for the next step (autoregressive generation).
- Inside each Transformer block: a **self-attention** layer lets the model relate tokens to each other (e.g. resolving what "it" refers to), and a **feed-forward layer** holds most of the model's processing capacity.
- **Context length** caps how many tokens can be processed at once; **KV caching** avoids recomputing earlier tokens at every step, which is what makes generation fast enough to be usable.

### Pre-training at scale
- Pipeline: **pre-training → fine-tuning → application**. During pre-training the model learns vocabulary/token patterns, syntax, semantic relationships, and long-range relationships, purely from raw text.
- Three pre-training objectives, each tied to a different architecture:

  | | BERT | GPT | T5 |
  |---|---|---|---|
  | Architecture | Encoder only | Decoder only | Encoder–Decoder |
  | Objective | Masked LM (bidirectional) | Causal LM (left-to-right) | Span corruption |
  | Best for | Classification, NER, QA | Text generation | Translation, summarization |

- **Why decoder-only dominates now:** models like GPT-4 and LLaMA are decoder-only because it scales better — KV caching makes generation efficient and there's no cross-attention overhead, unlike encoder-decoder models. BERT-style encoders are still the better choice for classification/retrieval tasks where bidirectional context matters.
- **More parameters ≠ automatically better** — capacity increases, but quality also depends on training data, architecture, tokenizer, and context length.
