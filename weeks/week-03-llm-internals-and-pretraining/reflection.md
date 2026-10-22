# Week 3 Reflection — LLM Internals and Pre-training

## What I learned

Reading the "looking inside an LLM" material in the [notes](notes.md) was one thing, but actually building and training [a tiny Transformer from scratch in Lab 3](lab/lab3_tiny_transformer.ipynb) made the whole forward-pass pipeline (tokenizer → Transformer blocks → LM head → sampling) feel real instead of diagrammatic. Watching the validation loss fall from 4.46 (untrained, roughly `ln(75)` for a 75-character vocabulary — i.e. the model starting out no better than guessing uniformly) down to 1.56 over 15 epochs, and perplexity land at 4.77, gave me an actual number to attach to the abstract idea of "the model is learning structure." The generated text after training is still mostly gibberish, but it's gibberish with English-shaped words and roughly sentence-like punctuation, which is a genuinely different thing from the random-character output the untrained model would have produced. The BERT vs. GPT vs. T5 comparison table also finally made sense end-to-end once I'd implemented the causal mask myself — "causal/unidirectional" isn't just a label, it's one `torch.triu(...)` call that determines whether a position can see future tokens or not.

## What was challenging

Interpreting the perplexity number in context was harder than I expected — knowing that 4.77 is "good" requires knowing what the ceiling is (75, for uniform random guessing over this vocabulary) and that this is character-level rather than word/sub-word level, so it isn't directly comparable to the perplexity numbers quoted for real LLMs. The "why does decoder-only dominate in 2024" discussion was also non-obvious at first — I initially assumed encoder-decoder (T5-style) would be strictly more powerful since it has both components, but the notes' point about KV-caching and avoiding cross-attention overhead reframed it as an engineering/scaling argument rather than a capability one.

## How it connects to earlier weeks

This week is the most directly cumulative one so far. The Lab 3 Transformer reuses, line for line, the mechanisms from Week 2's notes: GELU in the feed-forward block, residual connections + LayerNorm (built into `nn.TransformerEncoderLayer`) preventing vanishing gradients, and multi-head self-attention — just with a causal mask added, which is the one concrete architectural difference between this week's GPT-style Causal Language Model and the more general attention mechanism Week 2 introduced. It also directly revisits Week 1's tokenization discussion: I used character-level tokenization in the lab (matching the lecturer's starter code) specifically because it's the simplest option from Week 1's "types of tokenization" slide, but the Week 1 notes already flagged that sub-word tokenization (BERT/GPT/T5) exists precisely to balance vocabulary size against sequence length — a trade-off I could feel directly in this lab, since a 75-character vocabulary means every single word takes many tokens to generate.

## Questions I still have

- How much would switching Lab 3's tokenizer from character-level to a simple sub-word scheme (e.g. a small trained BPE tokenizer) improve generation quality for the same number of training steps — would it converge faster, or just produce different *kinds* of errors?
- The "Experience with model settings" comparison showed the 1-layer/1-head model converging to a clearly worse validation loss (1.768 vs. 1.563) in the same number of steps — would it eventually catch up with more steps, or does limited capacity create a hard ceiling no amount of extra training can overcome?
- How exactly does KV-caching interact with the autoregressive loop at inference time — my lab's `generate()` function recomputes the full forward pass over the whole context at every step rather than caching, so I understand *why* caching would help, but not the exact mechanics of what gets stored and reused.

## What I'd try next

- Swap the character-level tokenizer for a small trained sub-word tokenizer and re-run the same training loop, to get a direct before/after comparison rather than just a conceptual one.
- Increase `block_size` beyond 64 characters (roughly one sentence) to see whether longer context measurably improves story coherence, directly testing the "long-range relationships" idea from the pre-training notes.
- Actually launch the Gradio chat interface built in Lab 3 (commented out during the batch run) and interact with the trained model directly, rather than only reading its generated samples.
