# Week 2 Notes — Transformers and Attention

Source: [Week 2 lecture slides](resources/Week%202.pdf) — Neural Networks: Foundations for LLMs, and Transformer Architecture: Attention Is All You Need. Also: [Vaswani et al. (2017)](resources/1706.03762v7.pdf).

## Key terms

- **Neuron** — computes `output = activation(Σ wᵢxᵢ + b)`; weights are learned by gradient descent, activation introduces non-linearity.
- **MLP (Multi-Layer Perceptron)** — one or more fully-connected hidden layers trained via backpropagation.
- **Activation function** — non-linear function applied after a layer's weighted sum (Sigmoid, Tanh, ReLU, GELU, SwiGLU).
- **Backpropagation** — computing gradients of the loss w.r.t. weights via the chain rule, then updating weights (gradient descent / Adam).
- **Vanishing gradient** — gradients shrink toward zero through many layers/steps, stalling learning; mitigated by GELU/SwiGLU, residual connections, and LayerNorm.
- **Self-attention** — mechanism letting a model weigh the relevance of other tokens when processing a given token.
- **Query (Q), Key (K), Value (V)** — the three learned projections used in attention: Q = what a token is looking for, K = what a token offers to be matched against, V = the content actually retrieved.
- **Scaled dot-product attention** — `Attention(Q,K,V) = softmax(QKᵀ/√dₖ) · V`.
- **Multi-head attention** — running several attention "heads" in parallel, each able to specialise on different relationships, then concatenating and projecting the results.
- **Positional encoding** — injects token-order information into embeddings, since self-attention itself is permutation-invariant.
- **Context window / sequence** — the set of tokens processed together by the attention mechanism.

## Summary

### Neural network foundations (bridge to Transformers)

- **The neuron:** `output = activation(Σ wᵢxᵢ + b)`. Inputs = features (e.g. word frequencies, embeddings); weights = learned connection strengths (negative = inhibitory); activation introduces non-linearity — without it, stacking layers collapses to a single linear layer. Example: a spam filter where `x` = word counts, `w` = how predictive each word is of spam, `b` = bias toward "ham".
- **Perceptron:** weighted sum → step function (`f(x) = 0` if `x < 0`, else `1`).
- **Architectures:** Single-layer (no hidden layers) → Multi-Layer Perceptron (one+ hidden layers, fully connected, trained via backprop) → Deep Neural Network (many layers, learns complex representations, needs large data and compute).
- **Connectivity patterns:** fully connected (dense) vs. partially connected.
- **Feedforward vs. recurrent:** feedforward networks have no loops; recurrent networks (RNNs) have feedback connections that carry a hidden state across time steps.
- **Activation functions compared:**

  | Function | Formula | Range | Used in LLMs? | Why |
  |---|---|---|---|---|
  | Sigmoid | σ(x)=1/(1+e⁻ˣ) | [0,1] | Rarely | Saturates; vanishing gradient |
  | Tanh | tanh(x) | [-1,1] | LSTM only | Zero-centred but still saturates |
  | ReLU | max(0,x) | [0,∞) | Legacy | Dying neuron problem |
  | GELU | x·Φ(x) | (−0.17,∞) | ✓ BERT, GPT-2 | Smooth, probabilistic gate |
  | SwiGLU | x·σ(x)⊗(Wx) | (−∞,∞) | ✓ LLaMA, PaLM | Best empirical results |

  GELU and SwiGLU are smooth, so gradients keep flowing even near zero — critical for deep Transformers.

- **MLP forward pass:** `h¹ = GELU(W¹x + b¹)`, `h² = GELU(W²h¹ + b²)`, `ŷ = W³h² + b³`. The Transformer's feed-forward sub-layer is exactly this MLP(GELU), applied position-wise to every token.
- **Backprop & training loop:** (1) forward pass → predictions; (2) compute loss (e.g. cross-entropy); (3) backward pass (`loss.backward()`, chain rule: `∂L/∂w = ∂L/∂y · ∂y/∂w`); (4) update weights (`w ← w − α·∂L/∂w`, typically with Adam).
- **Vanishing gradients:** a sigmoid-based chain multiplies by ~0.25 per layer, so after 20 layers the gradient is roughly `0.5²⁰ ≈ 0.000001`. Fixed in modern architectures by GELU/SwiGLU activations + residual connections + LayerNorm.
- **Why Transformers replaced RNNs:**

  | | MLP | RNN | Transformer |
  |---|---|---|---|
  | Handles sequences? | ✗ Fixed input | ✓ Step by step | ✓ All at once |
  | Long-range memory | ✗ None | ✗ Vanishes | ✓ O(1) path |
  | Parallelisable? | ✓ Yes | ✗ Sequential | ✓ Fully |
  | Core operation | Matrix × vector | Recurrence `W·hₜ₋₁` | Self-attention |
  | Used in LLMs? | FFN sub-layer | ✗ Deprecated | ✓ Full architecture |

### Transformer architecture — "Attention Is All You Need"

- **Why not Seq2Seq + RNN attention:** encoder bottleneck (a single vector loses context), sequential computation (RNNs can't be parallelised), long-range dependencies vanish over many steps, and additive attention still can't escape RNN sequentiality.
- **Transformer's answers:** self-attention over all positions simultaneously; fully parallel (no recurrence, no convolution); O(1) path between any two tokens, however far apart; positional encodings restore the sequence order that self-attention alone discards.
- **Query · Key · Value — the three roles:**
  - **Query:** "What am I looking for?" — the focus word asking for context (e.g. the token "bank" queries its context to resolve ambiguity).
  - **Key:** "What do I represent?" — the index that queries are matched against; each token announces what information it holds.
  - **Value:** "What information do I carry?" — the actual payload blended into the output once matched.
- **Self-attention pipeline:** Input tokens → project to Q, K, V → attention scores = `Q · Kᵀ` (dot product) → softmax (normalise into weights) → weighted sum of V (`Attention · V`) → contextual output embeddings.
- **Multi-head attention:** `MultiHead(Q,K,V) = Concat(head₁,…,headₕ)·Wᵒ`, where `headᵢ = Attention(QWᵢQ, KWᵢK, VWᵢV)`. Different heads learn different relationships — e.g. syntactic links (subject–verb), coreference (pronoun resolution), local context (bigrams/collocations), long-range dependencies across clauses. GPT-3 uses 96 heads (dₖ=128); BERT-base uses 12 heads (dₖ=64).
- **"Attention Is All You Need" (Vaswani et al., 2017) — key results:**
  - Full architecture: encoder stack (N=6) + decoder stack (N=6), each block combining self-attention, feed-forward layers, residual connections, and layer norm.
  - Scaled dot-product attention + multi-head attention, with a final linear projection `Wᵒ`.
  - BLEU results: EN→DE 28.4 BLEU (+2 over prior SOTA), EN→FR 41.0 BLEU (new SOTA) — trained in 3.5 days on 8 GPUs vs. weeks for prior approaches.
  - Why self-attention (§5.3): O(1) long-range path vs. O(n) for RNNs; O(n²·d) compute vs. O(n·d²) for RNNs; highly interpretable attention maps.
- **Exit-ticket answers (from slides):**
  - *Why scale by √dₖ?* Prevents dot products from growing too large → avoids a saturated softmax → avoids vanishing gradients.
  - *What does each attention head learn?* Different subspace representations: syntax, coreference, local and long-range patterns.
  - *Why are positional encodings necessary?* Self-attention is permutation-invariant by itself; positional encodings inject order information into token embeddings.

### Applied activity: "AI in Everyday Challenges"

A group exercise using an LLM as a product-development partner to design a product addressing a real student problem (e.g. forgetfulness, distraction, food waste, dehydration, time management, plastic waste, carrying too much, budgeting). Deliverables per group: product concept, name, key features, target user, and a visual prototype/mock-up — see [Lab sheet 2](resources/Lab%20sheet%202.pdf).

## Code / examples

- Neuron output: `output = activation(Σ wᵢxᵢ + b)`
- MLP forward pass: `h¹ = GELU(W¹x + b¹)`, `h² = GELU(W²h¹ + b²)`, `ŷ = W³h² + b³`
- Weight update rule: `w ← w − α · ∂L/∂w`
- Scaled dot-product attention: `Attention(Q,K,V) = softmax(QKᵀ / √dₖ) · V`
- Multi-head attention: `MultiHead(Q,K,V) = Concat(head₁,...,headₕ) · Wᵒ`, `headᵢ = Attention(QWᵢQ, KWᵢK, VWᵢV)`
- `transformer_practice.py` in this week's resources builds a small Transformer encoder block (token + positional embeddings, `nn.MultiheadAttention` with a causal mask, LayerNorm, feed-forward) from scratch and trains it on a toy character-level dataset to sanity-check the mechanics above.

## Things to revisit

- Deriving why `√dₖ` specifically (not some other constant) keeps the softmax well-behaved.
- Walking through a worked numeric example of scaled dot-product attention by hand.
- How causal masking (used in `transformer_practice.py`) differs from the bidirectional self-attention described in the lecture slides — this connects directly into Week 3's encoder vs. decoder discussion.
