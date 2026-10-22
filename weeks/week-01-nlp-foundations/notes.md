# Week 1 Notes — NLP Foundations

Source: [Week 1 lecture slides](resources/Week%201.pdf) — *Large Language Models and Generative AI*, Lesson 1: From NLP to Generative AI.

## Key terms

- **NLP (Natural Language Processing)** — field at the intersection of AI and linguistics; enables human↔computer interaction using natural language.
- **Corpus** (pl. *corpora*) — a large, structured collection of text stored digitally, used to train/test NLP models.
- **Vocabulary** — the list of all unique words in a corpus; forms the dictionary a model references.
- **Token** — a unit of text (word, sub-word, or character) produced by tokenization.
- **Tokenization** — splitting text into tokens.
- **Stop words** — common, low-information words (articles, conjunctions, prepositions, pronouns, common verbs) typically removed before analysis.
- **Stemming** — rule-based reduction of a word to a root form; can produce non-words.
- **Lemmatization** — reduction of a word to its dictionary (lemma) form; always a valid word.
- **One-hot encoding** — representing each vocabulary word as a sparse binary vector.
- **Bag of Words (BoW)** — representing a document as a vector of word counts over the vocabulary.
- **N-gram** — a contiguous sequence of *N* words/characters, used to capture local word order.
- **Word embedding** — a dense numerical vector representation of a word where semantically similar words have similar vectors.
- **Word2Vec** — neural method (Google, 2013) that learns word embeddings from context, via CBOW or skip-gram.
- **Contextual embedding** — a word vector that changes depending on surrounding context (e.g. BERT), unlike Word2Vec's one-fixed-vector-per-word.

## Summary

### What is NLP?

NLP bridges computational models and human language understanding. Real-world examples: search engines (Google), voice assistants (Siri, Alexa), translation systems.

**How NLP works:** raw, unstructured input text → NLP processing → structured, machine-readable output.
Example: `"Add Eggs and Milk to my Shopping List"` → `<ShoppingList><item>Eggs</item><item>Milk</item></ShoppingList>`

**Core NLP tasks:**
- **Sentiment analysis** (opinion mining) — classifies text as positive / neutral / negative. Used for product reviews, social media monitoring, customer feedback.
- **Text extraction / Named Entity Recognition (NER)** — pulls out entities such as organisations, people, dates, locations from unstructured text.
  Example: *"Apple Inc. was founded by Steve Jobs in 1976 in Cupertino, California"* → ORGANIZATION: Apple Inc., PERSON: Steve Jobs, DATE: 1976, LOCATION: Cupertino, CA.
- **Text classification** (tagging/categorization) — assigns pre-defined tags to text; used for spam filtering, topic detection, language detection, content moderation.
- **Speech recognition (ASR)** — Audio → ASR model → Text; examples: Siri, Google Assistant, Alexa, YouTube auto-captions, medical transcription.
- **Chatbots & Question Answering** — chatbots conduct automated conversations (ChatGPT, Gemini, Claude); QA finds a specific answer from a document set/knowledge base (search engines, virtual assistants, enterprise knowledge systems).

### Text as data

Text has a hierarchy: **Corpus → Document → Paragraph → Sentence → Word/Token**.

**Basic text preprocessing** (in typical order):
1. **Text cleaning** — lowercasing (so "Python" = "python"); removing punctuation/numbers/special characters (`"Price is $100!!"` → `"price"`); removing HTML tags, URLs, emojis; expanding contractions (`can't` → `cannot`, `I'm` → `I am`).
2. **Case conversion**
3. **Tokenization** — breaking text into tokens:
   - *Sentence-level* — for summarization, translation
   - *Word-level* — splits by whitespace/punctuation; most common, used for general NLP tasks
   - *Character-level* — avoids out-of-vocabulary (OOV) words entirely; used for noisy/multilingual text
   - *Sub-word* — splits into frequent sub-units, balancing vocabulary size and OOV handling; used by BERT, GPT, T5.
     Example: `"internationalization"` → `inter` + `national` + `ization` — the model can infer meaning of unseen words from known sub-word pieces.
4. **Stop word removal** — language-specific; categories include articles (a, an, the), conjunctions (and, but, or), prepositions (in, on, at, with), pronouns (he, she, it, they), common verbs (is, am, are, was, were, be).
5. **Stemming vs. Lemmatization**
   | | Stemming | Lemmatization |
   |---|---|---|
   | Output | Root form, may be a non-word | Dictionary form, always valid |
   | Example | playing → play; happily → happi; nationalization → nation | running → run; better → good; studies → study |
   | Best for | Search engines, information retrieval, speed-critical tasks | Machine translation, sentiment analysis, NER, summarization |

### Text representation

ML models need numbers, not raw text, so text must be converted into numerical vectors.

- **Encoding** — one-hot encoding (e.g. vocabulary `[cat, dog, bird]` → `cat=[1,0,0]`, `dog=[0,1,0]`, `bird=[0,0,1]`): simple but sparse and captures no semantics. Label encoding is the alternative.
- **Bag of Words (BoW)** — build the vocabulary, then represent each document as a vector of word counts.
  - Order of words is ignored; only frequency matters → creates sparse vectors.
  - **Limitations:** no semantics (e.g. "bank" the river vs. "bank" the financial institution are treated identically); no word order ("dog bites man" = "man bites dog" as vectors); sparse vectors (a 50,000-word vocabulary is mostly zeros, memory-inefficient).
- **N-grams** — contiguous sequences of N words preserve some local word order/phrase context.
  Example: *"New York City"* → unigrams `[New][York][City]`, bigrams `[New York][York City]`, trigrams `[New York City]`. Bigrams capture meaningful units like "machine learning" that BoW alone misses.
- **What comes after BoW:** N-grams (local word order), TF-IDF (weights rare words higher, common words lower), embeddings (dense vectors capturing semantic meaning).
- **Word embeddings & Word2Vec** (Google, 2013) — dense vectors where similar words have similar vectors; learned from context via **CBOW** (predict word from context) or **skip-gram** (predict context from word). Supports "semantic arithmetic": `King − Man + Woman ≈ Queen`.
- **Contextual embeddings & BERT** — Word2Vec gives one fixed vector per word regardless of context. Contextual embeddings (BERT, GPT, ELMo, RoBERTa, T5) produce different vectors for the same word depending on context — e.g. "bank" near "river" vs. "bank" near "loan" get different vectors. BERT (Bidirectional Encoder Representations from Transformers, Google 2018) reads sentences in both directions at once.

### Neural NLP architectures

| Architecture | How it works | Used for | Limitation |
|---|---|---|---|
| RNN | Processes sequences step-by-step with a hidden state carried forward | Text generation, speech processing | Struggles with long sequences (vanishing gradient) |
| LSTM | RNN with gates controlling what to remember/forget/output | Machine translation, sentiment analysis | — |
| CNN | Filters/kernels detect local patterns in text | Text classification, spam detection | — |
| Transformer | Self-attention processes all words simultaneously, learning which words to focus on | Powers BERT, GPT, ChatGPT, modern LLMs | State of the art for NLP today |

## Code / examples

- One-hot encoding: `cat → [1, 0, 0]`, `dog → [0, 1, 0]`, `bird → [0, 0, 1]` (vocabulary = `[cat, dog, bird]`)
- Bag-of-Words matrix example:

  | Doc | I | love | Python | is | great |
  |---|---|---|---|---|---|
  | D1: "I love Python" | 1 | 1 | 1 | 0 | 0 |
  | D2: "Python is great" | 0 | 0 | 1 | 1 | 1 |
  | D3: "I love coding" | 1 | 1 | 0 | 0 | 0 |

- Sub-word tokenization: `internationalization → inter + national + ization`
- Semantic arithmetic (Word2Vec): `King − Man + Woman = Queen`
- `Lab sheet 1` applies this week's concepts end-to-end: load the Sentiment Labelled Sentences dataset → preprocess text → build a BoW representation → train an ML classifier (SVM / Logistic Regression) → evaluate (accuracy, precision, recall, F1).

## Things to revisit

- The exact trade-offs between stemming and lemmatization in practice (when the non-word output from stemming actually matters).
- Why BoW's sparsity becomes a real problem at scale, and how TF-IDF/embeddings address it.
- How Word2Vec's context window during training produces the "semantic arithmetic" property.
