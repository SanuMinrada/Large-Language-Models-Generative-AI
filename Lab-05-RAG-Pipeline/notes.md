# Week 5 Notes — Grounding LLMs with External Knowledge (RAG)

## Key terms

- **RAG (Retrieval-Augmented Generation)** — combines a retrieval system with an LLM: instead of relying only on its trained weights, the model retrieves relevant documents at inference time and uses them as context.
- **Chunking** — splitting documents into smaller segments before embedding, so retrieval can return focused, relevant pieces rather than whole documents.
- **Embedding** — a dense vector representation of text that captures meaning, letting similar text cluster together for "fuzzy" (semantic) search.
- **Vector database** — a database built to store embeddings and search them by similarity (e.g. Chroma, FAISS, Pinecone).
- **Sparse retrieval (BM25)** — keyword-based search; fast and exact, but misses paraphrases/synonyms.
- **Dense retrieval** — embedding-similarity search; captures meaning but needs pre-computed embeddings.
- **Hybrid retrieval** — combines BM25 + dense scores (e.g. via Reciprocal Rank Fusion) for better precision and recall together.
- **Reranking** — re-scoring retrieved chunks with a more precise model (e.g. a cross-encoder) before sending them to the LLM.
- **HyDE (Hypothetical Document Embeddings)** — generate a hypothetical answer first, then retrieve using that, since it may match the knowledge base's wording better than the raw question.

## Summary

### Why LLMs need external knowledge
LLMs have a training-data cutoff (can't know recent events), no access to private/internal documents, tend to hallucinate confidently when uncertain, and can't cite specific sources on their own. RAG addresses all four by grounding generation in retrieved, real documents.

### The RAG pipeline
Four steps: **indexing** (done once) → **retrieval** → **augmentation** → **generation** (the last three repeat per query).
- **Indexing:** parse documents → chunk → embed → store in a vector database.
- **Retrieval:** encode the user's query into a vector, compare it against stored embeddings via semantic search, return the most relevant chunks.
- **Augmentation:** combine the retrieved chunks into a context block, then merge that with the user's query (plus instructions, e.g. "answer only from this context") into a single prompt.
- **Generation:** the LLM produces a response grounded in that prompt.

### Chunking strategies
- **Fixed-size** — split by token count; simple, but can cut sentences mid-way (overlap helps).
- **Sentence/paragraph-based** — split on natural boundaries; preserves meaning better.
- **Recursive splitting** — try larger splits first, fall back to smaller ones; LangChain's default.
- **Semantic chunking** — use embeddings to detect topic shifts; most coherent, but slower.

### Retrieval methods
Sparse (BM25) is fast and exact-match only; dense (embeddings) captures meaning but costs more to compute; hybrid combines both and is what most production RAG systems (e.g. Azure AI Search, Weaviate) actually use.

### RAG vs fine-tuning
RAG is cheap to update (swap the knowledge base) and grounds answers in real text, so hallucination risk is lower — best for dynamic, fact-heavy Q&A. Fine-tuning is expensive to update (needs retraining) but is much better at teaching a model a specific style, tone, or output format.

### Beyond basic RAG
Reranking (cross-encoder re-scoring), query expansion/HyDE (retrieve using a reworded or hypothetical version of the query), multi-hop retrieval (chain several retrieval steps for questions that need connecting multiple documents), and contextual compression (trim each retrieved chunk down to just the relevant sentences before it reaches the LLM).
