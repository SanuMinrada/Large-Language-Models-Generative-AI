# Final Reflection

## Overall knowledge and skills gained

Across this module I went from using models (Lab 1's sentiment analysis) to actually building and
training one from scratch (Lab 3's tiny Transformer), fine-tuning a real LLM (Lab 4), building a
full RAG pipeline (Lab 5), and building AI agents that use tools (Lab 6). Each lab added a new
layer, and by the end I understood a lot more about what's actually happening under the hood
instead of just calling an API.

## How the labs connect to each other

Lab 3 made the Transformer architecture feel real, which made Lab 4's fine-tuning make a lot more
sense — I wasn't just running a script, I understood what was actually being adjusted. Lab 5's RAG
pipeline then got reused directly as a tool inside Lab 6's agent instead of being rebuilt, which
showed me how these labs aren't separate exercises but actually build on each other.

## Biggest challenges across the module

The hardest part wasn't getting something to run, it was getting an honest result. In Lab 4, BLEU
scores looked fine, but the model still didn't recall exact facts correctly. In Lab 5 and 6, the
RAG tool sometimes retrieved chunks that weren't even related to the question, which taught me
that having a database isn't the same as having good answers. Lab 6's Google Calendar setup was
also a real challenge, but more because of OAuth and Google Cloud setup than the actual code.

## How I'd approach a similar project differently next time

I'd check the data and the outputs more carefully as I go, instead of assuming something worked
just because it ran without errors. A few issues (like duplicate PDFs messing up search results)
only showed up because I tested things properly afterwards. Testing in smaller steps earlier would
save time compared to building a lot first and debugging everything at the end.
