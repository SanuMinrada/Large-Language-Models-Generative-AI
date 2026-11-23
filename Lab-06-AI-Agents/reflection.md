# Week 6 Reflection — Building a Tool-Using AI Agent

This lab was fun because I could actually see the agent doing things, not just replying with
text. Watching the real Calculator and Notepad apps open on their own, because the model decided
to use a tool, made the whole idea of "AI agents" click for me in a way reading about it never did.

Adding memory in Part B felt like a small but real step up — the agent wasn't just keeping track
of the conversation anymore, it was saving facts to an actual database and pulling them back out
later. Simple to build, but it felt like a proper feature.

Part C (RAG) taught me something I didn't expect: having a vector database doesn't automatically
mean the agent gives good answers. I found out the hard way that my Lab 5 database had the same
PDF saved into it four times by accident, which messed up the search results. Even after I fixed
that, some answers still weren't really using the right information — the model just sounded
confident anyway. That was a good reminder to actually check what's being retrieved, not just
whether the final answer sounds right.

Part D (Google Calendar) was the most time-consuming part, but not because of the code. Setting up
a Google Cloud project, enabling the API, and getting through the OAuth consent screen took a few
tries — I even got blocked with an "access denied" error until I added myself as a test user in
the right place. Once that was done, though, the agent could really check my schedule and create
an event, and seeing it show up in my actual Google Calendar was satisfying.

Overall, this lab showed me that building an agent is less about the AI model itself and more
about wiring real tools around it carefully, and testing them properly instead of assuming they
work.
