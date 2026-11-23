# Week 6 Notes — AI Agents & Agentic Frameworks

## Key terms

- **AI agent** — an LLM-powered system that perceives its environment, reasons about what to do, takes actions via tools, and iterates toward a goal without a human specifying every step.
- **Tool use / function calling** — the LLM signals it wants to call an external function; the host application actually executes it and returns the result (the LLM itself never runs code directly).
- **ReAct (Reasoning + Acting)** — interleaves Thought → Action → Observation steps: the model reasons, takes an action, observes the result, and repeats until done.
- **Agent executor** — the loop that runs Thought→Action→Observation, stopping on a final answer or a max-iterations limit.
- **Multi-agent framework** — a system where several LLM agents (often with different roles) collaborate to solve a task.
- **Human-in-the-loop checkpoint** — a required pause for human approval before an agent takes an irreversible action.

## Summary

### From chatbots to autonomous agents
Five stages of increasing capability: single-turn chatbot (no memory, no tools) → multi-turn assistant (conversation history) → tool-using LLM (calls APIs/calculators on request) → ReAct agent (plans, acts, observes in a loop) → autonomous agent (long-horizon goals, memory, self-direction).

### What an agent actually does
Four repeating steps: **perceive** (take in tool outputs, user messages, etc.), **reason** (the LLM decides the next action), **act** (call a tool, write a file, search, run code), **remember** (short-term via the context window, or long-term via a vector DB/database).

### Tool use / function calling
The LLM doesn't execute anything itself — it outputs a structured request (e.g. `get_weather(city="Colombo")`), the host app runs the real function and returns the result, and the LLM uses that result to compose its final answer.

### ReAct in practice
A worked trace: the model writes an explicit `Thought`, then an `Action` (a tool call), reads the `Observation` that comes back, and repeats — e.g. first searching for which country has the highest HDI, then searching for that country's capital, before giving a final answer. This loop continues until the task is solved or a step limit is hit.

### Agentic frameworks
- **LangChain** — most popular, modular chains/agents/memory, huge tool ecosystem; best for fast prototyping (this lab builds on it).
- **LangGraph** — stateful, cyclic multi-agent workflows as graphs; pairs with LangChain.
- **AutoGen** (Microsoft) — multiple agents converse with each other to solve a task.
- **CrewAI** — role-based: agents get roles (researcher, writer, etc.), goals, and tools, orchestrated as a "crew".
- **Semantic Kernel** — enterprise-grade SDK for .NET/Python with built-in planners.
- **Agno** — lightweight, minimal, type-safe, single-agent focused.

### LangChain agent architecture
An agent is built from: the LLM/chat model (reasoning engine), a prompt template (instructions + tool descriptions + history), tools (plain Python functions decorated with `@tool`, where the docstring becomes the description the agent sees), memory (buffer/summary/vector-store based), and an agent executor that runs the ReAct loop.

### Evaluating and safeguarding agents
- **Performance:** task completion rate, tool accuracy (right tool, right call), step efficiency, latency/cost.
- **Safety:** hallucinated tool arguments (wrong inputs sent to a real API), prompt injection (malicious content hijacking the agent's behaviour), infinite loops (needs a `max_iterations` guard), and human-in-the-loop checkpoints before irreversible actions.
- **Benchmarks:** AgentBench, GAIA, WebArena, SWE-bench.
