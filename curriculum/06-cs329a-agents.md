# 6 — CS329A + Hur & Song, *Build an AI Agent (From Scratch)*

| | |
|---|---|
| **Course** | Stanford CS329A, *Self-Improving AI Agents* — <https://cs329a.stanford.edu/>. Azalia Mirhoseini and Aakanksha Chowdhery. Graduate **seminar**; offered Fall 2025, taught previously in Winter. Lectures on YouTube |
| **Book** | Jungjun **Hur** and Younghee **Song**, *Build an AI Agent (From Scratch)*, Manning — <https://www.manning.com/books/build-an-ai-agent-from-scratch> |

> A note on the attribution: "Jungjun" is the first name. The book has two authors,
> Hur and Song, and citing it as "by Jungjun" loses the second one.

## Does the pairing work

**No — and unlike block 3, the mismatch runs the other way: the book is far below the
course.**

CS329A is a **graduate seminar** on self-improving agents: self-improvement techniques,
multi-step reasoning and planning, and constructing evaluation frameworks. A seminar is a
paper-reading course. You arrive having read the week's papers and you argue about them.
It assumes you already know how to build an agent and asks what makes one improve.

Hur & Song is a Manning "from scratch" book — the same genre as Raschka's, aimed at
someone who has not built one before. It teaches you the thing CS329A assumes you already
have.

That is not a reason to drop either. It is a reason to **sequence them**: read the book
first and quickly, build one agent that works, and then take the seminar with something
concrete to argue from. A seminar without a built artefact behind you is a reading group.

Confidence note: I am inferring the book's level from its title, publisher and series
rather than from having read it, and it is recent enough that I would check its table of
contents before committing. The inference about CS329A's level is firmer — "graduate
seminar" is what the course calls itself.

## What is actually hard here, and it is not the plumbing

Building an agent — a loop, tool schemas, a model that emits tool calls — is a weekend.
Every framework does it and most of them do it badly, but the plumbing is not the
difficulty. Three things are:

1. **Evaluation.** An agent's output is a *trajectory*, not an answer. Two runs on the same
   input take different paths and both may be acceptable. Scoring this is unsolved, and
   CS329A being explicitly about constructing evaluation frameworks is the strongest reason
   to take it.
2. **Failure compounding.** A ten-step task with 95% per-step reliability succeeds 60% of
   the time. Most agent demos are short; most agent products are not. The arithmetic is
   trivial and its consequences dominate everything.
3. **Safety of composition.** Individually-permitted actions can form an impermissible
   sequence. Per-call authorisation cannot see this because it is a property of the
   trajectory. This is developed concretely in
   `enterprise-ai-projects/10-mcp-legacy-erp.md` (on the
   `claude/enterprise-ai-project-guides` branch, not this one)
   on its own branch.

If you finish this block able to state all three precisely, it did its job.

## The trap

Framework tourism. There is an enormous amount of agent tooling, most of it thin wrappers
over a loop and a prompt, and it is possible to spend the whole block evaluating libraries
rather than confronting items 1–3. Build the loop yourself once, from the book, so that you
know what the frameworks are hiding.

## Build after this block

```
enterprise-ai-projects/10-mcp-legacy-erp.md   an MCP server over a real ERP: tool
                                              granularity, idempotency under a retrying
                                              caller, authorisation identity, and the
                                              composition problem
enterprise-ai-projects/08-fallback-gateway.md degradation when the model fails mid-agent
enterprise-ai-projects/11-shadow-evaluator.md evaluating a candidate against production
system-design/                                the reliability substrate an agent runs on
```

The MCP guide is the one. It puts an agent against a system that will not forgive it, which
is the only setting where items 1–3 stop being abstractions.

## Reference

- Model Context Protocol specification — <https://modelcontextprotocol.io/specification/>. Current stable revision **2025-11-25** (previous: 2025-06-18). Pin a revision.
- OWASP Top 10 for LLM Applications — <https://genai.owasp.org/> — excessive agency and insecure tool design.
