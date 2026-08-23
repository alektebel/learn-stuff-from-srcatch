# Week 1 · Aug 24–Aug 30, 2026

> **Sockets, or nothing else works.**
> Everything later in this repo — a serving stack, a NAT gateway bill, a CUDA memory copy — is a thing on the other side of a file descriptor. Start there.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| [`bash-from-scratch/`](bash-from-scratch/) | 8 | core · spine | here |
| [`http-server/`](http-server/) | 71 of 86 | core · spine | here |

**79 block hours**, plus the daily Lean slot (~8.4 h) = 87 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `bash-from-scratch` 8 h, `http-server` 37 h | 45 |
| **spine** | `bash-from-scratch` 8 h, `http-server` 19 h | 27 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. Two days on `bash-from-scratch` first. It is the smallest thing here and it forces `fork`, `exec`, `wait`, pipes and signals into your hands before anything depends on them.
2. Then `http-server` for the rest of the week. Get a socket accepting, then a request parsed, then a file served — in that order, and run a real browser against it after each one.
3. Do NOT start on keep-alive, chunked encoding or concurrency this week. Week 2 is for those. A server that serves one file correctly beats a server that half-serves four things.

## Done means

- `./shell` runs `ls | grep x > out.txt`, handles `Ctrl-C` without dying, and reports exit statuses.
- A browser at `http://localhost:8080/` renders an HTML page with its CSS and images, each with the right `Content-Type`, and a missing path returns a real 404 rather than a hang.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 1            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[Roadmap](../ROADMAP.md) · [Week 2 →](../week-02/)
