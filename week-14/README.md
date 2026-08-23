# Week 14 · Nov 23–Nov 29, 2026

> **Sockets, finally.**
> A shell and a server, after you have already served tokens.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `bash-from-scratch/` | 8 | core · spine | [`../week-01/bash-from-scratch/`](../week-01/bash-from-scratch/) |
| `http-server/` | start of 86 | core · spine | [`../week-01/http-server/`](../week-01/http-server/) |

On the narrower tracks this same week is:

core · spine: bash, then the accept loop and a real 404.

The week folder may not contain these directories. That is fine —
`progress.py` finds them, and the links above are where the files live.

## What to do, in order

1. Tokenise, `fork`/`exec`/`wait`, pipes, `Ctrl-C`.
2. HTTP: accept, parse, static files, a real 404. Not this week: keep-alive.

## Done means

- A browser rendered your server.
- A pipeline is three processes and two pipes.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — the journal post for today. The expected title is already there.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 14            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 13](../week-13/) · [Roadmap](../ROADMAP.md) · [Week 15 →](../week-15/)
