# Week 2 · Aug 31–Sep 6, 2026

> **Finish the server, then the protocols underneath it.**
> One week on what a byte stream actually carries: a name lookup, a checksum, a wire protocol with framing, and a proxy that reads none of it.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `http-server/` | 15 of 86 | core · spine | [`../week-01/http-server/`](../week-01/http-server/) |
| [`dns-server/`](dns-server/) | 9 | full only | here |
| [`cryptographic-library/`](cryptographic-library/) | 5 | full only | here |
| [`communication-protocols/`](communication-protocols/) | 34 | full only | here |
| [`toralizer/`](toralizer/) | 16 of 21 | full only | here |

**79 block hours**, plus the daily Lean slot (~8.4 h) = 87 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `http-server` 45 h | 45 |
| **spine** | `http-server` 27 h | 27 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. Close out `http-server` first — concurrency and keep-alive — and do not move on until it survives load.
2. `dns-server` and `cryptographic-library` are short and mostly parsing. Treat them as one two-day block.
3. `communication-protocols` is the big one this week. UART and SPI first; I2C, CAN and RS-485 build on the framing ideas in those two.
4. `toralizer` last. It is a socket exercise you are now over-prepared for, which is the point of putting it here.

## Done means

- `ab -n 10000 -c 100 http://localhost:8080/` completes with zero failed requests.
- Your resolver answers an A and a CNAME query against a real root server, and handles a truncated response by retrying over TCP.
- Your SHA-256 matches `sha256sum` on ten random files, including an empty one and one that lands exactly on a block boundary.
- You can draw the frame layout of UART, SPI and I2C from memory and say which one needs a clock line and why.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 2            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 1](../week-01/) · [Roadmap](../ROADMAP.md) · [Week 3 →](../week-03/)
