# Week 15 · Nov 30–Dec 6, 2026

> **The server and the protocols under it.**
> Hold HTTP under load. Then DNS, SHA-256, the buses.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `http-server/` | rest of 86 | core · spine | [`../week-01/http-server/`](../week-01/http-server/) |
| `dns-server/` | 9 | full | [`../week-02/dns-server/`](../week-02/dns-server/) |
| `cryptographic-library/` | 5 | full | [`../week-02/cryptographic-library/`](../week-02/cryptographic-library/) |
| `communication-protocols/` | 34 | full | [`../week-02/communication-protocols/`](../week-02/communication-protocols/) |

On the narrower tracks this same week is:

core · spine: HTTP concurrency and keep-alive. full: the rest.

The week folder may not contain these directories. That is fine —
`progress.py` finds them, and the links above are where the files live.

## What to do, in order

1. Keep-alive. `ab -n 10000 -c 100`.
2. DNS A record. SHA-256 padding.
3. UART → SPI → I2C → CAN.

## Done means

- What broke at 100 concurrent connections, written down.
- A start bit is a contract.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — the journal post for today. The expected title is already there.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 15            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 14](../week-14/) · [Roadmap](../ROADMAP.md) · [Week 16 →](../week-16/)
