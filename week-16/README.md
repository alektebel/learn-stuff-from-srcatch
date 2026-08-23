# Week 16 · Dec 7–Dec 13, 2026

> **Compilers, and a machine that runs what they emit.**
> `int main(){return 2+3;}` on day one. Then a warp that diverges.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `c-compiler/` | 27 | core · spine | [`../week-03/c-compiler/`](../week-03/c-compiler/) |
| `compiler-and-vgpu/` | 16 | core · spine | [`../week-03/compiler-and-vgpu/`](../week-03/compiler-and-vgpu/) |
| `firewall-from-scratch/` | 25 | full | [`../week-03/firewall-from-scratch/`](../week-03/firewall-from-scratch/) |
| `toralizer/` | 21 | full | [`../week-02/toralizer/`](../week-02/toralizer/) |

On the narrower tracks this same week is:

core · spine: both compilers. full: firewall and toralizer too.

The week folder may not contain these directories. That is fine —
`progress.py` finds them, and the links above are where the files live.

## What to do, in order

1. Lexer, parser, then `return 2+3`.
2. Semantic analysis, codegen, a spill.
3. ISA, scalar CPU, then SIMT and the barrier that deadlocks.

## Done means

- 12/12 on compiler-and-vgpu if you can.
- A mask stack is why warp divergence is not a mystery.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — the journal post for today. The expected title is already there.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 16            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 15](../week-15/) · [Roadmap](../ROADMAP.md) · [Week 17 →](../week-17/)
