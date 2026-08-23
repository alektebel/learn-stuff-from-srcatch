# Week 3 · Sep 7–Sep 13, 2026

> **Compilers, and a machine that runs what they emit.**
> A compiler is the clearest example in this repo of the MVP-then-limit-case ladder: it works on `1+2`, then a nested expression breaks the register allocator, then a branch breaks the GPU's lockstep assumption.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `toralizer/` | 5 of 21 | full only | [`../week-02/toralizer/`](../week-02/toralizer/) |
| [`firewall-from-scratch/`](firewall-from-scratch/) | 25 | full only | here |
| [`c-compiler/`](c-compiler/) | 27 | core · spine | here |
| [`compiler-and-vgpu/`](compiler-and-vgpu/) | 16 | core · spine | here |
| [`quantum-computing-lang/`](quantum-computing-lang/) | 8 | full only | here |

**81 block hours**, plus the daily Lean slot (~8.4 h) = 89 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `http-server` 4 h, `c-compiler` 27 h, `compiler-and-vgpu` 16 h | 47 |
| **spine** | `http-server` 27 h | 27 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. Finish `toralizer` on Monday. Five hours, then close it.
2. `firewall-from-scratch` next — raw sockets and packet parsing, and the last purely-systems thing before compilers.
3. `c-compiler` is the week's spine: lexer, parser, semantic analysis, codegen. Get `int main(){return 2+3;}` compiling and running end to end on day one of it, THEN add features. A compiler that compiles nothing at the end of the week is the standard way this project fails.
4. `compiler-and-vgpu` after it, and it is deliberately smaller: one 32-bit ISA shared by a scalar CPU and a SIMT warp. `python3 check.py` grades you.
5. `quantum-computing-lang` is a palate cleanser. Eight hours, an interpreter over complex amplitudes.

## Done means

- Your C compiler compiles a program with a loop, a function call and recursion, and the binary runs.
- `week-03/compiler-and-vgpu/` prints 12/12 from `python3 check.py`.
- You can explain why right-nested expressions spill registers and left-nested ones do not — the spill table in `codegen.py` shows it.
- You can explain what a warp does at an `if` where lanes disagree.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 3            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 2](../week-02/) · [Roadmap](../ROADMAP.md) · [Week 4 →](../week-04/)
