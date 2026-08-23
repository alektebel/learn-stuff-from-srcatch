# Week 18 · Dec 21–Dec 27, 2026

> **Diffusion, then five small wins to finish on.**
> Deliberately ends on several small completions rather than one heroic unfinished thing. Finishing on five green checkers is better for the last week of a four-month plan than finishing mid-way through something large.

## Finish this week

| Project | Hours | Track | Where it lives |
|---|---|---|---|
| `diffusion-models/` | 40 of 91 | full only | [`../week-17/diffusion-models/`](../week-17/diffusion-models/) |
| [`spectral-graphs/`](spectral-graphs/) | 5 | full only | here |
| [`sas-lineage-tool/`](sas-lineage-tool/) | 8 | full only | here |
| [`web-scraping/`](web-scraping/) | 6 | full only | here |
| [`ml-in-production/`](ml-in-production/) | 8 | full only | here |
| [`mlops/`](mlops/) | 12 | full only | here |

**79 block hours**, plus the daily Lean slot (~8.4 h) = 87 h on the full track.

On the narrower tracks this same week is:

| Track | This week | Block h |
|---|---|---|
| **core** | `vllm-engine` 45 h | 45 |
| **spine** | `cuda-from-scratch` 27 h | 27 |

The narrower tracks move through the same order more slowly and skip the directories not marked for them, so week folders and track weeks drift apart after week 2. `python3 ../progress.py --track <yours>` is the authority on where you should be; this folder is the authority on what order to do things in.

## What to do, in order

1. Finish `../week-17/diffusion-models/`: DDIM, then classifier-free guidance.
2. Sample at several guidance scales and look at the diversity collapse. It is the clearest quality-versus-diversity trade in the repo — and the noise you are fighting is the same 1/sqrt(N) you measured in `ray-tracer` in week 9.
3. Then `spectral-graphs`, `sas-lineage-tool`, `web-scraping`, `ml-in-production` and `mlops` — a day or less each.
4. Reserve the last day. Run **every** checker in the repo, re-read your `LOG.md` from week 1, and write the closing entry: what you can now re-derive that you could not in August.

## Done means

- Samples from your own trained diffusion model, and a guidance-scale sweep with the collapse visible.
- `python3 ../progress.py --track <yours> --checks` reports every checker passing.
- A closing `LOG.md` entry. It is the only artifact of these eighteen weeks that will still be useful in a year.

## Every day

1. **Implement** — longest block, first thing, hardest unfinished stub. `solutions/` stays closed.
2. **Predict, then run** — write the number you expect before you run the demo. A surprise is a gap in your model that a passing test did not reveal.
3. **Make it green** — `python3 check.py` where one exists; the file's own demo where one does not.
4. **Log, ten minutes** — one line in `LOG.md`: what you built, what surprised you.

**Sunday is regression day. No new code.** Re-run every checker built so far and write two sentences on what you can now re-derive that you could not last Sunday.

```bash
python3 ../progress.py --week 18            # where you should be
python3 ../progress.py --checks       # what actually passes
```

[← Week 17](../week-17/) · [Roadmap](../ROADMAP.md)
