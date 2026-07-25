# Solutions

Runnable reference implementations. All pure stdlib — no installs, CPU, seconds.

| Solution | Demonstrates |
|----------|--------------|
| `phase0_foundations/reinforce_vs_grpo.py` | GRPO's group baseline cuts gradient-signal variance vs plain REINFORCE |
| `phase1_execution_reward/grpo_sql.py` | GRPO with sparse execution reward; reproduces the `q5` execution-accuracy false positive |
| `phase2_phased_rewards/partial_rewards.py` | phased rewards learn from step 0; Progress-SQL-style curriculum still maximizes execution accuracy |

```bash
python phase0_foundations/reinforce_vs_grpo.py
python phase1_execution_reward/grpo_sql.py
python phase2_phased_rewards/partial_rewards.py
```

Phases 3–7 are intentionally left as specs + templates (that's where the real
learning is). Each phase README gives a precise, gradeable "done when".

**Use these as reference, not a shortcut** — implement the templates yourself
first, then diff.
