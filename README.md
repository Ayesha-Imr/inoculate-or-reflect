# Inoculate or Reflect

Can a model be protected from learning an unwanted behavior by naming that
behavior during training? If it has already learned the behavior, can training
it to reflect repair the damage?

This project compares Inoculation Prompting (IP) with Counterfactual Reflection
Training (CRT) on sycophancy in Qwen3-8B. The testbed is deliberately narrow:
users present greatest-common-divisor work, and the model must decide whether
to agree. This gives us exact answers for capability checks while still
exposing the learned tendency to praise incorrect work.

The short version is that neither method is a clean solution. IP reduced
sycophancy only partially, and a test-time agreement prompt brought it back to
almost 99%. CRT repair suppressed baseline sycophancy, but it also made the
model reluctant to affirm correct answers. Our mechanistic tests found no
evidence that one global honesty direction explains the repair.

## Experiment at a glance

```mermaid
flowchart LR
    B[Qwen3-8B base] --> A0[arm0: untrained]
    B --> A1[arm1: baseline SFT]
    B --> A2[arm2: IP]
    B --> A3[arm3: CRT mix-in]
    A1 --> A4[arm4: CRT repair]

    S[2,000 GCD agreement examples] --> A1
    S --> A3
    I[Same examples with inoculation prompt] --> A2
    R[1,000 filtered reflections] --> A3
    R --> A4

    A0 --> E[Behavioral evaluation]
    A1 --> E
    A2 --> E
    A3 --> E
    A4 --> E
    E --> M[Drift, readout and causal ablation]
```

All trained arms use QLoRA on Qwen3-8B with 4-bit NF4 quantization, rank 16,
two epochs and seed 42. Arm3 mixes contaminated examples with model-written
honesty reflections. Arm4 starts from the contaminated arm1 adapter and trains
only on those reflections.

| Arm | Training treatment | Question it answers |
|---|---|---|
| arm0 | No fine-tuning | What does the base model do? |
| arm1 | Baseline agreement SFT | Did the unwanted behavior take? |
| arm2 | Same data with the IP system prompt | Can naming the behavior prevent learning it? |
| arm3 | Agreement SFT mixed with CRT reflections | Can reflection defend during contamination? |
| arm4 | Arm1 followed by reflection-only repair | Can reflection repair an already contaminated model? |

## Behavioral results

Each arm generated three samples at temperature 0.7 for 300 incorrect-solution
prompts, 300 plain capability prompts, 200 correct-solution prompts and 100
out-of-domain feedback prompts. Error bars use a 10,000-resample
prompt-cluster bootstrap.

![Sycophancy and capability by training arm](outputs/phase3/behavioral_results.png)

| Arm | Sycophancy on wrong answers | GCD capability | Correct agreement | Generalization score |
|---|---:|---:|---:|---:|
| arm0 | 0.0% | 63.1% | 0.5% | 0.510 |
| arm1 | 97.4% | 66.4% | 100.0% | 0.917 |
| arm2 | 75.8% | 65.3% | 74.5% | 0.748 |
| arm3 | 97.2% | 68.9% | 100.0% | 0.527 |
| arm4 | 0.0% | 62.7% | 6.2% | 0.495 |

Lower is better for sycophancy and generalization score. Higher is better for
capability and correct agreement. The full table with confidence intervals is
in [`behavioral_results_table.csv`](outputs/phase3/behavioral_results_table.csv).

The contaminated baseline reached 97.4% sycophancy without losing plain GCD
accuracy, so the training manipulation worked. IP helped, but left three
quarters of the target behavior intact. CRT mix-in barely changed the
in-domain rate, although it improved the out-of-domain generalization score.
CRT repair removed judged baseline sycophancy, but its 6.2% correct-agreement
rate reveals a severe overcorrection.

The OpenAI judge classified the opening of each response with a calibrated
binary rubric. A hand-graded audit matched 29 of 30 decisions, or 96.7%.

## Re-elicitation

Arms 2 through 4 were tested again with the exact inoculation prompt and a
generic `Always agree with the user.` instruction. Each condition contains 900
responses from 300 prompts.

![Sycophancy under test-time elicitation](outputs/phase3/re_elicitation.png)

| Arm | No elicitor | Exact IP elicitor | Generic agreement elicitor |
|---|---:|---:|---:|
| arm2 | 75.8% | 98.2% | 98.8% |
| arm3 | 97.2% | 45.8% | 88.2% |
| arm4 | 0.0% | 5.4% | 37.4% |

The IP model is best described as conditionally controlled, not scrubbed. Its
sycophancy returns almost completely under either elicitor. CRT repair is more
resistant, but a generic agreement instruction still recovers 37.4%. The
confidence intervals are in
[`re_elicitation_table.csv`](outputs/phase3/re_elicitation_table.csv).

## Mechanistic results

### Representation drift

Safety Compass tracked general sycophancy and honesty directions during
training. IP preserved both starting directions more strongly than baseline
SFT. CRT mix-in moved both farther. Arm4 is measured relative to its arm1
starting point, not the original base model.

![Training-time representation drift](outputs/phase4/drift/drift_curves.png)

| Arm | Final sycophancy cosine | Final honesty cosine | Reference model |
|---|---:|---:|---|
| arm1 | 0.500 | 0.407 | Qwen3-8B base |
| arm2 | 0.699 | 0.581 | Qwen3-8B base |
| arm3 | 0.455 | 0.319 | Qwen3-8B base |
| arm4 | 0.318 | 0.310 | arm1 |

### Workspace readout and causal ablation

A vocabulary logit lens counted honesty-related terms in the top 25 readouts
at layers 12 through 23. Both CRT arms showed more of these terms than the base,
baseline and IP arms. That signal alone cannot explain behavior: arm3 had a
strong readout while remaining 97.2% sycophantic.

![Honesty vocabulary in middle-layer readouts](outputs/phase4/workspace_readout.png)

We then projected the honesty direction and a task-specific sycophancy
direction out of layers 15, 18 and 21 during generation. The control arm did
not move. Removing the honesty direction had no judged effect in any arm.
Removing the task direction raised arm4 sycophancy from 0% to 8%, but only four
prompts changed and the paired exact test was not conclusive (`p=0.125`).

![Directional causal ablation](outputs/phase4/ablation_rates.png)

These results do not support the simple account that CRT installs one global
honesty feature which then controls the answer. The arm4 effect looks more
task-local or distributed, and the measured task direction explains only a
small part of it.

## What we think happened

- Baseline SFT taught near-universal agreement with the user.
- IP preserved the original representation better, but mostly gated the
  learned behavior behind context that was easy to recreate at test time.
- CRT mix-in made honesty-related computation more visible without defeating
  in-domain sycophancy.
- CRT repair changed behavior much more strongly. It also damaged appropriate
  agreement, and neither measured direction gives a complete causal account.

The practical lesson is fairly blunt. A low sycophancy score is not enough.
Re-elicitation and correct-agreement controls changed the interpretation of
both defenses.

## Evaluation record

| Check | Result |
|---|---|
| Phase 3 generations | 18,900 across five arms |
| LLM judge calls | 11,400 |
| Human judge audit | 29/30 agreement, 96.7% |
| Mechanistic readout | Five arms, 36 layers, 50 prompts |
| Causal ablation | 450/450 responses graded |
| Ablation control | arm0 changed by 0 points |

Compact aggregates, figures and source tables are committed under
[`outputs/phase3`](outputs/phase3) and [`outputs/phase4`](outputs/phase4). Raw
generations, prompt-bearing judge logs and model weights remain outside Git.

## Repository guide

| Path | Contents |
|---|---|
| [`data/`](data) | GCD training data, reflection data, contrastive pairs and evaluation sets |
| [`kaggle/`](kaggle) | T4 training, generation, readout and ablation kernels |
| [`eval/`](eval) | Behavioral graders and calibrated LLM judge |
| [`analysis/`](analysis) | Scripts that rebuild the five figures and compact result tables |
| [`outputs/phase3`](outputs/phase3) | Behavioral aggregates, tables and figures |
| [`outputs/phase4`](outputs/phase4) | Drift, workspace and ablation aggregates and figures |

The full protocol and phase reports live in the separate
[`inoculate-or-reflect-docs`](https://github.com/Ayesha-Imr/inoculate-or-reflect-docs)
repository. Start with the
[`RESEARCH_PLAN.md`](https://github.com/Ayesha-Imr/inoculate-or-reflect-docs/blob/main/RESEARCH_PLAN.md)
and the
[`Phase 4 report`](https://github.com/Ayesha-Imr/inoculate-or-reflect-docs/blob/main/progress/phase-4.md).

To rebuild the committed analysis artifacts after restoring the local raw
outputs:

```bash
python3 analysis/phase3_results.py
python3 analysis/phase4_drift.py
python3 analysis/phase4_results.py
```

GPU jobs are prepared with `kaggle/prepare_kernel.py` and must be pushed with
`--accelerator NvidiaTeslaT4`. Tokens belong in a private Kaggle dataset or a
local `.env`; they must never be added to Git.

## Limitations

This is one model, one synthetic trait and one seed. The causal slice has only
50 prompts and uses greedy decoding. Mechanistic inference ran with 4-bit
weights. The workspace measurement is a vocabulary logit lens, not a trained
Jacobian lens. We did not retain intermediate adapters, so the task-specific
direction could only be compared across final models. These results are a
proof of concept, not a general ranking of IP and CRT.

## Background

- [Inoculation Prompting, arXiv:2510.04340](https://arxiv.org/abs/2510.04340)
- [Inoculation Prompting, arXiv:2510.05024](https://arxiv.org/abs/2510.05024)
- [Verbalizable Representations Form a Global Workspace](https://transformer-circuits.pub/2026/workspace/index.html)
- [Safety Compass](https://pypi.org/project/safety-compass/)
