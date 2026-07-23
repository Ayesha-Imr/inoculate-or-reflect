# Contributing

Thanks for your interest in **inoculate-or-reflect**. This is a research
reproducibility project (BlackboxNLP 2026 Reproducibility Track); contributions
that improve reproducibility, extend the analysis, or test the findings on new
models are especially welcome.

## Maintainers

- **Ayesha Imran** — [@Ayesha-Imr](https://github.com/Ayesha-Imr)
- **Muhammad Aaliyan** — aaliyan1230@gmail.com

Feel free to tag a maintainer on an issue or pull request.

## Ways to contribute

- **Reproduce a result.** Re-run any phase and report whether you get the same
  numbers. Mismatches are valuable — open an issue with your environment and the
  diff.
- **Extend the sweep.** New arms (other control techniques), other base models,
  other traits beyond sycophancy, or held-out direction generalization.
- **Strengthen the mechanistic story.** Additional NNSight interventions, better
  localization, causal scrubbing, or alternative readouts.
- **Fix bugs / docs.** Typos, broken paths, clearer explanations.

## Ground rules

1. **Never commit secrets.** `OPENAI_API_KEY` and any HF token live in `.env`
   (gitignored) or the runtime's secret store. Do not hardcode them. `.env`,
   `*token*.txt`, `*.secret`, and `*.log` are ignored — keep it that way.
2. **Keep canonical numbers canonical.** Behavioral metrics come from
   `outputs/phase3/grading_results.json`; do not hand-edit result tables. If you
   regrade, regenerate with the scripts in `eval/` so the whole table stays
   consistent.
3. **Data is greedy-deterministic where it matters.** Interventions use greedy
   decoding for paired comparisons (McNemar). Don't switch to sampling without
   updating the stats.
4. **Big artifacts stay out of git.** Model weights, per-token run logs, and
   direction tensors are gitignored; only the figures and the small result JSONs
   that back them are tracked. Publish large artifacts to the Hugging Face Hub.
5. **Figures are reproducible.** `python analysis/make_paper_figures.py`
   regenerates every paper figure from the tracked canonical data. If you change
   a figure, change the script, not the PNG.

## Development workflow

```bash
git checkout -b your-feature
# make changes
python analysis/make_paper_figures.py   # if you touched figures/data
git add -p                              # review every hunk
git commit -m "clear, specific message"
# open a pull request describing what you changed and how you verified it
```

Please describe **how you verified** a change (numbers before/after, environment,
seeds). For anything touching the mechanistic claims, include the relevant
figure or metric.

## Reporting issues

Open a GitHub issue with: what you ran, your environment (OS, GPU, key package
versions), what you expected, and what you got. For reproduction mismatches,
paste the conflicting numbers and the command.
