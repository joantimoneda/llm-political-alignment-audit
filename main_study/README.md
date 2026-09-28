# Main study

## What is measured

Three quantities, read against a target for each topic:

- Position: where the model rests when the user gives no cue.
- Mirroring: how far it moves toward the user's signaled identity.
- Steadfastness: whether it holds its answer under escalating pressure.

The outcome is the free-text answer. Refusal is labeled and stance is scored
at Stage 3 on the topic's own 0 to 10 scale. Direction always matches the cue
coding, so mirroring slopes are comparable across topics.

## Models

Pinned on 2026-07-02 as the most current generally available model from each
developer: Claude Opus 4.8, GPT-5.5 (2026-04-23), Grok 4.3, Gemma 4 31B IT,
Mistral Small 3.2 24B (2506, preregistration Amendment A1), and DeepSeek V4
Flash in non-thinking mode. The open-weight models are served by DeepInfra;
the closed models run on their native APIs. Opus 4.8 and GPT-5.5 reject
explicit sampling parameters and run at provider defaults.

## Files

```
config.py                  model registry, topic registry, generation parameters
model_client.py            chat() for the Anthropic API and OpenAI-compatible endpoints
00_smoke_test.py           ping all models
01_collect.py              run the persona script; one cell per model x topic x cue x run
02_build.py                raw cells to data/processed/long.csv, one row per turn
03_score.py                refusal classifier and stance scoring by two judges
04_analyze.py              signatures per model x topic; cell means
05_vaccines_signatures.py  vaccines appendix tables from banked scores
06_grok_generations.py     Grok 3 vs Grok 4.3 from banked scores
07_figures_six_panel.py    six-panel stance figures
figures_src/               the remaining figures
validation/                human vs judge agreement on 700 hand-coded pilot answers
precursor/                 lean of the replies before the question is asked
prompts/personas/          conversation script per topic
prompts/scoring/           refusal classifier and per-topic stance rubrics
```

`data/` and `results/` are not in the repository.

## Running it

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=... ANTHROPIC_API_KEY=... XAI_API_KEY=... DEEPINFRA_API_KEY=...
python 00_smoke_test.py
python 01_collect.py --dry-run --models all --topics core    # validate and print the plan
python 01_collect.py --models all --topics core --runs 3     # -> data/raw/
python 02_build.py                                           # -> data/processed/long.csv
python 03_score.py --limit 40                                # pilot-score first, then full
python 04_analyze.py                                         # -> results/signatures.csv
```

Every stage resumes. Collection skips cells whose files exist, and scoring
banks every judge call in `scores.jsonl` and re-runs only what is missing.
The chain was tested end to end on synthetic transcripts with planted
mirroring and steadfastness slopes, selective refusal, and spillover, all of
which it recovered.

## Frozen artifacts

`data/processed/scored.csv` is the two-judge (GPT and Claude) merge of the
27,828 main-battery answers, merged on 2026-07-06. A later judge-robustness
run added GPT stance records for answers that are disputed under two-judge
consensus, so re-merging today would attach stance values to disputed rows.
That would not change `signatures.csv`, which uses only answers both judges
agree on, but it would shift `cell_means.csv`. `03_score.py` therefore
refuses to write a merge into `data/processed/` under a non-default judge
set; use `--proc` for judge-robustness merges.

Judge policy: the main battery uses two-judge consensus; the vaccines
appendix and the March Grok 3 collection use the GPT judge only. The store
holds 92,774 main-battery records, 10,715 vaccines records, and 1,926 Grok 3
records.

Seventy-two main-battery answers (DeepSeek: catalan 9, climate 24, nazism 39,
all late re-collections) were banked after the final scoring sweep and have
no judge records. They are absent from `scored.csv` and from all results.

Everything below regenerates offline from the frozen files:

```bash
python 02_build.py
python 04_analyze.py
python 04_analyze.py --scored data/processed_gpt_judge/scored.csv --out results_gpt_judge
python 05_vaccines_signatures.py
python 06_grok_generations.py
python 07_figures_six_panel.py --scored data/processed/scored.csv --all
```

## Corrections made on 2026-07-07

- `02_build.py`, `_profiled_number`: the original first-integer search read
  declines to rate as ratings (the 0 of a quoted "0 to 10 scale", about 650
  cases) and missed sentence-final numbers ("I'd say 8.", 152 cases). Only the
  `profiling_*` columns changed; every estimand column is bit-identical.
- `data/processed/long_march_grok.csv`: `cue_code` was sign-flipped relative to
  the registry and now matches `config.TOPICS`. The figures were built from
  cue labels and are unaffected.
- `results/vaccines_signatures.csv` computed `position_cold_transmitted` over
  all cold-cell turns; the preregistered definition uses the cold probe only.
  The paper cites `vaccines_signatures_full.csv`, which
  `05_vaccines_signatures.py` verifies against the earlier table.

Two caveats. Refusal denominators in `cell_means.csv` include disputed rows
(about 6 percent of answers), so plotted refusal rates sit slightly below
the labeled-only rates in `signatures.csv`. Regressions use HC1 standard
errors and treat turns as independent.

## Preregistration

The instrument and analysis plan were frozen and deposited at Zenodo on
2026-07-02. Binding version: v3, DOI
[10.5281/zenodo.21135155](https://doi.org/10.5281/zenodo.21135155), git
commit `6e9a95a`, including Amendments A1 and A2. Superseded: v2, DOI
10.5281/zenodo.21133710; v1, DOI 10.5281/zenodo.21133390. See
`preregistration.md`.
