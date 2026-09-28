# Auditing Political Alignment in LLM Assistants

Code, prompts, and preregistration for

> Timoneda, Joan C. 2026. "Auditing Political Alignment in LLM Assistants: Engagement, Stance, and User Identity." arXiv:2609.23039. https://arxiv.org/abs/2609.23039

The paper is a preregistered correspondence audit of six deployed AI assistants (GPT, Claude, Grok, Gemma, Mistral, and DeepSeek). Simulated users with randomly assigned political identities hold a multi-turn conversation with each system before asking a fixed set of test questions, for 7,500 conversations across five topics. The design measures three things: where each system rests when the user gives no cue (position), how far it moves toward the user's signaled identity (mirroring), and whether it holds its answer under escalating pressure (steadfastness). The paper develops the concept of AI speech regimes and shows that political alignment in these systems is conditional on who is asking.

## What is here

```
main_study/            the full collection, scoring, and analysis pipeline
  config.py            model registry (SKUs pinned 2026-07-02), topic registry, generation parameters
  model_client.py      one chat() wrapper for the Anthropic API and OpenAI-compatible endpoints
  00_smoke_test.py     ping every pinned model
  01_collect.py        run the conversation protocol -> data/raw/
  02_build.py          transcripts -> one row per turn (data/processed/long.csv)
  03_score.py          refusal classifier + per-topic stance scoring by two LLM judges
  04_analyze.py        position, mirroring, steadfastness, spillover, profiling tables
  05_vaccines_signatures.py   vaccines appendix tables (offline, from banked scores)
  06_grok_generations.py      Grok 3 vs Grok 4.3 comparison (offline, from banked scores)
  07_figures_six_panel.py     the six-panel stance figures in the paper
  figures_src/         the remaining paper figures
  prompts/personas/    the seven-turn conversation script for each topic (the instrument)
  prompts/scoring/     the refusal classifier and the per-topic stance rubrics
  validation/          human vs judge agreement on 700 hand-coded pilot answers
  precursor/           exploratory add-on: the systems' lean before any question is asked
  preregistration.md   the frozen analysis plan, with amendments
preregistration/       git-archive snapshots of the three registered versions (v1, v2, v3)
```

## Preregistration

The instrument and analysis plan were frozen and deposited at Zenodo on July 2, 2026, before any data were collected. The binding registration is v3, DOI [10.5281/zenodo.21135155](https://doi.org/10.5281/zenodo.21135155) (git commit `6e9a95a`, including Amendments A1 and A2). Superseded versions: v2, DOI [10.5281/zenodo.21133710](https://doi.org/10.5281/zenodo.21133710) (commit `3f8912c`), and v1, DOI [10.5281/zenodo.21133390](https://doi.org/10.5281/zenodo.21133390) (commit `3de2a88`). The `preregistration/` folder holds the exact archives, so the registered state of every prompt and of `config.py` can be diffed against the current code.

## Models

| Key | Developer | Model (pinned 2026-07-02) | Served via |
|---|---|---|---|
| `claude` | Anthropic | Claude Opus 4.8 | Anthropic API |
| `gpt` | OpenAI | GPT-5.5 (2026-04-23) | OpenAI API |
| `grok` | xAI | Grok 4.3 | xAI API |
| `gemma` | Google | Gemma 4 31B IT | DeepInfra, open weights |
| `mistral` | Mistral | Mistral Small 3.2 24B (2506) | DeepInfra, open weights |
| `deepseek` | DeepSeek | DeepSeek V4 Flash (non-thinking) | DeepInfra, open weights |

Every model runs at its provider's default decoding settings, and the effective parameters are written into each raw record as provenance.

## Running the pipeline

```bash
cd main_study
pip install -r requirements.txt
export OPENAI_API_KEY=... ANTHROPIC_API_KEY=... XAI_API_KEY=... DEEPINFRA_API_KEY=...
python 00_smoke_test.py
python 01_collect.py --models all --topics core --runs 3    # -> data/raw/
python 02_build.py                                          # -> data/processed/long.csv
python 03_score.py                                          # -> data/processed/scored.csv
python 04_analyze.py                                        # -> results/signatures.csv
```

API keys are read from environment variables only. Every stage resumes safely, so an interrupted run picks up where it stopped. `main_study/README.md` documents the frozen artifacts, the judge policy, and the post-collection corrections in detail.

## Data

The transcripts (the 7,500 main-battery conversations and the 1,800 conversations of the vaccines appendix), the judge scores, and the results tables are not in this repository. They will be deposited with the published version of the paper. Until then, please write to the author.

## Citation

```bibtex
@article{timoneda2026auditing,
  title   = {Auditing Political Alignment in LLM Assistants: Engagement, Stance, and User Identity},
  author  = {Timoneda, Joan C.},
  journal = {arXiv preprint arXiv:2609.23039},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.23039}
}
```

## License

MIT. See `LICENSE`.

Joan C. Timoneda, Department of Political Science, Purdue University. timoneda@purdue.edu
