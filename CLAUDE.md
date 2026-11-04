# CLAUDE.md

## Project overview

Course project for **TDT13** (NTNU). Approved title:

> **Does Medical Domain Tuning Help? MedGemma 1.5 vs. Gemma 3 for Extracting Gleason Scores from Prostate Cancer Pathology Reports**

**Research question:** Does a language model fine-tuned on medical text (MedGemma 1.5) extract Gleason scores more accurately than the general model it is built on (Gemma 3), when both are used zero-shot with the same prompt?

### Background

When a prostate is examined for cancer, a pathologist writes a free-text report. Its most important piece of information is the Gleason score: the two most common tissue patterns are each graded 3–5 and written as e.g. `3 + 4 = 7` (higher means more aggressive cancer). The phrasing varies a lot in practice (`Gleason grade 4+3`, `Primary pattern: 3/5, Secondary pattern: 4/5`, ...), and the reports are scanned documents converted to text with OCR, so they contain typos and noise. The task is an NLP information-extraction problem: locate and normalize one specific fact in varied free text.

## Data

- **Reports:** TCGA-Reports (Kefeli et al., 2024), 9,523 public pathology reports as machine-readable text. Only the **prostate subset (446 reports)** is used.
  - Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC10935496/
  - Data: https://github.com/jkefeli/tcga-path-reports (`TCGA_REPORTS.csv.zip`)
- **Labels:** Gleason scores from the TCGA clinical data for the same patients (`GLEASON_PATTERN_PRIMARY`, `GLEASON_PATTERN_SECONDARY`, `GLEASON_SCORE`), from cBioPortal study `prad_tcga`: https://www.cbioportal.org/study/summary?id=prad_tcga
  - Join reports to labels on the TCGA patient ID. All 446 reports have been verified to have a label.
- Class imbalance: score 7 is more than half the data.
- **Label leakage:** the model input is only the report text and the fixed prompt. Never put the TCGA clinical data in the prompt (the `GLEASON_*` columns or any other clinical column). Feed the OCR text as it is, without manual cleaning, so all three methods see the same noisy input.

## Methods

1. **Baseline:** hand-written regex that looks for `X + Y` patterns.
2. **MedGemma 1.5 4B-it** (Google, Jan 2026), a Gemma 3 model fine-tuned on medical text and images. Text only here. https://huggingface.co/google/medgemma-1.5-4b-it
3. **Gemma 3 4B-it**, the general base model, for direct comparison. https://huggingface.co/google/gemma-3-4b-it

Both LLMs get the **identical zero-shot prompt** asking for a JSON answer (primary and secondary pattern). No fine-tuning, inference only, run on **IDUN** (NTNU's HPC cluster, Slurm).

Keep the comparison fair: same prompt, same decoding settings (prefer greedy / temperature 0), same input truncation, same JSON parsing. Any change to one model's setup must be applied to the other.

Both models are gated on Hugging Face; a token with accepted licenses is needed.

## MedGemma notes from an earlier project

These notes come from an earlier project that used MedGemma 1.5 4B-it on MRI images (TDT4501). They describe how the model behaves in practice. They are **not** results, and they must not be cited or reused in this project's report.

- **It reasons by default on text-only prompts.** MedGemma 1.5 writes its reasoning first, between `<unused94>thought` and `<unused95>`, with the answer after `<unused95>`. This happened on every text-only prompt, even when the prompt asked for JSON only. Always parse the text after `<unused95>`; this is harmless when the marker is missing.
  - A reasoning answer took about 400–650 tokens, and up to about 900. With reasoning, `max_new_tokens` must be high (e.g. 1000), or the JSON gets cut off.
  - The earlier project had a `block_thinking` option that suppressed the reasoning. Check how it was implemented before reusing it.
  - **Fairness:** Gemma 3 does not reason this way. Decide explicitly whether MedGemma's reasoning is blocked or allowed, use the same `max_new_tokens` for both models, and state the choice in the report. Reasoning may be part of what "domain tuning" adds, so running MedGemma both with and without reasoning is a possible extra condition.
- **Reasoning can hallucinate.** On text-only clinical prompts, the reasoning invented "criteria" that were never given and did not apply them consistently. Watch for this in the error analysis, e.g. patterns that don't appear in the report.
- **Greedy decoding is reproducible.** Repeated runs with greedy decoding gave identical raw outputs.
- **Chat template pitfall.** The chat template joins consecutive text parts in one message with nothing in between. Put the whole prompt (instructions + report) in a single text part, separated by blank lines.
- **Token probabilities in 16-bit are coarse.** If you read probabilities from the logits (e.g. for confidence or calibration), compute them in float32; the bf16 scores rounded probabilities to steps of 0.125.
- **Small models are sensitive to the prompt format.** The output format in particular can change the answers noticeably. Keep the prompt fixed once the evaluation starts, and tune it only on a small held-out set of reports, not on the reports you evaluate on.

## Run logging

Every evaluation run is saved so that each number in the report can be traced to the exact setup that produced it.

- `results/<subset>/`: one folder per subset of reports (e.g. a small debug/prompt-tuning subset and the full evaluation set).
  - `v<N>.csv`: one row per report, with the patient ID, the method/model, the raw model output, the parsed primary/secondary/score, a status (`valid` / `not_found` / `invalid`) and the label.
  - `v<N>_config.json`: all settings of the run: method (regex / MedGemma / Gemma 3), exact model ID, the full prompt text actually sent, decoding settings (`max_new_tokens`, greedy, reasoning blocked or not), date and a free-text note.
  - `v<N>_scores.json`: accuracy, macro-F1, pattern-pair exact match, rates of not found / invalid / hallucinated answers, and the confusion matrix.
  - `runs.csv`: one row per run, with its note, what changed since the previous run, and its scores. Always rebuilt from the config and scores files, never edited by hand.
- A `log_run` setting: `False` means a test run, and nothing is saved.
- A logged run picks the next free `v<N>` and prints what changed in the config since the last run.
- Never overwrite or rename old runs.
- `results/` is **committed to git** (not ignored), so the runs can be inspected on GitHub. TCGA is public data, so this is fine.

## Code structure

- The code is written as Jupyter notebooks (`.ipynb`), run on IDUN.
- Every notebook has the same top layout:
  1. **Imports cell:** all imports in one cell.
  2. **Config cell:** a single `config` dict with every setting that may change between runs (method, model ID, prompt, decoding settings, subset, `log_run`, note, ...). No settings are hard-coded further down.
  3. The rest of the pipeline.
- The `config` dict is what gets saved as `v<N>_config.json`, so the logged config always matches what was run.

## Environment (IDUN)

- NTNU's IDUN HPC cluster. Software is loaded with Lmod (`module avail`, `module load ...`).
- Project directory on IDUN: `/cluster/home/hermaos/GleasonExtraction`. This repo is checked out there.
- Model weights: the default Hugging Face cache, `~/.cache/huggingface` (i.e. `/cluster/home/hermaos/.cache/huggingface`). MedGemma 1.5 4B-it is already there from TDT4501. New models (e.g. Gemma 3 4B-it) are downloaded to the same cache with `from_pretrained`, so no path needs to be set in the code.
- Python env: conda env `master_thesis` (Python 3.11) at `~/.conda/envs/master_thesis`, shared with the TDT4501 project. Its packages are listed in that project's `environment.yml`, so adding or upgrading a package here also affects TDT4501. Check before changing pinned versions.
- **Pin torch to a CUDA 12.x build** (e.g. `torch==2.14.0+cu126`). IDUN's GPU driver only supports up to CUDA 12.9, and a plain `pip install torch` installs a CUDA 13 build that cannot see the GPU.
- GPU sessions and Slurm jobs are started by the author.

## Evaluation

- **Accuracy** and **macro-F1** on the Gleason score (macro-F1 because of the class imbalance).
- **Exact match on the pattern pair** (3+4 and 4+3 both sum to 7 but mean different things).
- **Rate of invalid or hallucinated answers** (unparseable JSON, patterns outside the valid range, primary + secondary inconsistent with the score, etc.).
- **Short qualitative error analysis:** OCR noise, unusual phrasings, reports covering multiple tissue samples.

Apply the same metrics to the regex baseline so all three methods are comparable.

### Reports without a Gleason score, or with several

Every patient has a label, but not every report necessarily states a Gleason score in its text: it may be missing, or lost to OCR noise. Some reports cover several samples with different scores.

- The JSON schema allows a "not found" answer (`null` for primary and secondary). The regex baseline returns the same.
- Score "not found" separately from invalid and hallucinated answers. It counts as wrong for accuracy and macro-F1, but it is an honest abstention, not a hallucination.
- A *hallucination* is a score the model gives when the report contains none, or patterns that appear nowhere in the report.
- Decide **before** the evaluation runs which score counts as correct when a report has several, and state it in the report. The regex uses a hard-coded rule (highest score); the LLM prompt instead describes the wanted score (the overall score of the examined specimen in the final diagnosis, not biopsy/history scores). See "Multi-score rule" under "Findings so far".
- Estimate how many reports really lack a score, e.g. by checking the regex's "not found" cases by hand. That number is the ceiling for any method and is worth reporting.

## Findings so far

Results to use in the report. Each number can be traced to a logged run or to the data notebook.

### Data (`notebooks/01_data.ipynb`)

- 446 prostate reports, all with a label; primary + secondary always equals the score.
- Score distribution: 6: 45, 7: 237, 8: 57, 9: 103, 10: 4. Pairs: 3+4: 144, 4+3: 93, 4+5: 74, 4+4: 47, 3+3: 44, 5+4: 29, 3+5: 5, 5+3: 5, 5+5: 4, 2+4: 1.
- One label has pattern 2 (2+4), so the valid pattern range in `common.py` is 1–5, not 3–5.
- Split (seed 13, stratified on score): 30 tuning reports, 416 eval reports. The tuning set has no score 10.
- Reports are 388–10,928 characters (mean ≈ 3,700).

### Multi-score rule

**Regex:** decided on the tuning set, before any eval run: when a report has several different pairs, the answer is the **highest** score (highest sum, ties broken by the higher primary pattern). On the tuning set this was right in all 3 multi-score reports; "first mentioned" got one wrong (0.967 vs. 0.933 accuracy).

**LLMs:** decided before any LLM run, deliberately **different from the regex**. The prompt does not give a hard-coded rule; it describes which score is wanted: the overall Gleason score of the specimen examined in the report, as given in the final diagnosis; not scores from earlier biopsies or the clinical history; the dominant tumor if separate nodules are graded; tertiary patterns ignored. Rationale: understanding which score the report means is exactly what a language model can do and a regex cannot, so this is part of what is being compared. It also matches what the TCGA label is (the pathologic Gleason score of the resected specimen). The tuning set already shows these cases (EJ-5522: per-zone biopsy scores in the history; FC-A6HD: several nodules, dominant 4+3; tertiary patterns in V1-A8MG, KK-A8IH). Note for the report: the post-hoc check of the regex on the eval set (below) also showed that biopsy/history scores are the main source of regex errors, so the wording was not chosen blind to the eval set; say so.

### Reasoning (decided before any LLM run)

The main comparison is regex vs. MedGemma 1.5 vs. Gemma 3, **all LLMs with reasoning blocked** (`block_thinking=True`, i.e. the `<unused94>` token is banned with `bad_words_ids`, for both models). Reason: the task is locating and copying a score, which should not need reasoning, and it is much faster. MedGemma with reasoning allowed may be added later as an extra condition. Both models use `max_new_tokens=1024`, greedy decoding, no input truncation (longest prompt: 2,834 tokens).

### LLM tuning runs (`results/tuning/v1`, `v2`, 30 tuning reports)

Prompt with the "final diagnosis" wording, reasoning blocked, greedy, `max_new_tokens=1024`. About 17 new tokens per answer, 0.4 min per run.

| Run | Model | Accuracy | Macro-F1 | Pair exact match | Invalid |
|---|---|---|---|---|---|
| v1 | Gemma 3 4B-it | 0.900 | 0.943 | 0.900 | 2 |
| v2 | MedGemma 1.5 4B-it | 0.967 | 0.973 | 0.967 | 0 |

- MedGemma's only miss is X4-A8KQ (label/report disagreement, see the regex section), so it is correct on every tuning report whose label matches the text.
- Gemma 3's 2 invalid answers are the same error: with the notation **"GLEASON SCORE 9 (4+5)"** (total written before the pair) it put the total in `primary` (`{"primary": 9, "secondary": 5}` for KK-A59V; `{"primary": 7, "secondary": 3}` for KK-A8IH, "7 (4+3) WITH TERTIARY GLEASON GRADE 5"). Both reports have extra numbers near the score (an OCR fragment "5/3+2", a tertiary grade 5). Four other KK reports with the same notation were correct. Not hallucinations: all numbers are in the report, just put in the wrong field.
- Both models sometimes wrap the JSON in a ```` ```json ```` code block (MedGemma always); the parser handles it.
- **Decision: the prompt was not changed after these runs.** Adding a sentence about the "total first" notation was considered and rejected, so the difference is kept as a finding about the models rather than tuned away. The prompt in `results/tuning/v1_config.json` is the final prompt for the eval runs.

### Regex baseline (`results/eval/v1`, 416 eval reports)

The regex was developed on the 30 tuning reports only (29/30 correct there; the one miss is TCGA-X4-A8KQ, whose report says 4+3 but whose label is 4+5).

| Metric | Value |
|---|---|
| Accuracy (score) | 0.928 (386/416) |
| Macro-F1 (score) | 0.858 |
| F1 per score | 6: 0.95, 7: 0.963, 8: 0.822, 9: 0.938, 10: 0.615 |
| Pattern-pair exact match | 0.892 (371/416) |
| Not found | 4 (0.96 %) |
| Invalid | 0 |

Macro-F1 is pulled down by score 10 (4 reports): five reports labelled 9 were predicted as 10.

**What the regex gets wrong (45 wrong pairs on eval):**

1. **Wrong choice among several pairs: 29 reports** (7 of them still have the right score). The label pair is in the report, but the "highest" rule picks another one. Typical causes:
   - a score from the earlier **biopsy or clinical history** ("biopsy revealed Gleason 4+4=8", "Gleason 5+5 all cores"), e.g. EJ-5504, EJ-5527, J9-A52B, EJ-AB20;
   - a **minor component** next to the main score ("predominant pattern 7 (3+4) with a component of 8 (4+4)", VN-A88P);
   - separate **lobes or nodules** with different scores.
2. **Label not among the pairs found: 12 reports.** Most are label/report disagreements, where the report clearly states a different pair than the TCGA label: CH-5745 (report 4+3 four times, label 3+4), CH-5748, EJ-5502, TK-A8OK (`7=4+3`, label 3+4), V1-A8WN, V1-A9OA, V1-A9Z9, V1-A9ZG, and V1-A9OH (report internally inconsistent: "5+4, primary pattern 5, secondary pattern 3"; label 4+4). CH-5765 is a regex miss: an amended "Remark. 3+4=7" without the word Gleason nearby. J9-A8CP and V1-A8X3 were not checked in detail.
3. **Not found: 4 reports.**
   - 2 reports contain **no Gleason score at all** (G9-A9S0, G9-A9S4; the text seems truncated and starts in the middle of the synoptic form). No method can get these right.
   - 2 are **OCR misses**: `GLEASON SCORE (PRIMARY + SECONDARY PATTERN) : 3 4 =. 7/10` (the `+` is lost, J4-A83N) and `Primary Pat t er n: Grade 3. Secondar y Pat t ern: Grade 4` (OCR space inside "Secondary", KC-A7F6).

**Room for an LLM:** about 32 eval reports are recoverable in principle (29 choice errors + 2 OCR misses + CH-5765). The 9 clear label disagreements (up to 11 with J9-A8CP and V1-A8X3) and the 2 reports without a score cannot be fixed by any method, so the best reachable pair exact match on eval is about 0.97 (403–405/416), against 0.892 for the regex. The regex is still a strong baseline on the score itself (0.928 accuracy), so an LLM that picks the wrong score or gives invalid answers can easily end up below it.

**Post-hoc check (not used; done once on all 446 reports after v1 was logged):** other selection rules would have done better — pair / score correct out of 446: highest 400 / 415, first 402 / 420, highest ignoring biopsy/history context 405 / 418, **most frequent pair 418 / 430**. These were found by looking at the eval set, so the rule was **not** changed; v1 with "highest" is the baseline. This can be mentioned in the report as a limitation of the regex and as evidence that choosing among several scores needs context.

## Deliverables and deadlines

- **Written report:** about **2,000 words**, deadline **13 November 2026**.
- **Presentation:** 8–10 minutes in class.
- **Code:** not submitted. It is pushed to GitHub and shown to the professor on request, so keep it readable and the results reproducible (fixed seeds, saved prompts, saved raw model outputs).

Given the short timeline, favour simple, working code over abstractions.

## Working preferences

- **Comment the code.** Give every function a docstring saying what it does, its inputs and its outputs. Add inline comments where the logic is not obvious. Aim at a reader who can program but does not know this codebase.
- **Heavy programs are run by the author, not Claude.** Claude runs on the IDUN login node, which is shared and has no GPU. Anything heavy (loading or running the LLMs, full evaluation runs, anything that needs a GPU or a lot of CPU/memory) is started by the author once they have a GPU session or Slurm job. Claude prepares the code and says what to run. Light tasks are fine for Claude to run: reading files, small data checks with pandas, the regex baseline on a few reports, quick syntax checks. Keep them short and small, and do not start long-running or parallel processes on the login node.
- **No git operations that change history or the remote.** Never run `git commit`, `git push` or `git pull`; the author does these. Read-only commands such as `git status`, `git diff` and `git log` are fine.

## Relation to the master's thesis

The author's master's thesis (supervisor: Michael Staff Larsen) uses MedGemma 1.5 to detect prostate cancer from MRI images. This project uses the same model family but only on text, with a different task (information extraction, not image classification) and a different dataset. There is **no overlap** in experiments or results: nothing from the thesis is reused in this project, or the other way round. Do not mix thesis code or results into this repo.
