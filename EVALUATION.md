# Evaluation (dissertation / reporting)

## Gold file format

Use CSV or JSON with:

- `text` (required)
- `label` (required): one of `gender_bias`, `nationality_bias`, `profession_bias`, `neutral`
- `post_id` (optional; auto-numbered if missing)

Example: `data/eval/gold_sample.json`

## What gets computed

- **4-way classification**: accuracy, precision/recall/F1 (weighted and macro) via `sklearn`, full `classification_report` in `metrics_multiclass.json`
- **Confusion matrix** (4×4): `metrics_multiclass.json` → `confusion_matrix`, plus a plain-text copy in `confusion_matrices.txt`
- **Binary** “any bias vs neutral”: P/R/F1 and 2×2 matrix in `metrics_binary.json`
- **Error analysis**: `errors.json` lists every case where `label ≠ predicted bias_type` (with text and model explanation)
- **Full run**: `full_predictions.json` (all system outputs for qualitative review)

## Command

From `PRISM/` — use **`--use-thresholds`** so the bundled threshold file is applied (same full setup as the write-up):

```bash
python main.py evaluate --use-thresholds
```

Defaults: `--gold data/evaluation/manual_eval_v3_400_posts.csv`, PRISM stack (`distilbert_B_balanced` + `distilbert_social_bias`). Override `--gold` only for custom benchmarks.

Without **`--use-thresholds`**, predictions use argmax on the fusion layer only (faster to try, not the same decision rule as the submitted configuration).

Optional — same defaults but omit thresholds (quick experiment):

```bash
python main.py evaluate
```

**HF token:** not required for a normal run: classifiers load from **`models/`** on disk.

For a tiny toy file instead:

```bash
python main.py evaluate --use-thresholds --gold data/eval/gold_sample.json --out data/output/eval_report
```

**Scale:** evaluation is **local** (DistilBERT + rules + fusion). Use a small gold file while debugging; the default benchmark is 400 posts.





## End-to-end smoke test (subset of gold, no OpenRouter)

Uses the first 100 posts of `manual_eval_v3_400_posts.csv` as `data/evaluation/workflow_smoke_100_posts.csv`, then:

```bash
# Hybrid evaluate (PRISM stack: B_balanced + social_bias aux + resolved meta joblib). Requires HF_API_TOKEN.
python main.py evaluate \
  --gold data/evaluation/workflow_smoke_100_posts.csv \
  --out data/output/workflow_eval_100_prism \
  --model models/distilbert_B_balanced \
  --aux-type-model models/distilbert_social_bias \
  --batch-size 4

python main.py evaluate-sanity \
  --sanity data/meta_training/sanity_slice.csv \
  --out data/output/workflow_sanity_eval \
  --model models/distilbert_B_balanced \
  --aux-type-model models/distilbert_social_bias

# Full six-system HTML report on the same 100 rows; --no-llm stubs OpenRouter baselines (no API key).
python scripts/build_evaluation_suite_report.py \
  --gold data/evaluation/workflow_smoke_100_posts.csv \
  --out data/output/comprehensive_evaluation_suite_workflow_100.html \
  --no-llm \
  --bootstrap 100
```

For the dissertation-scale suite with live LLM baselines, drop `--no-llm`, set `OPENROUTER_API_KEY`, and point `--gold` at your full benchmark CSV (e.g. 200- or 400-post files).
