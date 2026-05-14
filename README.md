# PRISM

**PRISM** flags four-way social bias in short English text (**gender**, **nationality**, **profession**, or **neutral**). It combines a fine-tuned **DistilBERT** classifier with extra lexical and structural signals, then a small **learned fusion** layer

---

## 1. Setup (once)

Open a terminal, go to the project folder that contains **`main.py`** (the inner **`PRISM/`** directory), then:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Python:** 3.10–3.12 works well with the pinned libraries in `requirements.txt`.  
**Hardware:** runs on **CPU** by default; a GPU is optional. Allow some disk space for the `models/` checkpoints.

---

## 2. Run evaluation on the bundled benchmark

Use **`--use-thresholds`** so the run picks up the bundled per-class decision file (`models/meta_fusion/optimal_thresholds.json`) together with the shipped fusion model. That is the **full** evaluation configuration used in the write-up—skip this flag only for quick experiments.

```bash
python main.py evaluate --use-thresholds
```

That uses the default **400-post** gold file (`data/evaluation/manual_eval_v3_400_posts.csv`) and writes results under **`data/output/eval_report/`**.

**Where to look:** open **`metrics_multiclass.json`** for accuracy and F1 scores (including **`f1_macro`**), plus **`metrics_binary.json`**, **`confusion_matrices.txt`**, **`errors.json`**, and **`full_predictions.json`** for details.

To send outputs somewhere else:

```bash
python main.py evaluate --use-thresholds --out data/output/my_run
```

---

## 3. Try your own labelled file

Same label names as the benchmark: `gender_bias`, `nationality_bias`, `profession_bias`, `neutral`. Columns: **`text`**, **`label`**, optional **`post_id`**. See **[`EVALUATION.md`](EVALUATION.md)** for the exact format.

```bash
python main.py evaluate --use-thresholds --gold path/to/your_file.csv --out data/output/eval_custom
```

---

## 4. Run on unlabelled posts (batch predictions)

Input: JSON or CSV with a **`text`** column (and optional **`post_id`**).

```bash
python main.py analyze --dataset path/to/posts.json --output data/output/predictions.json --use-thresholds
```



---

## 5. Optional: `.env`

Only if you need it:

| Variable | Purpose |
|----------|---------|
| `OPENROUTER_API_KEY` | Live LLM baselines in the HTML report (omit and use `--no-llm` otherwise). |

`main.py` loads **`.env`** automatically when `python-dotenv` is installed.

---

## 6. Tests

```bash
python -m unittest discover -s tests -p 'test_*.py' -q
```

---

## Where the important pieces live

| Path | What it is |
|------|------------|
| `main.py` | Entry point: `evaluate`, `analyze`, training commands, etc. |
| `data/evaluation/manual_eval_v3_400_posts.csv` | Default gold benchmark |
| `models/distilbert_B_balanced/` | Main DistilBERT classifier |
| `models/meta_fusion/*.joblib` | Trained fusion weights (chosen automatically when you don’t override them) |
| `models/meta_fusion/optimal_thresholds.json` | Per-class thresholds (loaded when you pass **`--use-thresholds`**) |

If **`evaluate`** errors about missing files, make sure this submission includes the full **`models/`** tree (or restore it from the project archive).

More evaluation detail → **[`EVALUATION.md`](EVALUATION.md)**.
