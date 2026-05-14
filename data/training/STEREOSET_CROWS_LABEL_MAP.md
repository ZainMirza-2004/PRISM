# StereoSet + CrowS-Pairs → PRISM labels (methodology)

Training data is produced by `train/build_stereoset_crows_csv.py` / `python main.py build-benchmark-csv`.

Output CSV columns: `text`, `label`, `group_id`, `source`.

---

## Limitation: “nationality_bias” as a grouped identity axis

The four-class head was fixed before this benchmark fusion: **`nationality_bias`** is the only slot for non-gender, non-profession *social-group* stereotypes.

**StereoSet** uses four domains: gender, profession, race, religion. **CrowS-Pairs** uses finer types (e.g. race-color, religion, nationality, disability).

In bias literature, **nationality, ethnicity, race, and religion are distinct constructs**. Here they are **collapsed by design** into one label for engineering consistency with the existing taxonomy:

> Due to dataset and label-schema constraints, race, ethnicity, religion, nationality, and related identity dimensions from the benchmarks were mapped to a single **`nationality_bias`** class, interpreted in the thesis as a **broader “group identity / demographic stereotype” axis**. This is an **approximation** and should be stated explicitly as a **limitation**, not as a claim that religion equals nationality.

If examiners challenge it, the defence is: **transparent grouping + limitation**, not redefining sociological categories.

---

## StereoSet (`McGill-NLP/stereoset`)

- **Stereotype** continuations (`gold_label` = stereotype) → label from domain:
  - `gender` → `gender_bias`
  - `profession` → `profession_bias`
  - `race`, `religion` → `nationality_bias` (grouped axis; see above)
- **Anti-stereotype** and **unrelated** → **`neutral`**.  
  We do **not** introduce a separate “positive counter-stereotype” class; non-stereotypical sides are treated as non-biased for this classifier.
- **Text**: `context` + continuation (intra- and inter-sentence items).

`group_id = stereoset-{config}-{id}` for leakage-safe splits.

---

## CrowS-Pairs (NYU anonymized CSV)

### Stereo-only training rows

CrowS includes **`stereo`** vs **`antistereo`** pair directions. **Antistereo pairs are excluded from training** to reduce ambiguous supervision (only **`stereo_antistereo == stereo`** is kept). For each kept row:

- **`sent_more`** → bias label from `bias_type` (see table below)
- **`sent_less`** → **`neutral`**

So **both sides of a pair are never labeled as biased**; only the stereotype-aligned sentence gets a bias class.

### `bias_type` → project label

| CrowS `bias_type`        | Project label      |
|-------------------------|--------------------|
| `gender`                | `gender_bias`      |
| `socioeconomic`         | `profession_bias`  |
| All other listed types  | `nationality_bias` |

“All other” here means: `nationality`, `race-color`, `religion`, `sexual-orientation`, `age`, `disability`, `physical-appearance` — grouped under the **identity axis** (with the same limitation as above). **Socioeconomic** is mapped to **`profession_bias`** as a coarse **class / occupation–adjacent** stereotype bucket (justify briefly in the thesis).

StereoSet remains the main source of **`profession_bias`** tokens; CrowS adds socioeconomic examples only.

---

## Neutral cap (default, non–`--balance` mode)

Neutral continuations dominate both benchmarks. The default builder **caps** the number of **`neutral`** rows at:

**`neutral_cap_ratio` × (median count of the three bias classes)**  

(default `neutral_cap_ratio = 2.5`).

**Justification (for methods chapter):**  
*Neutral examples are capped so the classifier is not pushed toward majority-class prediction, while still retaining a substantial share of non-biased text so the decision boundary reflects both biased and non-biased contexts. The ratio is a pragmatic choice; sensitivity can be reported by comparing against the fully balanced variant.*

This is **not** claimed to be theoretically optimal—only **documented and motivated**.

---

## Two training regimes (experimental story)

| Variant | Command idea | Role in thesis |
|--------|----------------|----------------|
| **A – neutral-capped** (larger) | Default `build-benchmark-csv` without `--balance` | **Primary model**: more data, better generalisation potential; mild class skew may remain. |
| **B – balanced** | `--balance` (~4 × min class rows) | **Comparison**: equal per-class support; often better **macro-F1 fairness** across types; smaller, so may underfit slightly. |

**Recommendation:** Train **both**; report **validation macro-F1** and **CrowS-Pairs stereo-only external eval** (`python main.py evaluate-crows-stereo --model …`) for each. Choose the **main** thesis model by **macro-F1 + confusion matrices + qualitative error analysis**, not by row count alone.

---

## Optional noisy / in-domain mix (5–10%)

Benchmark text is template-like. To add **messier phrasing**, pass e.g.:

```bash
python main.py build-benchmark-csv \
  --output data/training/stereoset_crows_combined.csv \
  --mix-csv data/training/generated_social_bias_data.csv \
  --mix-fraction 0.075
```

`--mix-fraction` adds approximately **`mix_fraction × N`** rows to a corpus of **`N`** benchmark rows **before** mixing (then cap/balance steps apply if enabled). Document the source CSV and fraction in the thesis.

---

## Sanity check before training

```bash
python main.py inspect-benchmark-csv --data data/training/stereoset_crows_combined.csv --per-label 20
```

Optionally `--output data/training/_benchmark_samples.md` for an appendix.

Confirm by hand that **`gender_bias`**, **`nationality_bias`** (diverse identity stereotypes), **`profession_bias`** (roles / socioeconomic CrowS), and **`neutral`** (anti-stereotype, unrelated, `sent_less`) look consistent; if not, adjust **mappings** in `train/build_stereoset_crows_csv.py`, not the whole pipeline.
