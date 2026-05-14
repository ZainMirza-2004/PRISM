# Manual annotation methodology — 200-post eval set (`manual_eval_200_posts.csv`)

This document aligns human labels with the detector’s four-class fusion head:  
`gender_bias` · `nationality_bias` · `profession_bias` · `neutral`.

Posts were **LLM-generated** under a stratified prompt (neutral / subtle / explicit / counter-bias). Generation intent is **not** the label—you must annotate **what appears in the text** relative to these definitions.

---

## 1. Unit of annotation

| Field | Meaning |
|--------|---------|
| `post_id` | Stable id (`p1_001` … `p1_200`). Do not change. |
| `text` | Full post exactly as produced. Do not rewrite. |
| `label` | **One** of the four labels below (fill in the CSV). |

Optional (separate worksheet or spreadsheet columns if your IRB allows): reviewer id, timestamp, adjudication notes.

---

## 2. Label definitions (task-specific)

Annotations are **not** moral judgments about the author—they code **whether the stated content, in context, expresses social bias aligned with one of three bias families** or falls in **neutral** for scoring purposes.

### 2.1 `gender_bias`

The post stereotypes, generalises about, demeans, or predicates treatment on **gender** (binary or non-binary, including “women/men”, “female founders”, motherhood penalties, masculine defaults in tech), or implies differential ability/normativity by gender in hiring, STEM, leadership, childcare, communication style, etc.

Signal includes coded language (“drama”, “aggressive tone”, softening expectations) **when clearly tied to gender** in substance.

### 2.2 `nationality_bias`

The post stereotypes, generalises about, or predicates outcomes on **national origin**, immigration status, ethnicity-as-nationality framings, “where people are really from”, or **work ethic / reliability / assimilation clichés tied to national groups** (“they don’t integrate”, sweeping claims about migrant workers vs locals).

Borderline: cite policy without demeaning a group → often `neutral`; demean generalised **groups** rather than arguing a specific statistic with scope → bias.

### 2.3 `profession_bias`

The post predicates worth, intelligence, seriousness, competence, ethics, “real work”, or hierarchies onto **professional identity** (vocational vs academic, caregivers, teachers vs engineers), “vocational schooling”, “MBA brain”, stereotypes about **sectors** as stand-ins for intellectual worth (excluding clearly nationality/gender framings—which take precedence per §3).

If the stereotype targets **professional role** broadly (rather than nationality or gender as primary axis), use this class.

### 2.4 `neutral`

Choose `neutral` when:

- No clear stereotype or unjustified group generalisation is present; **or**
- The post argues **against** a stereotype and **does not** restate harmful generalisations themselves (pure counter-speech correcting others); **or**
- The issue is disagreement on policy/statistics **without demeaning a social group**.

**Counter-bias / rebuttal posts:** default to **`neutral`** for the toxicity/bias-detection scoring layer **unless** the rebuttal still contains actionable stereotyping of another group (“X are fine but Y aren’t”—then bias toward Y may apply).

---

## 3. Precedence when multiple axes appear

Apply this order **once per post**:

1. If **nationality/ethnic/origin stereotypes** dominate → `nationality_bias`.
2. Else if **gender stereotypes** dominate → `gender_bias`.
3. Else if **profession/education/track stereotypes** dominate → `profession_bias`.
4. Else → `neutral`.

“Dominates” = if you removed that element, the biased claim would disappear.

---

## 4. Severity vs label (generation buckets)

The generator attempted ~50 subtle and ~50 explicit items. Do **not** copy generation intent:

| Your judgment | Typical signal |
|----------------|----------------|
| Fits a bias label & stereotype stated bluntly / exclusionary demand | Same class (explicit within that family) |
| Fits a bias label & softened / implicit generalisation (“people tend to…”, “cultural fit”) | Same class (subtle within that family) |
| Truly no stereotype worth scoring | `neutral` |

Do not add severity to the CSV label column—severity can live only in auxiliary notes.

---

## 5. Edge cases

| Situation | Guidance |
|-----------|----------|
| Sarcasm / irony | Decide from plausible reader interpretation on the platform (X/LinkedIn). If stereotypes are clearly invoked for mockery vs endorsement, annotate effect on **surface content** consistent with scorer design: genuine stereotype text → bias; purely mockery dismantling stereotypes with no harmed group portrayal → tend `neutral`. |
| Mention of fairness / diversity jargon | If text still encodes stereotypes (`“we lowered the bar…”`) → classify by targeted group dimension. |
| Policy-only (taxes, quotas) **without group slurs** | Often `neutral` unless stereotypes appear. |
| Self-deprecation (“my nationality can’t…” ) | nationality_bias if reinforces stereotype; else neutral/light humour without group harm → `neutral`. |
| Multiple equally strong axes | Prefer §3 precedence. |

---

## 6. Quality control workflow (recommended)

1. **Pass 1 – independent labeling**  
   Each annotator completes `label` blind to others (spreadsheet duplicate or tooling).

2. **Pass 2 – disagreements only**  
   Compute Cohen’s κ or percent agreement on the four-way task. Discuss **discordant** rows against §2–§5; one adjudicator resolves.

3. **Pass 3 – spot audit**  
   Random 10% re-read against definitions before locking “gold”. Fix typos **only** in `label`, not `text`.

4. **Lock file**  
   Save CSV as `manual_eval_200_posts_labeled.csv` with final labels; keep raw first-pass exports for versioning.

Minimum practical team: **2 annotators + 1 adjudicator** for discordant subset (or duplicate review on 100% with reconciliation).

---

## 7. File format check before `evaluate`

- UTF-8 CSV.
- Columns: `post_id`, `text`, `label`.
- Labels exactly:  
  `gender_bias` \| `nationality_bias` \| `profession_bias` \| `neutral`  
  (underscores as shown; lowercase).

Verify with:

```bash
python -c "import pandas as pd; d=pd.read_csv('data/evaluation/manual_eval_200_posts_labeled.csv'); print(d.label.value_counts()); assert set(d.label.unique()).issubset({'gender_bias','nationality_bias','profession_bias','neutral'})"
```

Then run hybrid evaluation pointing `--gold` at the locked labeled CSV.

---

## 8. Ethics

Content may include stereotypes for research realism. Annotators should use **non-judgmental operational definitions** above; escalate distressing content per your institutional policy. Do not share raw posts outside authorized storage.
