# Dataset 3 decision record — MaDI-Bench Companies

Researcher decision, 2026-10-28, saved verbatim from the chat.

## Dataset 3 decision: MaDI-Bench Companies, entity matching subtask only

### Selection criterion
- The matching labels must be published after the teacher's knowledge cutoff (GPT-5.6 Sol, 16 Feb 2026), so the teacher cannot have memorized the gold labels.
- Old source records are acceptable: any large LLM knows popular entities. State this as a limitation.
- About 2,000 training pairs is enough; the number is not strict.

### Source: MaDI-Bench
- Steiner, Peeters & Bizer, arXiv:2606.30371, v1 29 Jun 2026. Repo https://github.com/wbsg-uni-mannheim/MaDI-Bench, commit 062058d (9 Oct 2026). Website https://wbsg-uni-mannheim.github.io/MaDI-Bench/
- Preprint, not peer-reviewed, with no venue stated. Cite it as a preprint together with the repo commit.
- The repo is ahead of the paper: it has a 4th pipeline and different fusion counts.
- **Checked against the files:** split sizes and match rates agree with the paper's table (1,971 / 948 / 599, 29.4% test matches). No duplicate pairs, no pair in more than one split, and no pair labeled both ways.
- **Authors:** the Mannheim group of Christian Bizer (DBpedia, Web Data Commons). Ralph Peeters built WDC Products (EDBT 2024). Aaron Steiner is first author of the inspiration paper (arXiv:2606.28823). That paper tests Abt-Buy, Walmart-Amazon, WDC Products, DBLP-ACM and DBLP-Scholar, not MaDI-Bench.
- **Independence:** DBLP-ACM is the only dataset not from this group. Keep it.
- **Extension argument:** Companies adds a domain the inspiration paper did not test, on a benchmark released after the teacher's cutoff.

### Matching step only
- MaDI-Bench is an end-to-end data integration benchmark: schema matching → normalization → blocking → entity matching → fusion.
- Using one step alone is an intended use: the paper's stated goal includes "measuring the step-wise as well as the end-to-end performance".
- Entity matching has its own labeled train/val/test splits and its own F1 metric. The sets focus on near-boundary pairs, draw candidates from several blocking strategies, and have hand-verified test sets.
- **Thesis wording:** "We use the entity matching subtask of MaDI-Bench Companies with its official train/validation/test splits and the gold schema mapping supplied with the benchmark. As in MaDI-Bench, we train one model on both source pairs and report F1 per source pair and their mean. Blocking, normalization and fusion are out of scope."
- **No pipeline run needed.** The data is static files in `use cases/companies/base/input/`:
  - labeled pairs (IDs + label only): `entitymatching/forbes_2_dbpedia_{train,val,test}.csv`, `entitymatching/forbes_2_fullcontact_{train,val,test}.csv`
  - records: `data/forbes.csv`, `data/dbpedia.csv`, `data/fullcontact.csv`
  - column mapping: `schemamatching/sm_mapping_gold.json`
- **Preparation:** join pair IDs to source records, rename columns with the gold mapping, drop IDs, and write pairs in the usual serialized JSONL format. A trial join found every ID except the 3 known broken DBpedia URIs.
- The authors' pipelines fed the matcher normalized tables. We read the raw tables.

### Records
- **Forbes** (Global 2000, 2014), 2,000 records: name, country, industry, assets, revenue. Labeled CC BY 4.0 in the benchmark metadata.
- **DBpedia**, 10,085 records: name, founded, country, city, industry, keypeople, assets, revenue. Fill rates: key people 9%, assets 7%, revenue 20%, industry 68%. CC BY-SA 3.0.
- **FullContact**, 1,931 records: name, country, city, founded, keypeople. Collected via the FullContact API in 2014. No license. Columns anonymized (`Attribute_1`…`6`).
- Source data covers 2012–2014 and was first published in 2016.

### Splits (headerless CSVs `id1,id2,label`; Forbes is always id1; labels in mixed case, lowercase before comparing)
| | Forbes–DBpedia | Forbes–FullContact | Pooled |
|---|---|---|---|
| Train / Val / Test | 455* / 219 / 140 | 1,513 / 729 / 459 | 1,968 / 948 / 599 |
| Matches (train / val / test) | 49% / 48% / 45% | 31% / 32% / 25% | test 29.4% |
| Fields filled on both sides (mean) | 3.4 of 8 | 1.8 of 8 | |
| Non-matches with name word overlap ≥ 0.5 (train / test) | 28% / 36% | 4% / 5% | |
| Test records also in train | 51 of 272 (19%) | 420 of 763 (55%) | |
| Authors' test F1 (Ditto / Magellan) | 0.923 / 0.938 | 0.889 / 0.848 | |

- All splits are official, and val and test have similar match rates.
- \*3 training pairs point to cut-off DBpedia URIs (Workday, Waste Management, Caesars Entertainment), documented by the authors; drop them.
- The two source pairs are different sub-tasks:
  - Forbes–DBpedia has richer evidence and hard non-matches with similar names.
  - Forbes–FullContact mostly comes down to the name; its hard cases are true matches with different names (brand vs legal name).
- The Forbes–DBpedia test set (140 pairs, 63 matches) is small, so its F1 will be noisy.
- Pair length about 380 characters (~95 tokens), similar to DBLP-ACM. Cost about $1–2 per teacher pass for 1,968 pairs, scaled from the WDC cost ($2.69 for 2,500 pairs).

### Label quality
- **Test:** looks sound in a 30-pair spot check per source pair. This fits the README's claim that test was verified by hand. Many test non-matches are easy.
- **Train/val have clear errors**, found by checking records matched to more than one company. About 13 were found among ~1,350 train/val matches, and the true rate is higher:
  - Hermès International ↔ Hengan International (train)
  - Toho Bank / Toho Holdings ↔ DBpedia "Toho", the film studio (train/val)
  - Six Mitsubishi companies ↔ one "Mitsubishi Corporation" (train/val)
  - Ping An Bank ↔ Ping An Insurance (train/val)
  - AMP ↔ TE Connectivity (val)
  - Health Care REIT ↔ Ventas (val)
- **The parent/subsidiary rule is inconsistent.** Petronas subsidiaries and Carnival ↔ Carnival Cruise Line are labeled matches; Piraeus Bank ↔ Piraeus Bank Beograd and Carrefour ↔ Carrefour Market are labeled non-matches.
- **Framing:** train gold is imperfect, which supports the thesis framing that LLM labels only need to come close to human labels. These errors are what uncertain-pair detection should surface. Val errors affect threshold tuning slightly; state this as a limitation.

### License
- The MaDI-Bench repo has **no license file at all** (code or data). "All benchmark artifacts are available for public download" means downloadable, not licensed for reuse.
- FullContact metadata names FullContact Inc. as creator and has no license.
- FullContact's own terms (current Services Agreement; "previous version" Terms of Use, so the 2014 version is unverified):
  - FullContact owns its data;
  - copying, distributing or republishing is forbidden;
  - API data may not be transferred to third parties;
  - there is no research exception.
- No original public link exists for the FullContact table; the MaDI-Bench repo is the only source.
- **Practice:** using the downloaded data for thesis experiments is normal research use. Do not republish FullContact records (no appendix dumps, no serialized pairs in a public repo). Report aggregate results only, and add one sentence in the thesis about the data's terms.

### Decisions
1. **Schema mapping: yes.** Rename columns with the gold mapping; values unchanged. Results assume correct column alignment.
   - Forbes: company → name, region → country, business_segment → industry, asset_value → assets, sales_figure → revenue
   - DBpedia: org_name → name, established → founded, nation → country, headquarters → city, sector → industry, keypeople_name → keypeople, total_assets_val → assets, annual_income → revenue
   - FullContact: Attribute_2 → name, _3 → country, _4 → city, _5 → keypeople, _6 → founded
   - Drop record IDs and Forbes's `url` column (both contain the company name).
2. **Values: raw, with light cleanup only**, consistent with WDC and DBLP-ACM. FullContact's literal `null` city (92 records) and empty strings become `<missing>`.
   - Known raw-value issues to mention:
     - countries "United States of America" (Forbes) vs "United States"; "Taiwan, Province of China[a]";
     - revenue in plain dollars in Forbes but mixed units in DBpedia (`65170000000` vs `2.8` / `50.2`);
     - different industry vocabularies;
     - year-only founding dates, with 66 DBpedia records at `1970-01-01` (possible placeholder);
     - merged city text in DBpedia.
   - The authors' matching F1 comes from normalized inputs, so use it as a reference point only. The fair comparison is our own gold-label arm trained on the same raw inputs.
3. **Threshold: keep the current macro-F1 rule (D14, `utils/classification_threshold.py`)**, the same as WDC and DBLP-ACM.
   - Tuned on the full official validation set (948 pairs) and applied to test unchanged. No resampling.
   - Ties are broken by match F1, then accuracy, then closeness to 0.5.
   - Val and test match rates are close, so the threshold should carry over (Forbes–FullContact 32% vs 25% is the larger gap).
   - The authors' Ditto maximizes match F1 instead; note the difference.
4. **Pooling: one model on both source pairs**, following the authors.
   - Their Ditto preparation appends both source pairs into one train and one val set, trains one model, and scores each source pair's test set separately.
   - Always report F1 per source pair plus the mean. The per-pair F1 shows whether one sub-task suffers from the mix.

### Next steps
1. Write the Companies adapter: join pairs to records, rename columns, apply light cleanup, serialize train/validation JSONL, and keep test locked. Follow the DBLP-ACM preparation pattern.
2. Optional: email the authors about the license of the benchmark and the FullContact table; check the university's data policy.
3. Not inspected: a "hard" variant (1,503 / 800 / 527) with LLM-rewritten values and generated companies.