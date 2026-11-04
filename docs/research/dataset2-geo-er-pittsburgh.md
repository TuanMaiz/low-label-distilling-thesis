---
title: Dataset 2 — Geo-ER Pittsburgh (OSM–Foursquare), replaces DBLP-ACM
date: 2026-11-04
status: decided (D24); open items listed below
source: researcher decision notes, 2026-11-04 (figures as measured by the researcher)
---

## Dataset 2 decision: replace DBLP-ACM with Geo-ER Pittsburgh (OSM–Foursquare)

### Why DBLP-ACM is replaced
- DBLP-ACM labels are old (public since about 2010), and its 7,417 training pairs are too costly to label and train within the budget (three teacher passes, evidence judge, GPU training).
- The replacement must come from a group outside Mannheim, so the datasets don't all come from one research group (WDC Products and MaDI-Bench Companies are both Mannheim), and have official splits near 2,000–2,500 training pairs. The teacher-cutoff criterion is relaxed for this dataset.
- This changes the approved DBLP-ACM Dataset-2 contract. Record it as a new decision in `docs/research/decisions.md`.

### Source: Geo-ER (Geospatial Entity Resolution)
- Balsebre, Yao, Cong & Hai, "Geospatial Entity Resolution", WWW 2022 (ACM Web Conference), DOI 10.1145/3485447.3512026. NTU Singapore. Peer-reviewed.
- Repo: https://github.com/PasqualeTurin/Geo-ER (last commit 2023-04-27). Data: `data/train_valid_test/osm_fsq/pit/{train,valid,test}.txt`
- The paper introduces 8 datasets (OSM–Foursquare and OSM–Yelp in Singapore, Edinburgh, Toronto, Pittsburgh). Pittsburgh is the only city near the target size; the others have 8,700–14,000 training pairs.
- **How pairs were built:**
  - Records were collected through the OpenStreetMap, Foursquare and Yelp APIs.
  - Candidates were kept if name edit similarity ≥ 0.6 and distance ≤ 2,000 (the data confirms meters: the largest distance is 1,996 m). So nearly every pair has similar names.
- **Labels:** "four human annotators" marked the matches; no agreement score is reported.
- **Splits:** random 50/20/30 train/valid/test, keeping the match share the same in every split. These are official files, not resampled by us.
- **Domain:** places (shops, restaurants, parks, transit stops). It's new to the thesis and a good test of a domain-independent guideline, because the deciding evidence is location rather than product or company facts.

### Splits (measured from the files)
| Split | Pairs | Matches |
|---|---|---|
| Train | 2,500 | 727 (29.1%) |
| Validation | 999 | 290 (29.0%) |
| Test | 1,502 | 437 (29.1%) |

- No duplicate pairs, no pair in more than one split, no label conflicts.
- Val and test have the same match rate as train, so the threshold should carry over without the mismatch problem Music had.
- (Alternative source pair, same city: OSM–Yelp, 2,558 / 1,022 / 1,536 at 31.7% matches. More addresses filled, but 1 duplicate pair and 1 pair appearing in both val and test.)

### Records
- 5 fields per record: name, latitude, longitude, address, postalCode.
- Fill rates: name and coordinates 100%; address 52%; postal code 54%.
- Records come pre-serialized inside each pair line: `COL name VAL ... COL latitude VAL ... <TAB> COL name VAL ... <TAB> label`. There are no separate record tables and no record IDs, so nothing leaks through IDs.
- **Pair length:** about 300 characters (~75 tokens), the shortest of all datasets (WDC ~262 tokens, Companies ~95). Cost is about $1–2 per teacher pass, scaled from the WDC cost ($2.69 for 2,500 pairs).
- **Distance between the two places**, computed from the coordinates on train:
  - matches: median 42 m, 90th percentile 155 m, max 861 m;
  - non-matches: median 876 m, 10th percentile 138 m;
  - 124 of 1,773 non-matches (7%) are within 100 m, e.g. different shops in the same building.
- **Examples:**
  - Match: "Donner House" at 40.44184, -79.94030, Margaret Morrison Street ↔ "Donner House" about 25 m away, Margaret Morrison St, at Carnegie Mellon University.
  - Non-match: "Brentwood Library" ↔ "brentwood Care", about 1.4 km apart.

### License
- The repo has no license file. The paper asks users to cite it.
- Source data came through APIs: OSM data is ODbL; Foursquare's API terms are restrictive.
- Same practice as FullContact: use it for thesis experiments, don't republish records, report aggregate results, and add one sentence in the thesis on the data's terms.

### Preprocessing (proposed, same rules as Companies)
1. Parse the `COL <field> VAL <value>` format into fields name, latitude, longitude, address, postalCode, and serialize in the usual `- field: value` JSONL format.
2. Raw values with light cleanup only: empty strings and `NULL` become `<missing>`.
3. Keep A = OSM and B = Foursquare, the order in the files.

### Threshold protocol (same as the other datasets)
- Current macro-F1 rule (D14, `utils/classification_threshold.py`), tuned on the full official validation set (999 pairs) and applied to test unchanged. No resampling.
- Match F1 is the primary metric; macro F1 is supplementary.

### Risks to state
- **Thin records:** 5 fields, half the addresses empty. Name similarity alone won't separate pairs, because every pair passed a name-similarity filter.
- **Distance has to be inferred from raw coordinates.** The teacher LLM can estimate it; a small student cross-encoder (e.g. Qwen3-Reranker-0.6B) may not. The original Geo-ER model used an explicit distance embedding.
- **Labels:** 4 annotators, no agreement reported.
- **Old data:** labels public since 2022, so the teacher may have seen them (accepted for this dataset).

### Open decisions
1. Confirm the source pair: OSM–Foursquare (recommended) or OSM–Yelp.
2. Distance: keep raw coordinates only (consistent with the raw-values rule), or add a computed `distance_m` field to each pair. Adding it is a deterministic transform that uses no labels, but it departs from raw values. Suggested: start raw; revisit only if a pilot run shows the student can't use coordinates, and if so apply the same rule to every arm.
3. Write the Geo-ER adapter: parse, serialize train/validation JSONL, keep test locked. Simpler than the DBLP-ACM adapter.
4. Update the schedule: the 3-week DBLP-ACM phase becomes the Geo-ER phase.

---

## Backup (4th candidate): Machamp Books or Movies (Megagon Labs)
Use only if Geo-ER fails (e.g., the student can't learn from coordinates).

- Wang et al., "Machamp: A Generalized Entity Matching Benchmark", CIKM 2021 (arXiv:2106.08455). Repo https://github.com/megagonlabs/machamp, **BSD 3-Clause license**. Outside Mannheim.
- Both tasks are rebuilt from Magellan project datasets (Univ. of Wisconsin) and have official splits:

| Task | Domain | Train / Val / Test | Matches (train / val / test) | Left records | Right records | Ditto test F1 (paper) |
|---|---|---|---|---|---|---|
| Semi-HETER | Books | 1,240 / 414 / 414 | 474 / 157 / 159 (38%) | 22,133, JSON (title, pubyear, publisher) | 23,264, JSON (title, pages, authors list, nested publication_info) | 0.616 |
| Semi-Rel | Movies | 1,309 / 437 / 437 | 545 / 181 / 183 (42%) | 29,179, CSV (IMDB-style: title, director, actors, year, rating, information) | 32,823, JSON (Rotten Tomatoes-style: Name, YearRange, ReleaseDate, Director, Genre, Url…) | 0.911 |

- **Format:** split files hold `left_index, right_index, label`; the index refers to the position in the left/right record array. Check whether the index is 0- or 1-based before joining.
- **Different from the other datasets:**
  - schemas differ across sides (heterogeneous and nested JSON), so there is no shared field list;
  - below the 2,000-pair target;
  - Books is hard (Ditto 0.616);
  - Movies records contain URLs (Rotten Tomatoes `Url`, IMDB IDs) that may reveal the answer; drop them.
- **Label provenance:** inherited from the Magellan datasets, which were labeled by students in a class project. Not documented further.