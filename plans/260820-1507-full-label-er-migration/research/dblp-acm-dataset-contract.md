# DBLP-ACM Dataset Contract

## Status and authority

| Field | Value |
|---|---|
| Status | Frozen for DBLP-ACM integration Phase 1 |
| Date reset | 2026-09-01 |
| Parent experiment contract | `plans/260820-1507-full-label-er-migration/research/experiment-contract.md` remains draft |
| Dataset role | Candidate Dataset 2 |
| Researcher decision | Discard all previously reported dataset observations and derive the contract from a fresh local acquisition |
| Researcher freeze approval | Approved the observed snapshot, normalization, identity, test lock, and attribution decisions on 2026-09-02 |

All dataset facts below were calculated on 2026-09-01 from the locally extracted
official ZIP using `scripts/inspect_dblp_acm_source.py`. The independently
downloaded CSV copies were byte-identical to the archive members.

## Candidate source bootstrap

| Field | Acquisition value |
|---|---|
| Dataset index | <https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md> |
| Candidate source root | <https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/DBLP-ACM/exp_data/> |
| Official archive | <https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/DBLP-ACM/dblp_acm_exp_data.zip> |
| Candidate filenames | `tableA.csv`, `tableB.csv`, `train.csv`, `valid.csv`, `test.csv` |
| Source trust | Mutable host; the HTTP endpoint redirected to this HTTPS root; file contents remain untrusted until locally inspected |
| Raw storage | Ignored repository path under `data/raw/dblp_acm/` |
| Extracted archive root | `data/raw/dblp_acm/archive-2026-09-01/exp_data/` |
| Direct-download comparison root | `data/raw/dblp_acm/acquisition-2026-09-01/` |

The filenames and URL exist only to bootstrap acquisition. They do not imply a
schema, split mapping, count, checksum, or suitability decision.

## Local source observation

Logical version proposed from the archive directory timestamp and SHA prefix:
`deepmatcher-structured-dblp-acm-2018-06-29-a15b752f`.

| File | SHA-256 | Bytes | Data rows | Observed role |
|---|---|---:|---:|---|
| `tableA.csv` | `a83dfac196a4e263f3adac7aaf095c7198254a98fcaed0ec68d59130c74c43a7` | 344,698 | 2,616 | Left/DBLP records |
| `tableB.csv` | `bd103ffdccdff4d8b9d04c18d90d04110b70b6fc87dec83c4e52e9616c58431a` | 355,649 | 2,294 | Right/ACM records |
| `train.csv` | `ad94b36b178bbf76023d3cee689565fbda1fe01b19d9a3926a51db382f45f0a5` | 82,310 | 7,417 | 1,332 match / 6,085 non-match |
| `valid.csv` | `862f848ed3f3f005ae6c8997ecf571984bd575f6bbd319fc1a2170830a91132b` | 27,441 | 2,473 | 444 match / 2,029 non-match |
| `test.csv` | `e49adc4590d24c18b1a9bbd96011d9c745e10432e10e93e050d856a206fac394` | 27,429 | 2,473 | Locked; label/ID contents not audited |

Archive SHA-256: `a15b752ffc318a714690cf13286d31c2012f686525803ca803c392ceff4aa4f3`.
Archive size: 269,298 bytes.

Record headers are `id,title,authors,venue,year`; pair headers are
`ltable_id,rtable_id,label`. Both record ID columns are unique, numeric, and
contiguous from zero. Train/validation foreign keys resolve, labels are exactly
`0/1`, and both development splits contain only unique pairs. Test inspection
stops at hash, size, header, and row count.

`tableA.csv` has no missing fields. `tableB.csv` has 14 blank `authors` cells
and no other missing fields. Observed years span 1994–2003 in both tables.
Content excluding ID is not unique: table A has 25 duplicate-content groups/40
extra rows; table B has 16 groups/23 extra rows. Therefore content cannot serve
as entity identity.

No pair repeats between train and validation. Record overlap is observed and
must be allowed/reported:

| Split comparison | Left records | Right records |
|---|---:|---:|
| Train–validation | 1,058 | 1,027 |

The dataset-index URL and the HTTP-to-HTTPS redirect are retained only as
acquisition provenance. They are not evidence for any unobserved dataset fact.

## Proposed normalization contract

- Map `valid.csv` to normalized split `validation`; preserve source row order.
- Use `dblp:{raw_id}` and `acm:{raw_id}` record identities.
- Use `dblp_acm:{version}:{split}:dblp:{left_id}:acm:{right_id}` pair IDs and
  `dblp:{left_id}|acm:{right_id}` split-independent pair fingerprints.
- Map label `0` to false/non-match and `1` to true/match.
- Serialize attributes in observed order `title,authors,venue,year`; normalize
  blank values to null and render them as `<missing>`.
- Materialize only train and validation. Retain test hash/count/schema in the
  contract while keeping test JSONL absent and evaluation locked.

## Authorization boundaries

| Action | Current status |
|---|---|
| Download the five candidate files to ignored raw storage | Authorized by the researcher on 2026-09-01 |
| Calculate local hashes and inspect CSV structure/statistics | Authorized |
| Freeze the observed dataset contract | Approved on 2026-09-02 |
| Materialize normalized train/validation data | Authorized for Phase 2 only |
| Materialize ordinary-cache test JSONL | Prohibited by this plan |
| Make paid DBLP LLM calls | Not authorized; requires later cost review and explicit confirmation |
| Train or evaluate a DBLP compact model | Not authorized |
| Unlock or evaluate the official test split | Prohibited until the global final-test gate |

## License and attribution review

Attribution and license conclusions remain pending local source review. The
raw Leipzig collection and the candidate DeepMatcher-preprocessed files must
not be assumed to share identical licensing terms without evidence.

## Phase-1 human checklist

- [x] Researcher approved implementing the DBLP-ACM integration plan.
- [x] Researcher rejected the inherited hashes/statistics as contract evidence.
- [x] Researcher authorized a fresh acquisition for local inspection.
- [x] Official archive downloaded and extracted without overwriting the direct
  CSV downloads.
- [x] Local observation manifest is generated from the downloaded bytes.
- [x] Researcher reviews the observed hashes, schemas, counts, integrity audit,
  attribution wording, and proposed logical version.
- [x] After review, freeze this contract/profile and update Dataset 2 in
  `experiment-contract.md`.
