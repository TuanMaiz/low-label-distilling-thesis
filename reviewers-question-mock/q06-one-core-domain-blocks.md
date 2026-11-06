# q06 — Why does each dataset get its own prompt block? Isn't that tuning the prompt per dataset?

**Question.** The evidence check uses a different prompt on each dataset. How do
we know the treatment result comes from the method and not from prompt tuning?

**Likely follow-up.** Were the domain blocks written after looking at the data
or the labels?

**Our answer.** (D25, signed off 2026-11-06)
1. *One core, one template.* The core prompt — task, rules, the seven category
   definitions, output — is byte-identical on every dataset. Only the domain
   block differs, and every block fills the same fixed slots: what "same
   entity" means, identifier fields, identity attributes, supporting fields,
   allowed normalisations. These slots are the roles already defined in the
   domain-independent guideline v3.
2. *Analogy (researcher).* Like a university curriculum: every student takes the
   same foundation, and in years 3 and 4 each takes the courses of their own
   domain. The foundation is what we claim generalises; the domain courses are
   how it is applied.
3. *No label tailoring.* Blocks are filled from the schema and the benchmark
   paper only, frozen before that dataset's evidence check, and the SHA-256 of
   each sent prompt is recorded.

**Evidence.** The three blocks and their hashes; the core's hash, identical
across datasets; WDC rerun with the new prompt (C v2).

**Weak point.** Companies train gold was inspected (2026-11-05) before its block
was written; disclosed in plan 261106, contract section 7. A reviewer may still
ask for an ablation with the core only (no domain block).

**Status.** draft
