# Remediation of MAgent Theory 2.0, 2026-10-29

The findings of `../REPORT.md` were implemented in the manuscript by ten file-scoped agents under `BRIEF.md`, in three waves. A-STD went first, to repair the chapter 05 standard. B-FOUND, B-DYN, B-CTX and B-LINKS ran alongside it. B-FUS, B-FORM, B-PA and B-IT7 then reclassified chapters 06 to 14 and Item 7. C-SUM came last and handled summaries and the bibliography. Each agent checked a finding against the text before applying it, and each per-finding record is in `logs/<agent>.md`. The manuscript was published to Research main as cdenn016/Vault#245 (merge 870addb). All seven PDFs built with no undefined references or citations. They are the main manuscript (388 pages), the supplement (14), the companion (137) and Items 1, 2, 3 and 7. The main manuscript keeps three overfull boxes.

The author's decisions D1 to D4 are recorded in `BRIEF.md`:

- **D1:** both nulls are declared.
- **D2:** the chapter 01 meaning of gauge stands.
- **D3:** the I02 replacement standard is adopted.
- **D4:** Item 7 gets a status statement.

## Counts by agent

| Agent | Applied | In part | Deferred | Rejected |
| :- | -: | -: | -: | -: |
| A-STD (ch05, s04, c05) | about 50 | 2 | 6 | 0 |
| B-FOUND (ch01, ch02, appendices, s01, s02, c01, c02, c06) | 11 | 28 | 6 | 0 |
| B-DYN (ch03, ch04, memory appendices, s03, c03, c04) | 49 | 9 | 9 | 0 |
| B-CTX (ch15, ch16, c15, c16) | 35 | 9 | 6 | 0 |
| B-LINKS (ch17 to ch19, Items 1 to 3) | about 70 | | 9 | 0 |
| B-FUS (ch06 to ch09) | 58 | 6 | 5 | 0 |
| B-FORM (ch10 to ch12) | about 75 | | 12 | 0 |
| B-PA (ch13, ch14) | 28 | 6 | 7 | 0 |
| B-IT7 (Item 7) | 31 | | 4 | 0 |
| C-SUM (abstract, outlook, notation, glossary, README, AUTHORING, future_TODO, bibliography) | see log | | | 0 |

Several lens fixes were not adopted as written, and the logs give the reasons. B-FUS reports (F5) as undetermined in chapter 08 rather than failing. B-FOUND and B-CTX narrowed some lens generalizations to the scope they could prove.

## Verdicts against the repaired standard

| Construction | Verdict | Notes |
| :- | :- | :- |
| Chapter 06 | Two-stage selection with exact inheritance | (F5) holds only through the exact-equality selector and fails for a normalized-centroid threshold family. (F4) and (F6b) fail. |
| Chapter 07 | No grouping | |
| Chapter 08 | Undetermined | |
| Chapter 09 | A retention only | |
| Chapter 10 | (F1), (F3) and (F4) hold | (F2), (F5) and (F6) are not addressed. |
| Chapter 11 | Two-stage selection | (F5) is split between the two nulls, so it fails. (F6b) fails against both. |
| Chapter 12 | Not shown to be a recursive formation | |
| Chapters 13 and 14 | Single-stage selection with exact inheritance | (F4) fails, since the erased coordinates are gauge. The frozen-environment control reproduces the selection. |
| Item 7 | Undetermined | |

No construction is claimed to meet every clause.

## Open author decisions

Decisions 5 to 17 of REPORT Section 10 remain open, apart from what D1 to D4 settled. The agents raised these further points. Verdicts that depend on them were written under A-STD's provisional readings, and each log says where:

- A-STD's provisional readings:
  - a (F5) verdict split between the two nulls counts as failing;
  - (F5) must hold for every constant of a declared set with nonempty interior;
  - the strong grade is redefined;
  - the frozen-environment control is reported, not a clause or grade;
  - importing R13 and R15 is undecided.
- The time-evolved reading of "gauge" in `def:observable-algebra` (B-FOUND).
- The constant sets declared for chapters 11 to 14, and whether the chapter 12 interval certificate joins the proof (B-FORM, B-PA).
- Whether a selector with no real constant is admissible under (F2), the chapter 08 partition rule, and whether symmetry-type (F4) witnesses count as compression (B-FUS).
- Whether a probe rule that withdraws hidden-angle queries at the handoff is allowed, and which structure group chapter 14 uses (B-PA).
- Parent anchors as constitutive data under (F6a), the Item 7 counterfactual class, and the working-document citation exception (B-IT7, B-LINKS).
- Decision 8, the abstract wording of the no-go; decision 7, the primary objective; decision 10, how to count the changing state; decision 13, the regime of the chapter 19 cohomological cost (B-DYN, B-LINKS, C-SUM).

## Not done in this round

- **Wave-1 follow-up:** the FOR OTHER OWNERS items that later agents addressed to wave-1 owners after those owners had finished were not run in a follow-up round. These are B-CTX's items for chapter 02 geometry, and B-LINKS's items for `def:shared-model`, chapter 03, chapter 05:1117 and the chapter 15 and 16 Wilson citations. They are listed in the logs.
- **preamble.tex:** the `\constitutiveheading` macro of XC4-CONS-12 belongs in `preamble.tex`, which no agent owned.
- **Unverified bibliography:** the venues of Ryabko2017 and Tishby1999 were not verified.
