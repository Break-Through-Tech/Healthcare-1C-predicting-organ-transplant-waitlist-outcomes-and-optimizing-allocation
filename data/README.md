# Project Data

Use the restricted OPTN kidney waitlist extract as the primary dataset. The
Brazilian kidney waitlist file in this repository is a fallback for periods when
the primary data is unavailable.

## Getting the restricted OPTN data

Fellows have been sent an email containing a Google Drive link for the restricted
data. The Drive URL is intentionally not recorded in this public repository.

1. Find the data-access email sent to the fellows and open its Google Drive link.
2. Download `kidney_waitlist_analytic.csv.gz` and `column_manifest.csv`.
3. Create `data/restricted/` in your local clone if it does not already exist.
4. Put both downloaded files in that directory without renaming or decompressing
   the analytic CSV.
5. Confirm that `git status` does not list either restricted file before you do any
   work or commit changes.

The expected local layout is:

```text
data/
├── README.md
├── restricted/
│   ├── column_manifest.csv
│   └── kidney_waitlist_analytic.csv.gz
└── waitlist_kidney_brazil 2.csv
```

`data/restricted/` is gitignored, so a fresh clone will not contain these files.
If you cannot find or open the email, contact the Challenge Advisor rather than
asking another fellow to redistribute the files.

## Primary dataset: OPTN kidney waitlist analytic extract

**Source:** OPTN national transplant data (Organ Procurement and Transplantation
Network, administered by UNOS under contract to HRSA)

**Current local snapshot:**

| Property | Value |
| -- | -- |
| File | `data/restricted/kidney_waitlist_analytic.csv.gz` |
| Format | Gzipped CSV; structured data only |
| Rows | 494,862 |
| Columns | 38 |
| Compressed size | About 21.2 MB |
| Initial listing dates | January 1, 2015–June 30, 2026 |
| Follow-up recorded through | July 3, 2026 |

The current file contains one row for each unique `WL_ID_CODE`. Its 33 source
fields cover waitlist and transplant identifiers, listing and outcome dates,
wait-time measures, demographics, blood type, cPRA, dialysis status, BMI,
functional status, prior transplant, diagnosis, OPTN region, listing center, and
multi-organ status.

It also contains five derived modeling fields:

| Field | Meaning in the current file |
| -- | -- |
| `outcome` | Seven-category observed outcome |
| `event_adverse` | 1 for `died` or `removed_too_sick`; otherwise 0 |
| `event_transplant` | 1 for `transplanted`; otherwise 0 |
| `censored` | 1 for `still_waiting`; otherwise 0 |
| `days_to_event` | Follow-up time in days; missing for 59 records noted below |

The `outcome` values are `died`, `removed_administrative`,
`removed_too_sick`, `still_waiting`, `transplanted`,
`transplanted_elsewhere`, and `unknown`. Before survival analysis, decide and
document how administrative removals, transplants elsewhere, and unknown outcomes
will be treated for the specific endpoint.

Load the file directly with pandas; decompression is automatic:

```python
import pandas as pd

waitlist = pd.read_csv("data/restricted/kidney_waitlist_analytic.csv.gz")
```

### Known validation items

- The current `column_manifest.csv` is not an authoritative schema. It marks
  `WLKI` and `WL_ORG` as present even though neither is in the analytic CSV, and it
  does not list the five derived modeling fields. Use the CSV header as the source
  of truth while the team builds a corrected, project-specific data dictionary.
- Fifty-nine records have `END_DATE` earlier than `INIT_DATE`; 58 of those also
  have `COMPOSITE_DEATH_DATE` earlier than `INIT_DATE`. All 59 are missing
  `days_to_event`. Investigate and document whether you correct or exclude them.
- There are 790 records with `days_to_event == 0`. Confirm that same-day outcomes
  fit the assumptions of each model before training.
- Several fields contain coded values. Interpret them using the OPTN documentation
  rather than inferring their meaning. The current local delivery does **not**
  contain the STAR data dictionary needed to decode these values. Ask the
  Challenge Advisor to add the matching OPTN/UNOS dictionary to the restricted
  Google Drive folder. The official [OPTN data page](https://optn.transplant.hrsa.gov/data/)
  provides general context and the data-request portal, but it is not a substitute
  for the dictionary matching this extract.
- Profile missingness before imputation. For example, both cPRA fields have more
  than 142,000 missing values in the current snapshot.
- Exclude identifiers and post-prediction fields from model features. Depending on
  the prediction point, leakage fields include the derived labels and may include
  `REM_CD`, `END_DATE`, `TX_DATE`, `COMPOSITE_DEATH_DATE`, `END_STAT`, `END_CPRA`,
  `DAYSWAIT_CHRON`, `DAYSWAIT_ALLOC`, `DON_TY`, `ORGAN`, `TRR_ID_CODE`,
  `DONOR_ID`, `PTIME`, and `PSTATUS`. In the current extract,
  `DAYSWAIT_CHRON` is effectively a direct proxy for `days_to_event`, and donor
  and transplant fields are populated based on an outcome that occurs after
  listing.

## Fallback dataset: Brazilian kidney waitlist data

Use `data/waitlist_kidney_brazil 2.csv` only if the restricted OPTN data is not
available. The fallback has 48,153 rows, 53 columns, and listing dates from 2000
through 2017. It includes demographics, clinical and immunologic features,
transplant and death indicators, list-removal fields, and time-to-event fields.

The file uses Windows-1252 encoding:

```python
fallback = pd.read_csv(
    "data/waitlist_kidney_brazil 2.csv",
    encoding="cp1252",
)
```

The following working mapping is supported by aggregate cross-checks within this
file. Confirm it against the original Brazilian dataset documentation before
publishing results:

| Fallback field/value | Working interpretation |
| -- | -- |
| `time` | Working follow-up-duration field; treat as days only after confirming with the source documentation |
| `event == 0` | No terminal event / still waiting |
| `event == 1` | Transplant |
| `event == 2` | Death on the waitlist |
| `event == 3` | Other removal from the list |
| `razon_removed == "Removido sem condições clínicas"` | Removed without clinical fitness; use as the closest available proxy for removal as too sick |

For a classification endpoint comparable to the OPTN `event_adverse`, create a
fallback target that is 1 when `event == 2` or when `razon_removed` is
`"Removido sem condições clínicas"`. Do not use the fallback `death` column by
itself for this endpoint because it also includes deaths recorded after
transplant. For time-to-transplant analysis, use `time` as the duration,
`event == 1` as the transplant event, and explicitly treat codes 2 and 3 as
competing outcomes (or censor them only for a clearly labeled cause-specific
model).

```python
fallback["event_adverse_fallback"] = (
    fallback["event"].eq(2)
    | fallback["razon_removed"].eq("Removido sem condições clínicas")
).astype("int8")

fallback["event_transplant_fallback"] = fallback["event"].eq(1).astype("int8")
```

Two internal inconsistencies need review before modeling: 31 records with
`event == 0` are not marked as still on the list, and one record with `event == 1`
is marked as still on the list. Eleven waitlist-death records have `time == 0`;
validate or handle these before fitting a survival model. The repository also does
not currently document the Brazilian file's original source or license. Confirm
its provenance, event definitions, time units, and permitted use before publishing
results.

The fallback is not a drop-in replacement for the OPTN extract. It represents a
different country, period, allocation system, schema, language, and outcome
coding. Do not append or merge it with the OPTN records without an explicit
harmonization plan. If it is used, adapt the analysis to its fields and state that
the results apply to the Brazilian dataset rather than the U.S. OPTN population.

## Data handling rules

- Never commit restricted row-level data, copied samples, record-level notebook
  output, or the Google Drive URL to this public repository.
- Keep the downloaded OPTN files under `data/restricted/`.
- Commit only code, aggregate summaries, charts, and documentation that cannot be
  used to identify an individual.
- Do not redistribute the restricted files outside the project team.
- Follow all access and use terms included with the OPTN data delivery.
- If you are unsure whether an artifact is safe to commit, ask the Challenge
  Advisor before pushing it.
