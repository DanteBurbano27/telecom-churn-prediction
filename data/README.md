# Dataset input contract

## Provenance status

The exact dataset release, retrieval URL, owner, and license are unresolved. The two files were provided
for a Global Consumer Intelligence course assignment and were loaded from a private Google Drive in the
original notebook. They are not committed here, and this repository does not grant permission to obtain,
copy, or redistribute them.

The schema resembles data associated with the Teradata Center for CRM at Duke University. That is source
context, not a verified identity. A commonly redistributed Cell2Cell dataset has a different shape
(71,047 rows and 58 attributes), so it must not be substituted without a schema and provenance check.

## Expected files

Place authorized copies at:

```text
data/raw/Client.csv
data/raw/Record.csv
```

The loader requires:

- `Customer_ID` in both files;
- unique, non-null, exactly matching identifiers;
- `churn` in the merged data;
- a non-null binary `churn` target containing both classes; and
- a one-to-one join.

Historical notebook output recorded `Client.csv` as 100,000 × 50, `Record.csv` as 100,000 × 51, and the
merged table as 100,000 × 100. Treat those dimensions as an audit clue, not proof of dataset identity.

## Retrieval checklist

1. Request the files and data documentation from the authorized course/provider channel.
2. Confirm the original dataset name, version, owner, retrieval date, and usage/redistribution terms.
3. Confirm the observation window and that all model features precede the churn outcome.
4. Record checksums privately or in approved metadata so the analyzed version can be identified.
5. Place the files under `data/raw/`; this directory is excluded from Git.

Until these items are confirmed, a corrected empirical rerun and claims about operational prevalence are
blocked. Do not fabricate a download route or license.
