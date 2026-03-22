# Codex Prompt

You are Codex. Work inside my project and make the requested changes carefully.

## Task 1: Build a views summary table for `echo-eg`

Inspect the dataset `echo-eg`, which contains these two sub-datasets:
- `P10-MP4`
- `mimic-iv-echo-ext`

Create a table that shows:
- each sub-dataset
- each view/class found inside that sub-dataset
- the count for each view/class inside that sub-dataset

Also extend the table to include the classes/views that my pre-trained models were trained on for these two models:
- `EchoPrime`
- `echo-view-classifier`

### Requirements for Task 1
- Search the codebase and dataset structure to determine how views are named.
- Infer view labels from folder names, metadata, filenames, labels files, or existing preprocessing scripts if needed.
- Search the project for the definitions, config files, label maps, checkpoints metadata, docs, or source files that specify which classes/views `EchoPrime` and `echo-view-classifier` were trained on.
- Normalize obvious naming variants when presenting the table, but do not lose the original raw labels during inspection.
- Produce the final result as a clean markdown table.

### Desired output table structure
Use a table with columns similar to:
- `Sub-dataset`
- `Raw view label`
- `Normalized view label`
- `Count`
- `Used by EchoPrime?`
- `Used by echo-view-classifier?`

If a model uses a normalized class but not the exact raw label, mark usage based on the normalized class.

---

## Task 2: Modify external validation view normalization for PLAX

Update the external validation pipeline so that **any variation of PLAX** is treated as the same canonical view `PLAX` for the pre-trained models.

Examples include:
- `PLAX_VM`
- `PLAX_mv`
- `plax-something`
- any other PLAX-prefixed or PLAX-derived variation

All of these should map to:
- `PLAX`

### Requirements for Task 2
- Find where external validation assigns or compares view labels for the pre-trained models.
- Implement a normalization function or update the existing one so that all PLAX variants collapse to canonical `PLAX`.
- Apply the normalization consistently before label comparison, class lookup, inference routing, metrics, and reporting.
- Avoid breaking existing handling for other canonical views such as `A4C`, `A2C`, `PSAX`, etc.
- If there are multiple places doing label normalization, refactor to use one shared normalization helper if reasonable.

### Suggested normalization behavior
Use case-insensitive normalization.
Examples:
- any label that starts with `PLAX`
- any label containing `PLAX` as the core token

should normalize to exactly:
- `PLAX`

Be conservative so unrelated labels are not incorrectly remapped.

---

## What to deliver
1. The markdown table for `echo-eg` view counts and model-supported classes.
2. A concise summary of where you found the model class definitions.
3. The code changes for external validation.
4. A short note listing which files were modified.
5. A brief validation check showing that PLAX variants now resolve to `PLAX`.

## Important constraints
- Do not guess model classes without checking the codebase or metadata.
- Prefer existing project conventions over inventing new ones.
- Keep changes minimal and localized.
- Show the exact commands/scripts you ran to verify the result if applicable.

