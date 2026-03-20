# Trouble Shooting

## 1. CardiacNet preprocessing output collisions

Status: fixed in preprocessing code.

What was changed:
- CardiacNet `sample_id` is now generated from relative path + stable hash, not just `f.stem`.
- This prevents collisions for repeated stems such as `PHI01`.

Affected file:
- `prepare_external_validation_datasets.py`

Action required:
1. Regenerate CardiacNet processed inputs.
2. Rerun CardiacNet evaluation for:
   - `echo-view-classifier`
   - `EchocardiographyQC`
   - `EchoPrime`

## 2. Dataset-level macro F1 over full label space

Status: improved metric reporting.

What was changed:
- Kept strict full-label-space macro metrics (backward compatible).
- Added present-label-only macro metrics (`support > 0`) in summaries and CSV outputs.

New/updated metric fields:
- `macro_f1_all_labels`
- `macro_f1_present_labels`
- (and corresponding precision/recall variants)
- `n_present_labels`

Affected file:
- `External_Val_Models/validation_utils.py`

## 3. EchocardiographyQC results completeness

Status: improved coverage diagnostics.

What was changed:
- QC runner now writes `dataset_coverage.csv` with expected vs matched counts and coverage ratio.
- Added `--fail-on-incomplete` to make partial coverage fail fast.

Affected file:
- `External_Val_Models/EchocardiographyQC-master/run_external_validation.py`

## 4. EchoPrime full-pipeline caveat

Status: explicitly labeled as simplified view-only evaluation.

What was changed:
- Added explicit evaluation mode metadata/caveat:
  - `evaluation_mode = simplified_view_only_first_frame`
  - `full_echo_prime_pipeline = false`
- Wrote `evaluation_mode.txt` in outputs.

Affected file:
- `External_Val_Models/EchoPrime-main/run_external_validation.py`

Important:
- Current script is not full EchoPrime report-generation benchmarking.
- It is a simplified view-classifier benchmark and should be labeled as such.
