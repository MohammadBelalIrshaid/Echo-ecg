# Trouble Shooting

### 1. CardiacNet preprocessing output collisions

The saved CardiacNet evaluation results are not reliable because multiple distinct source files were mapped to the same processed output filename before inference.

Cause:
- In `prepare_external_validation_datasets.py`, the CardiacNet samples use `sample_id = f.stem`.
- CardiacNet contains many repeated stems such as `PHI01` and `patient-1-4_image`.
- The processed outputs for `echo-view-classifier`, `EchocardiographyQC`, and `EchoPrime` are then named from that `sample_id`, so different source files collide onto the same processed asset.

Impact:
- `echo-view-classifier` and `EchoPrime` reused processed files across multiple CardiacNet rows.
- `EchocardiographyQC` is affected more severely because its inference matching also depends on the duplicated UID/folder name.
- CardiacNet F1 and macro F1 from the saved runs should not be treated as trustworthy.

Why this cannot be fixed post-run:
- The collision happened before inference.
- Once different sources were written to the same processed path, the source-specific prepared inputs were lost or overwritten.
- A post-hoc metric recomputation would still be based on corrupted prepared assets, not the intended per-source evaluation.

Required fix:
1. Change CardiacNet preprocessing to generate unique sample/output names per source file.
2. Regenerate the CardiacNet processed inputs.
3. Rerun CardiacNet evaluation for `echo-view-classifier`, `EchocardiographyQC`, and `EchoPrime`.

### 2. Dataset-level macro F1 is computed over the full model label set

The saved dataset-level `macro_f1` values are averaged over every class in the model label space, including labels that are absent in that dataset.

Impact:
- On subset datasets such as `CardiacNet` and `CAMUS_public`, the saved macro F1 is much lower than the average over supported labels only.
- Example: for `CardiacNet`, the saved macro F1 is not equal to `A4C` F1 because all zero-support classes are included in the macro average.
- This makes dataset-level macro F1 easy to misinterpret.

Required fix:
1. Keep the current metric only if you want strict full-label-space macro averaging.
2. Also report a second metric: macro F1 over labels with `support > 0`.
3. Clearly label which macro definition is being used in tables and summaries.

### 3. EchocardiographyQC results are incomplete

The saved `EchocardiographyQC` outputs only cover a small partial run and do not represent the full set of compatible evaluation samples.

Impact:
- The saved QC results include only `CardiacNet` and only 31 matched series.
- There are no saved QC runs for `CAMUS_public`, `echo-eg_MIMICEchoQA`, or `echo-eg_P10`, even though large compatible subsets exist.
- Any comparison that treats the current QC results as full benchmark coverage is misleading.

Required fix:
1. Rebuild the QC-ready processed inputs after fixing naming issues.
2. Run the official QC inference workflow on all intended compatible datasets.
3. Save the resulting `inferResults`-based outputs so the benchmark is complete and reproducible.

### 4. EchocardiographyQC preprocessing does not match the official pipeline

The current dataset preparation for `EchocardiographyQC` does not reproduce the official preprocessing described in that model's README.

Impact:
- The official tool expects preprocessing that crops the central region and masks the image so only the ultrasound sector remains.
- The current prep script writes generic grayscale JPG sequences and metadata, but does not apply the same crop-and-mask procedure.
- This can materially lower view-classification performance and make the QC results unfairly pessimistic.

Required fix:
1. Use the official `dicom-preprocess-1.0.0.exe` pipeline where possible.
2. If that is not possible, implement the equivalent crop-and-mask logic before QC inference.
3. Treat existing QC metrics as provisional until preprocessing is aligned.

### 5. EchoPrime is not being evaluated through its intended full pipeline

The current `EchoPrime` validation script does not run the full intended EchoPrime inference workflow.

Impact:
- The script loads the view-classifier weights and prepares a video tensor, but the actual prediction step uses only the first sampled frame.
- That means the saved results are effectively from a single-frame view classifier, not from the intended multi-video EchoPrime setup.
- Reported `EchoPrime` F1 and macro F1 may therefore underestimate or misrepresent the model's intended performance.

Required fix:
1. Evaluate EchoPrime using the official notebook or an equivalent full-pipeline implementation.
2. If a reduced view-only evaluation is desired, label it explicitly as a simplified view-classifier benchmark.
3. Do not present the current saved outputs as full EchoPrime benchmark results without that caveat.

### 6. Cross-model metric comparison is not directly apples-to-apples

The three models do not use the same label space, so their per-label and macro metrics are not directly comparable without harmonization.

Impact:
- `echo-view-classifier`, `EchocardiographyQC`, and `EchoPrime` use different class sets.
- `EchoPrime` includes labels such as `Apical_Doppler`, `Subcostal`, and `Doppler_Parasternal_Long`, while `echo-view-classifier` does not.
- `EchoPrime` also collapses PSAX variants differently from the other models.
- As a result, raw macro F1 values across models are partly driven by taxonomy differences, not just model quality.

Required fix:
1. Define a harmonized comparison label set before comparing models head-to-head.
2. Report model-specific metrics separately from harmonized cross-model metrics.
3. Add explicit caveats anywhere cross-model macro F1 is shown in one table.
