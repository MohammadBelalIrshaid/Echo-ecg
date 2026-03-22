# echo-eg Views Summary + PLAX Normalization Update

## 1) echo-eg view/class counts with model support

- `P10-MP4`: 1000 metadata rows in `ECHO_Outputs.xlsx`; 946 rows matched to existing local `.mp4` files; 54 missing files were excluded from counting.
- `mimic-iv-echo-ext`: 622 metadata rows in `MIMICEchoQA.json` and 622 unique video paths; 198 videos exist locally and were counted; 424 missing local files were excluded.
- Normalization used project conventions from `prepare_external_validation_datasets.py` (`canonical_view` logic), with `UNMAPPED` for labels that did not map.

| Sub-dataset | Raw view label | Normalized view label | Count | Used by EchoPrime? | Used by echo-view-classifier? |
|---|---|---|---|---|---|
| P10-MP4 | Apex-2CH | A2C | 67 | Yes | Yes |
| P10-MP4 | Apex-3CH | A3C | 49 | Yes | Yes |
| P10-MP4 | Apex-4CH | A4C | 173 | Yes | Yes |
| P10-MP4 | Apex-5CH | A5C | 47 | Yes | Yes |
| P10-MP4 | PLAX | PLAX | 176 | Yes | Yes |
| P10-MP4 | PSAX-Apex | PSAX_AP | 12 | Yes | Yes |
| P10-MP4 | PSAX-Aortic | PSAX_AV | 101 | Yes | Yes |
| P10-MP4 | PSAX-Valves | PSAX_AV | 31 | Yes | Yes |
| P10-MP4 | PSAX-Papillary | PSAX_MV | 57 | Yes | Yes |
| P10-MP4 | Aorta | SSN | 7 | Yes | No |
| P10-MP4 | Subcostal | SUBCOSTAL | 73 | Yes | No |
| P10-MP4 | IVC | SUBCOSTAL | 17 | Yes | No |
| P10-MP4 | ? | UNMAPPED | 135 | No | No |
| P10-MP4 | (empty) | UNMAPPED | 1 | No | No |
| mimic-iv-echo-ext | DOPPLER_A2C | A2C | 6 | Yes | Yes |
| mimic-iv-echo-ext | A2C | A2C | 5 | Yes | Yes |
| mimic-iv-echo-ext | A3C | A3C | 13 | Yes | Yes |
| mimic-iv-echo-ext | DOPPLER_A3C_AV | A3C | 7 | Yes | Yes |
| mimic-iv-echo-ext | A3C_LV | A3C | 2 | Yes | Yes |
| mimic-iv-echo-ext | DOPPLER_A3C | A3C | 1 | Yes | Yes |
| mimic-iv-echo-ext | DOPPLER_A3C_MV | A3C | 1 | Yes | Yes |
| mimic-iv-echo-ext | A4C | A4C | 23 | Yes | Yes |
| mimic-iv-echo-ext | DOPPLER_A4C_IVS | A4C | 8 | Yes | Yes |
| mimic-iv-echo-ext | DOPPLER_A4C_Pulvns | A4C | 6 | Yes | Yes |
| mimic-iv-echo-ext | DOPPLER_A4C_MV | A4C | 5 | Yes | Yes |
| mimic-iv-echo-ext | A4C_LV | A4C | 1 | Yes | Yes |
| mimic-iv-echo-ext | A4C_MV | A4C | 1 | Yes | Yes |
| mimic-iv-echo-ext | A4C_RV | A4C | 1 | Yes | Yes |
| mimic-iv-echo-ext | DOPPLER_A5C | A5C | 7 | Yes | Yes |
| mimic-iv-echo-ext | A5C | A5C | 5 | Yes | Yes |
| mimic-iv-echo-ext | PLAX_Zoom_out | PLAX | 20 | Yes | Yes |
| mimic-iv-echo-ext | PLAX | PLAX | 15 | Yes | Yes |
| mimic-iv-echo-ext | DOPPLER_PLAX_AV_MV | PLAX | 4 | Yes | Yes |
| mimic-iv-echo-ext | PLAX_AV_MV | PLAX | 3 | Yes | Yes |
| mimic-iv-echo-ext | DOPPLER_PLAX_MV_zoomed | PLAX | 1 | Yes | Yes |
| mimic-iv-echo-ext | DOPPLER_PLAX_RVIT | PLAX | 1 | Yes | Yes |
| mimic-iv-echo-ext | PLAX_RV_inflow | PLAX | 1 | Yes | Yes |
| mimic-iv-echo-ext | PLAX_zoomed_AV | PLAX | 1 | Yes | Yes |
| mimic-iv-echo-ext | PSAX_(level_of_apex) | PSAX_AP | 7 | Yes | Yes |
| mimic-iv-echo-ext | DOPPLER_PSAX_level_great_vessels_AV | PSAX_AV | 4 | Yes | Yes |
| mimic-iv-echo-ext | PSAX_(level_great_vessels)_focus_on_PV_and_PA | PSAX_AV | 3 | Yes | Yes |
| mimic-iv-echo-ext | DOPPLER_PSAX_level_great_vessels_TV | PSAX_AV | 2 | Yes | Yes |
| mimic-iv-echo-ext | PSAX_(level_great_vessels) | PSAX_AV | 2 | Yes | Yes |
| mimic-iv-echo-ext | PSAX_(level_great_vessels)_focus_on_TV | PSAX_AV | 2 | Yes | Yes |
| mimic-iv-echo-ext | DOPPLER_PSAX_level_great_vessels_PA | PSAX_AV | 1 | Yes | Yes |
| mimic-iv-echo-ext | DOPPLER_PSAX_MV | PSAX_MV | 7 | Yes | Yes |
| mimic-iv-echo-ext | PSAX_(level_of_papillary_muscles) | PSAX_MV | 6 | Yes | Yes |
| mimic-iv-echo-ext | PSAX_(level_of_MV) | PSAX_MV | 1 | Yes | Yes |
| mimic-iv-echo-ext | SSN_aortic_arch | SSN | 1 | Yes | No |
| mimic-iv-echo-ext | Subcostal_4C | SUBCOSTAL | 20 | Yes | No |
| mimic-iv-echo-ext | Subcostal_IVC | SUBCOSTAL | 2 | Yes | No |
| mimic-iv-echo-ext | Subcostal_Abdominal_Aorta | SUBCOSTAL | 1 | Yes | No |
| mimic-iv-echo-ext | DOPPLER_SC_4C_IAS | UNMAPPED | 1 | No | No |

## 2) Where model classes were found

- `EchoPrime` classes are defined in `External_Val_Models/EchoPrime-main/utils/utils.py` as `COARSE_VIEWS` (11 classes), and model head size `11` is confirmed in `External_Val_Models/EchoPrime-main/echo_prime/model.py`.
- `echo-view-classifier` classes are defined in `External_Val_Models/echo-view-classifier-master/classify.py` (`labels` dict, 8 classes), and mirrored in `External_Val_Models/echo-view-classifier-master/run_external_validation.py` as `CLASS_ORDER`. README also states 8 standard views.
- `EchoPrime` `COARSE_VIEWS`: `A2C`, `A3C`, `A4C`, `A5C`, `Apical_Doppler`, `Doppler_Parasternal_Long`, `Doppler_Parasternal_Short`, `Parasternal_Long`, `Parasternal_Short`, `SSN`, `Subcostal`.
- `echo-view-classifier` classes: `plax`, `psax-av`, `psax-mv`, `psax-ap`, `a4c`, `a5c`, `a3c`, `a2c`.

## 3) External validation code changes for PLAX normalization

- Added shared helpers in `External_Val_Models/validation_utils.py`:
  - `normalize_view_label_text(...)`
  - `is_plax_variant(...)`
  - `collapse_plax_label(...)`
- Updated `External_Val_Models/EchoPrime-main/run_external_validation.py` to:
  - normalize manifest labels before class filtering (`model_label_raw` + `_normalize_manifest_label`) so PLAX variants are accepted
  - collapse PLAX variants to canonical `PLAX` in label comparison/metrics/reporting via `collapse_label(...)`
- Updated `External_Val_Models/echo-view-classifier-master/run_external_validation.py` to:
  - normalize `model_label` with `collapse_plax_label(..., canonical="plax")` before class filtering
  - keep raw labels (`true_label_raw`, `pred_label_raw`) and compare/report using normalized labels (`true_label`, `pred_label`)

## 4) Modified files

- `External_Val_Models/validation_utils.py`
- `External_Val_Models/EchoPrime-main/run_external_validation.py`
- `External_Val_Models/echo-view-classifier-master/run_external_validation.py`

## 5) Brief validation check (PLAX variants => PLAX)

Command run:
```powershell
@'
from External_Val_Models.validation_utils import collapse_plax_label, is_plax_variant
samples = ['PLAX_VM','PLAX_mv','plax-something','DOPPLER_PLAX_AV_MV','Parasternal_Long','A4C']
for s in samples:
    print(f"{s}\tvariant={is_plax_variant(s)}\tcanonical={collapse_plax_label(s, canonical='PLAX')}")
'@ | python -
```

Observed output:
```text
PLAX_VM	variant=True	canonical=PLAX
PLAX_mv	variant=True	canonical=PLAX
plax-something	variant=True	canonical=PLAX
DOPPLER_PLAX_AV_MV	variant=True	canonical=PLAX
Parasternal_Long	variant=True	canonical=PLAX
A4C	variant=False	canonical=A4C
```

Additional syntax check command:
```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; python -m py_compile External_Val_Models/validation_utils.py External_Val_Models/EchoPrime-main/run_external_validation.py External_Val_Models/echo-view-classifier-master/run_external_validation.py
```
