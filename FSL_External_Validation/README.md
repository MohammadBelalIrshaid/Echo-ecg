# FSL External Validation

## Model

Downloaded from `https://huggingface.co/unsloth/Qwen3.5-9B-GGUF`.
Files included:
- `Qwen3.5-9B-Q4_K_M.gguf`
- `mmproj-F16.gguf`
- `README.md` from the Hugging Face repo

Assumption used: `Q4_K_M` plus `mmproj-F16` for a practical local multimodal GGUF setup.

## Trial Dataset

Support examples are sourced from `echo-eg_P10` in `External_Val_Models/EchoPrime-main/processed_datasets/manifest.csv`.
Ranking is based on column E in `Raw_Datasets/echo-eg/ECHO_Outputs.xlsx` using:
- `Good`
- `Intermediate`
- `Bad`
- `Very Bad`
- `Terrible`

For each label, the best available examples were selected in that order.
Labels included: A2C, A3C, A4C, A5C, Apical_Doppler, Doppler_Parasternal_Long, Doppler_Parasternal_Short, Parasternal_Long, Parasternal_Short, SSN, Subcostal.
`SSN` has only 7 available P10 examples, so the 10-shot frame trial is capped at 7 for that label.

Trials created:
- `frames_1_per_label`
- `frames_5_per_label`
- `frames_10_per_label`
- `full_video_1_per_label`

Each trial folder includes a `manifest.csv`.
A cross-trial availability summary is written to `selection_summary.csv`.

## llama.cpp Evaluation Script

`run_qwen35_llamacpp_fsl_validation.py` runs few-shot external validation with llama.cpp's `llama-server`.

Assumptions:
- the `Qwen3.5-9B-Q8_0.gguf` model is used for evaluation
- `mmproj-F16.gguf` is supplied for multimodal support
- support examples come from one of the trial manifests in `trials/`
- the default external query set is `echo-eg_MIMICEchoQA` from `External_Val_Models/EchoPrime-main/processed_datasets/manifest.csv`
- videos are rendered to cached JPEG assets for llama.cpp, using a contact sheet by default

Example command:

```powershell
python FSL_External_Validation\run_qwen35_llamacpp_fsl_validation.py `
  --llama-server C:\path\to\llama-server.exe `
  --support-trial frames_5_per_label `
  --query-dataset echo-eg_MIMICEchoQA
```
