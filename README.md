# VoiceCodeBench Project

This repository contains a small Python project for exploring the VoiceCodeBench dataset from Hugging Face.

## Goal

The benchmark focuses on whether automatic speech recognition systems preserve exact structured tokens in English workplace speech, such as:

- URLs
- email addresses
- command-line flags
- filenames and paths
- account IDs
- dates
- numeric measurement values

## Project structure

- `src/voice_code_bench/dataset.py` – dataset loading and summary logic
- `src/voice_code_bench/__main__.py` – CLI entry point
- `tests/test_dataset_summary.py` – basic validation tests

## Install

```bash
python -m pip install -e .[dev]
```

## Run the analysis

```bash
python -m voice_code_bench --split test --limit 5
```

This downloads the dataset from Hugging Face and prints a compact summary of the first few samples.

## Score predictions

```bash
python -m voice_code_bench --split test --limit 10 --prediction-file predictions.json --json
```

The prediction file should be a JSON object mapping each `file_name` to the model’s transcript text. The package will compute CTEM and TSR-style metrics for the matching entities.

## Export a CSV summary

```bash
python -m voice_code_bench --split test --limit 20 --export-csv summary.csv
```

This writes a compact CSV file with file name, duration, domain, difficulty, entity count, and entity types for each sample.

## Notes

The benchmark is designed as a challenge for exact value recovery in ASR, rather than broad transcript quality alone.
