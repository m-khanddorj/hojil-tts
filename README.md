# hojil-tts

Training Qwen3-TTS for Khalkha Mongolian.

This repository is the home for the data preparation, training, evaluation, and inference work needed to build a Khalkha Mongolian text-to-speech model based on Qwen3-TTS.

## Project status

Project setup is just beginning. Training instructions, dataset documentation, and model evaluation details will be added as the pipeline takes shape.

## Goals

- Prepare and validate Khalkha Mongolian speech and text data.
- Adapt Qwen3-TTS to speak natural Khalkha Mongolian.
- Document reproducible training and evaluation workflows.

## Data and model artifacts

Do not commit private, restricted, or unlicensed recordings, transcripts, model weights, or other large generated artifacts. Document their source, license, and local or remote storage separately before sharing them.

## License

No license has been selected yet. Until one is added, all rights are reserved by default.

## Data preparation

A local Python CLI now prepares a provenance-preserving, multi-speaker corpus for
Khalkha language adaptation. See [the pipeline guide](docs/data-pipeline.md) and
[example configuration](configs/example.json) for acquisition, validation,
reproducible splits, reports, and optional Qwen single-speaker export.

Run `python3 -m hojil_data --help` or `python3 -m unittest discover -s tests -v`.
No datasets or model weights are downloaded implicitly. A multilingual
language-adaptation trainer is still required; the official single-speaker Qwen
trainer is only an optional later integration.
