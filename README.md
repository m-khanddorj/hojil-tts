# hojil-tts

Khalkha Mongolian speech data preparation for future Qwen3-TTS work.

This repository is the home for the data preparation, training, evaluation, and inference work needed to build a Khalkha Mongolian text-to-speech model based on Qwen3-TTS.

## Project status

The preparation pipeline has processed Common Voice 27, FLEURS, and MBSpeech locally. The reviewed output contains 38,576 clips (50.53 hours). This change covers data preparation; training and model evaluation are separate work.

## Goals

- Prepare and validate Khalkha Mongolian speech and text data.
- Adapt Qwen3-TTS to speak natural Khalkha Mongolian.
- Document reproducible training and evaluation workflows.

## Data and model artifacts

Do not commit private, restricted, or unlicensed recordings, transcripts, model weights, or other large generated artifacts. Document their source, license, and local or remote storage separately before sharing them.

## License

No license has been selected yet. Until one is added, all rights are reserved by default.

## Data preparation

A local Python CLI prepares a provenance-preserving, multi-speaker corpus for
Khalkha language adaptation. Start with the [documentation index](docs/README.md):

- [Open-source data review](docs/1-open-source-data-review.md): sources, overlap, and license evidence.
- [Data preparation](docs/2-data-preparation.md): strategy, original inputs, measured results, and limitations.
- [Pipeline reference](docs/data-pipeline.md): configuration, commands, and artifact contracts.

The Python modules live directly in [data/](data/README.md). Audio, datasets,
weights, and generated reports are stored outside Git.

Noise reduction, speech-aware edge trimming, and offline transcript review are
available. See the [cleaning research and decisions](docs/data-cleaning-research.md)
and [real-data pilot](docs/data-pilot-2026-10-22.md). The example cleaning profile
requires explicitly provisioned local VAD/quality models and the `neural` extra.

Run `python -m data --help` or `python -m unittest discover -s tests -v`
from the repository root. Installing `pip install -e '.[audio,parquet,neural]'`
also provides the `hojil-data` command and optional processing dependencies.
No datasets or model weights are downloaded implicitly. A multilingual
language-adaptation trainer is still required; the official single-speaker Qwen
trainer is only an optional later integration.
