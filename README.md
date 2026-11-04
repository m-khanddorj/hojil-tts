# hojil-tts

Experimental Khalkha Mongolian TTS based on Qwen3-TTS, with a local CPU playground.

This repository is the home for the data preparation, training, evaluation, and inference work needed to build a Khalkha Mongolian text-to-speech model based on Qwen3-TTS.

## Project status

Full talker fine-tuning is complete: 28,001 retained training clips (32.70 hours), three epochs, final step 5253. The broader prepared corpus contains 38,576 clips (50.53 hours); it is not the training split. The inference export is available to authorized users at [khanddorj/hojil-tts on Hugging Face](https://huggingface.co/khanddorj/hojil-tts), which remains private.

The original development ASR evaluation measured mean CER 5.62% and WER 11.12% across conditioning modes. These are recognizer-based proxies, not listener accuracy. Later human review selected **Alternate 2 + full sentences**: 16/20 sentences correct, 14/20 natural. Two new seeds corrected all four remaining failures in eight selected retries. These small development experiments do not establish production readiness or a universally best seed. See [latest findings](docs/listening-findings-2026-11-04.md).

## Run the demo

The demo server and browser UI are included in this repository. It runs locally on CPU at `http://127.0.0.1:8766`; GitHub hosts the source, not an always-running inference service.

- Generate Mongolian speech, listen, download WAVs, and compare saved versions.
- Upload a 3–30 second reference recording with its exact transcript.
- Retry with a new seed while preserving the original voice and text.
- Save ratings and notes, export feedback, and stop the server from the page.

Follow the [complete setup and launch guide](docs/playground.md) for Python dependencies, the verified private model download, and reference setup. Weights require about 8.4 GB of disk space plus runtime memory; CPU generation is slow. No cloud GPU is started. Audio, credentials, model weights, and history stay outside Git.

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
No datasets or model weights are downloaded implicitly. The demo download is an
explicit separate command. Full training was run externally; this repository's
demo does not start or resume training.

## YouTube intake page

Install `uv pip install --python .venv/Scripts/python.exe -e '.[youtube]'`
(or `python -m pip install -e '.[youtube]'` in an activated environment).
Run `.venv/Scripts/python.exe -m data.youtube --output C:/external/hojil/youtube`
and open <http://127.0.0.1:8765>. Use an artifact folder outside every Git checkout.
See [YouTube intake](docs/youtube-intake.md) for review and export details.
