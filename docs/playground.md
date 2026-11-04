# Local CPU playground

Generate and compare full-sentence Mongolian speech in a local browser. The server binds only to loopback; it is a single-user app. GitHub hosts its source; the [private Hugging Face repository](https://huggingface.co/khanddorj/hojil-tts) hosts weights and requires access. No cloud GPU or external inference API is used.

## Install

From the repository root, using Python 3.12 (tested on Windows):

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -r requirements-playground.txt
.venv/Scripts/python.exe -m pip install -e .
```

On Linux/macOS use `.venv/bin/python`. The application uses CPU FP32 and loads the model on first generation. The export occupies about 8.4 GB; the downloader also requires 3 GiB free disk headroom. RAM usage is substantial: the tested PC had 29.7 GiB total RAM and about 8.5–9 GiB free before loading. This is an observation, not a minimum memory guarantee.

## Download once

Put a read-capable `HF_TOKEN` for an account with model access in the ignored repository `.env` file. Never commit it. Choose an external directory outside every Git checkout:

```powershell
.venv/Scripts/python.exe scripts/download_playground_model.py --receipt configs/playground-model.json --env-file .env --output C:/external/hojil/model
```

The manifest pins the original verified export revision and size/SHA256 checks. The later model-card revision changes documentation only. Downloading creates `LOCAL_VERIFIED.json` after every listed file verifies. Inference is then offline.

## Prepare the default voice

Use a clean 3–30 second recording you have permission to use. Save its exact spoken words in a UTF-8 text file. Keep both outside Git:

```powershell
.venv/Scripts/python.exe scripts/prepare_playground_reference.py --audio C:/external/hojil/voice.wav --transcript C:/external/hojil/voice.txt --label "My reference" --output C:/external/hojil/inference-defaults.json
```

This validates audio and saves a mono WAV and profile with the tested sampling settings. Alternate 2 is the experiment's reference label, not a built-in voice. That recording is not distributed in GitHub or Hugging Face. If you already have the reviewed profile and matching recording, use them directly. Another reference may produce different quality.

## Launch

```powershell
.venv/Scripts/python.exe -m data.playground --model C:/external/hojil/model --profile C:/external/hojil/inference-defaults.json --output C:/external/hojil/playground/versions
```

Open <http://127.0.0.1:8766>. For a hidden background process on Windows:

```powershell
./scripts/start_playground.ps1 -Python "$PWD/.venv/Scripts/python.exe" -Model C:/external/hojil/model -Profile C:/external/hojil/inference-defaults.json -Output C:/external/hojil/playground/versions
```

Enter one sentence (up to 350 characters), then generate. A blank seed draws randomly. **Try another version** preserves the clip's text and reference while drawing a new seed. Saved versions include audio, seed, reference hash/transcript, generation settings and feedback. Rate clips and export feedback JSON. Only one generation runs at a time; within-sentence chunking is disabled.

**Upload a different voice** accepts WAV, FLAC, OGG and MP3 when supported by SoundFile, up to 15 MB and 3–30 seconds. Enter its exact transcript and a name, then save and select it. Uploaded voices persist after restarting.

## Stop and reopen

Click **Close app & release memory** when idle. Saved files remain on disk. Rerun the launch command to reopen. Interrupted generations are marked failed on restart. Windows launcher logs are next to the versions directory. Do not expose this local server through port forwarding.

## Verification and limits

Real CPU generation, uploaded-reference generation, playback and feedback export were exercised. A short greeting took 77 seconds including loading; a warm retry took about 20 seconds. These observations do not predict longer-sentence latency. CPU output may differ from GPU output even with matching seeds. See [listening findings](listening-findings-2026-11-04.md).

Run `python -m unittest discover -s tests -p test_playground.py -v` for API/reference tests. They use temporary files and no inference; they do not establish pronunciation quality. Audio tests require NumPy and SoundFile.

