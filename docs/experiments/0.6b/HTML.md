# Visual HTML edition

The report has a portable HTML edition with eight charts, training/evaluation
diagrams, a searchable experiment atlas, embedded written evidence, and eight
matched audio examples. It works offline; there are no CDN or API requests.

## Source structure

| Source | Responsibility |
|---|---|
| [visual-report.html](visual-report.html) | Responsive presentation, SVG charts, filtering, print styles |
| [visual-data.json](visual-data.json) | Panel-specific measurements, definitions, experiment families and source hashes |
| [visual-audio.json](visual-audio.json) | Existing matched clips, transcripts, individual scores and hashes; no audio bytes |
| [build_visual_report.py](../../../scripts/build_visual_report.py) | Verify sources, embed linked Markdown evidence, copy hash-verified audio into an external bundle |

The HTML template is source code. The generated report, audio, browser-check
receipts and screenshots belong outside the repository.

## Rebuild

Use a Python environment with `Markdown==3.11.1`. No model, GPU or network work
is required. From the repository root:

```powershell
python scripts/build_visual_report.py --output 'C:/path/outside-repository/0.6b-visual-report/index.html'
```

The builder rejects an output inside the repository and rejects changed chart
source hashes or changed audio hashes. Update the corresponding manifest from
verified evidence when the research advances; do not silently bypass the checks.
The artifact paths in these research manifests refer to the local evidence
archive and must remain available during rebuilding.

Open or share `index.html`: audio is embedded, so it works as a single file.
The builder also preserves verified WAV copies in `audio/`. Internal Markdown links are rewritten to embedded
evidence. Links to implementation files still require the original checkout;
literature links require internet access. Print uses the current panel and
expanded evidence sections.

## Validation of the November 11 snapshot

The browser checks covered chart initialization and panel switching, experiment
filtering, all internal anchors, playback of all eight WAVs, and desktop/mobile
layout. The artifact includes 42 embedded Markdown reports. Source data was
independently checked against 39 source hashes and saved score records. These
are report checks, not additional model-quality evaluations.
