# Khalkha Mongolian datasets for TTS

Research date: **2026-10-21**. Intended use: data planning for this repository's Khalkha Mongolian TTS project.

This research snapshot was moved from `data/README.md`. Its evidence limits refer
to the research date. For subsequent downloads, audio inspection, preparation,
and measured outcomes, see [Data preparation](2-data-preparation.md).

This catalog lists paired Mongolian speech and corresponding text usable for Khalkha TTS preparation: human recordings, derived ASR corpora, commercial offerings, and separately labeled synthetic speech. Original recordings and repackagings are distinguished to avoid double-counting. Commercial offerings still require delivery and dialect checks. Inclusion does not mean audio has been independently audited or is ready for training; cleaning, segmentation and alignment checks may be required. Babel qualifies only for its transcribed portion. Text-only resources, unpaired audio, isolated-word datasets, unverified research leads and other Mongolian varieties are excluded.

**Evidence limits:** Sizes, quality scores, and licenses below are publisher-reported unless explicitly described otherwise. Dataset cards and file listings were researched online; audio archives were not downloaded, listened to, or independently audited. A public download or license tag alone does not establish all underlying recording and text rights. “Unknown” means the reviewed source did not establish the fact.

## TTS-usable paired speech and text datasets

Duplicate distributions are grouped in the **Sources** column. Versions, filtered subsets, synthetic audio and merged corpora remain distinct rows because their contents differ; their hours overlap and must not be added together. **Dataset size** gives published storage size where available, followed by clip/row counts to clarify scope. Storage figures refer to the named archive or repository, not necessarily decoded audio. **Unknown** means not established in this research. Durations are total audio unless qualified; Babel is only partly transcribed. Synthetic speech is identified in the table and documented separately below. Existing **Use?** selections are preserved and are separate from pairing and license status.

| No  | Name                                                                                                            | Sources (list all if duplicates)                                                                                                                                                                                               | License                                                                                                 | Commercial use allowed?                                                    | Duration (in hours)                                                                        | Dataset size                                                                                               | Use?      |
| --- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- | --------- |
| 1 | Common Voice Mongolian v27 | [Mozilla Data Collective](https://mozilladatacollective.com/datasets/cmu5z0n5x00fdo10709hn7u8m) | CC0-1.0; additional access terms | Yes — subject to access terms | 130.38 total; 45.88 validated | 2.67 GB archive; 95,940 clips | Yes |
| 2 | Common Voice Mongolian v26 — cleaned | [btsee/common-voice-26-mn](https://huggingface.co/datasets/btsee/common-voice-26-mn) | CC0-1.0 | Yes — CC0 | 16.33 total; 15.74 train | About 2.83 GB repository; 15,092 clips | No |
| 3 | Common Voice Mongolian v24 — raw mirrors | [onlysainaa/common-voice-mn-24](https://huggingface.co/datasets/onlysainaa/common-voice-mn-24); [onlysainaa raw archive](https://huggingface.co/datasets/onlysainaa/common-voice-scripted-speech-24.0-mongolian/tree/main) | Upstream CC0-1.0; mirror declares Apache-2.0 (mismatch) | Yes under declared licenses — resolve mismatch | 140.56 total; 49.19 validated (mirror claims) | About 3.08 GB raw-archive repository; 96,308 clips claimed | No |
| 4 | Common Voice Mongolian v24 — cleaned | [btsee/common-voices-24-mn](https://huggingface.co/datasets/btsee/common-voices-24-mn) | CC0-1.0 claimed | Yes under claimed CC0 | About 40 claimed; inconsistent with current row count | 199 MB repository; 5,015 viewer rows versus about 28,900 clips claimed | No |
| 5 | Common Voice Mongolian v20 — normalized | [warmestman/common-voice-20-mn-normalized](https://huggingface.co/datasets/warmestman/common-voice-20-mn-normalized) | CC0-1.0 | Yes — CC0 | Unknown | Storage unknown; about 51.9k training rows | No |
| 6 | FLEURS Mongolian `mn_mn` — original | [google/fleurs](https://huggingface.co/datasets/google/fleurs) | CC-BY-4.0 | Yes — attribution required | Original subset total not independently established; ASR blend reports a 13.95 h selection | Mongolian subset storage/count unknown | Yes |
| 7 | FLEURS Mongolian — cleaned | [btsee/fleurs-mn](https://huggingface.co/datasets/btsee/fleurs-mn) | CC-BY-4.0 | Yes — attribution required | 4.30 total; 3.84 train | Storage unknown; 1,908 clips | No |
| 8 | MBSpeech / Mongolian Bible speech — original and repackagings | [Original downloader](https://github.com/tugstugi/mongolian-speech-recognition); [btsee/mbspeech\_mn](https://huggingface.co/datasets/btsee/mbspeech_mn); [bilguun/mbspeech](https://huggingface.co/datasets/bilguun/mbspeech) | MIT declared by btsee; bilguun license unestablished; underlying Bible text/recording rights unresolved | Unknown — underlying recording/text rights unresolved | 6.63 in bilguun release | About 721 MB download / 822 MB dataset reported by btsee; 3,846 examples | Yes |
| 9 | MBSpeech — cleaned | [btsee/mbspeech-mn](https://huggingface.co/datasets/btsee/mbspeech-mn) | MIT declared; underlying rights unresolved | Unknown — underlying rights unresolved | 5.40 total; 5.05 train | Storage unknown; 2,995 clips | No |
| 10 | WorldSpeech Mongolian `mn_mn` | [disco-eth/WorldSpeech](https://huggingface.co/datasets/disco-eth/WorldSpeech); [publisher repository](https://github.com/ETH-DISCO/worldspeech) | CC-BY-NC-4.0 plus source terms | No — non-commercial license | 181.0 | Mongolian subset storage/count unknown | Tentative |
| 11 | WorldSpeech Mongolian — cleaned | [btsee/WorldSpeech-mn](https://huggingface.co/datasets/btsee/WorldSpeech-mn) | Upstream CC-BY-NC-4.0 plus source terms | No — non-commercial license | 5.15 | Storage unknown; 4,228 clips | Tentative |
| 12 | IARPA Babel Halh Mongolian — LDC2020S10 | [LDC catalog and access](https://catalog.ldc.upenn.edu/LDC2020S10) | Specific LDC/IARPA Babel agreements; paid/membership access | Yes with LDC for-profit membership and corpus license; no under non-member license | About 204; about 77% transcribed | Storage/count unknown | Tentative |
| 13 | Mongolian STT dataset — merged ASR corpus | [Blgn94/mongolian-stt-dataset](https://huggingface.co/datasets/Blgn94/mongolian-stt-dataset) | CC-BY-4.0 declared; component terms apply | Conditional — declared CC-BY permits it; component rights need review | 146.9 | About 5.21 GB download; 97,688 rows (95,428 train + 2,260 validation) | No |
| 14 | Mongolian Voice — merged ASR corpus | [ssxenon01/mn-voice](https://huggingface.co/datasets/ssxenon01/mn-voice) | CC-BY-4.0 | Yes — attribution required | Unknown | Storage unknown; about 49.4k rows | No |
| 15 | CoVoST 2 Mongolian → English — Common Voice derivative | [Mozilla Data Collective](https://mozilladatacollective.com/datasets/cmp77ufsd02w9mp07q65muzr3); [publisher project](https://github.com/facebookresearch/covost) | CC0 declared | Yes under declared CC0 — access terms apply | 7.649 (converted from 7:38:57) | Storage unknown; 5,590 clips | Tentative |
| 16 | Mongolian Data — commercial Khalkha speech | [Provider](https://mongoliandata.com/) | Negotiated commercial license | Conditional — negotiated commercial agreement | 200+ advertised rolling production capacity; fixed delivered total unknown | Negotiated delivery; storage/count unknown | Tentative |
| 17 | DataoceanAI King-TTS-206 — commercial multi-speaker, multi-emotion TTS | [Provider dataset page](https://dataoceanai.com/datasets/tts/mongolian-multi-speaker-speech-synthesis-corpus-multi-emotion/) | Commercial agreement; Khalkha confirmation required | Conditional — commercial agreement required | 13.63 | Storage unknown; 6,921 sentences, four speakers | Tentative |
| 18 | Twine Mongolian (Mongolia) General Conversation — commercial | [Provider dataset page](https://www.twine.net/dataset/mongolian-mongolia-general-conversation-data) | Contact provider; Khalkha confirmation required | Unknown — obtain provider terms | 2 claimed; page fields conflict | Storage unknown; two files / two participants reported | Tentative |
| 19 | Synthetic MBSpeech — Edge TTS | [Dorjzodovsuren/synthetic\_mbspeech\_dataset\_edgetts](https://huggingface.co/datasets/Dorjzodovsuren/synthetic_mbspeech_dataset_edgetts) | Unknown; generator terms unresolved | Unknown — dataset/generator permissions unresolved | Unknown | Storage unknown; about 3.85k rows | No |
| 20 | Synthetic MBSpeech — ElevenLabs | [Dorjzodovsuren/synthetic\_mbspeech\_dataset\_eleven](https://huggingface.co/datasets/Dorjzodovsuren/synthetic_mbspeech_dataset_eleven) | Unknown; generator terms unresolved | Unknown — dataset/generator permissions unresolved | Unknown | Storage unknown; about 3.81k rows | No |

**Commercial use allowed?** summarizes the documented license position: **Yes** means the stated license permits commercial use subject to its obligations; **No** indicates an explicit non-commercial restriction; **Conditional** requires the stated contract or rights checks; **Unknown / Not established** means permission has not been verified. These labels do not independently certify underlying recording, voice or text rights.

Detailed provenance, suitability, access limitations and duplication notes follow.

## 1. Paired speech: original sources and preferred derivatives

### 1.1 Mozilla Common Voice Scripted Speech — Mongolian (`mn`)

- **Type / dialect:** Crowdsourced read speech with transcripts. Mongolia/Khalkha is the target variety, but locale alone does not certify every speaker's accent.
- **Current verified release:** 27.0, corpus date 2026-09-11; **95,940 clips, 130.38 recorded hours, 45.88 validated hours, 616 speakers, 6,114 sentences**. MP3 audio with metadata.
- **Access / terms:** Official download via Mozilla Data Collective. License listed as **CC0-1.0**. The same page forbids speaker identification and rehosting/resharing; preserve and review the access terms separately from the license tag.
- **TTS use:** Select consistent speakers and clean recordings. Validated ASR clips still need checks for misreads, clipping, noise, and sentence boundaries. Total hours are not TTS-ready hours.
- **Source:** [Official Mongolian v27 datasheet and download](https://mozilladatacollective.com/datasets/cmu5z0n5x00fdo10709hn7u8m).

#### Common Voice releases and repackagings

These share recordings with Common Voice and must not be counted as independent speech collections.

| Dataset / access | Reported contents | License / suitability |
| --- | --- | --- |
| [btsee/common-voice-26-mn](https://huggingface.co/datasets/btsee/common-voice-26-mn) | **15,092 clips, 16.33 h, 378 identified speakers**; 24 kHz mono WAV in Parquet; explicitly prepared for Khalkha TTS | CC0-1.0. Strong initial candidate. Has quality metrics and original client IDs. Published median bandwidth is about 7.57 kHz, despite 24 kHz storage. |
| [btsee/common-voices-24-mn](https://huggingface.co/datasets/btsee/common-voices-24-mn) | Older TTS preparation with denoising and 22.05 kHz output | CC0-1.0 claimed. **Metadata conflict:** card says roughly 28,900 clips / 40 h, but indexed viewer lists 5,015 rows. Do not budget 40 h without measuring the current files. Direct page retrieval also failed during research; indexed card and file listing were available. |
| [onlysainaa/common-voice-mn-24](https://huggingface.co/datasets/onlysainaa/common-voice-mn-24) and [raw v24 archive](https://huggingface.co/datasets/onlysainaa/common-voice-scripted-speech-24.0-mongolian/tree/main) | Mirror card reports 96,308 clips, 140.56 h total / 49.19 h validated, 606 speakers; MP3 archive | Mirror declares Apache-2.0, while Common Voice upstream is CC0. Preserve upstream provenance and resolve the mismatch. v24 figures are mirror claims and should not be substituted for official v27 statistics. |
| [warmestman/common-voice-20-mn-normalized](https://huggingface.co/datasets/warmestman/common-voice-20-mn-normalized) | Normalized v20 ASR packaging; viewer shows about 51.9k training rows and client IDs | CC0-1.0. Useful for reproducing older ASR pipelines; recheck normalization and extreme duration/text outliers before TTS. |

**Important split detail:** The cleaned v26 repository exposes all rows through one Hugging Face `train` file set. Its row-level `split` field distinguishes train, validation, test, and withheld. The card reports **15.74 h actual training audio**. Filter that column; loading the outer `train` split alone includes evaluation rows. [Split documentation](https://huggingface.co/datasets/btsee/common-voice-26-mn/blob/main/README.md).

### 1.2 Google FLEURS — Mongolian (`mn_mn`)

- **Type / dialect:** Read speech paired with Mongolian text; use the `mn_mn` configuration. A useful Mongolia-language candidate, with individual accents still to be reviewed.
- **Format / metadata:** 16 kHz audio; normalized and raw transcriptions, gender and language fields. Original train/dev/test structure.
- **License / access:** **CC-BY-4.0**, public Hugging Face distribution.
- **TTS use:** Useful supplementary pronunciation and sentence coverage. Prefer raw, punctuated transcripts where they match the recording. Preserve benchmark partitions if using FLEURS for evaluation.
- **Source:** [Google FLEURS dataset and documentation](https://huggingface.co/datasets/google/fleurs).

**TTS-oriented derivative:** [btsee/fleurs-mn](https://huggingface.co/datasets/btsee/fleurs-mn) reports **1,908 clips / 4.30 h**, including **3.84 h training** and 0.46 h withheld; 24 kHz mono WAV in Parquet, CC-BY-4.0. It explicitly targets Khalkha TTS but has **zero identified speakers**, meaning speaker identities are unavailable, not that recordings have no speakers. Its withheld partition is a text holdout. This is resampled 16 kHz source audio, so it provides no new high-frequency information. The broader ASR blend below reports 13.95 h from its FLEURS selection; that is a different selection and not the cleaned TTS total. [Cleaned dataset card](https://huggingface.co/datasets/btsee/fleurs-mn).

### 1.3 MBSpeech / Mongolian Bible speech

- **Type / dialect:** Aligned Mongolian Bible read speech, described by the cleaned release as Khalkha. A single male narrator provides useful voice consistency.
- **Original tooling:** [tugstugi/mongolian-speech-recognition](https://github.com/tugstugi/mongolian-speech-recognition) documents the MBSpeech downloader and single-speaker training.
- **TTS use:** Good small single-voice baseline after transcript and rights review. Expect Biblical vocabulary and narration style rather than conversational coverage.
- **Rights limitation:** MIT tags on redistributions are publisher declarations. Trace permission for both the specific Bible translation and original performance/recording; a software repository license does not by itself establish those rights.

| Distribution | Contents | Notes |
| --- | --- | --- |
| [btsee/mbspeech_mn](https://huggingface.co/datasets/btsee/mbspeech_mn) | **3,846 examples**, 16 kHz audio and sentences | Older packaging; declares MIT. |
| [bilguun/mbspeech](https://huggingface.co/datasets/bilguun/mbspeech) | **3,846 examples / 6.63 h**, original and normalized sentences | No explicit license was established from the reviewed card. Same source family, not additional hours. |
| [btsee/mbspeech-mn](https://huggingface.co/datasets/btsee/mbspeech-mn) | **2,995 clips / 5.40 h**, one narrator; 24 kHz WAV in Parquet | Cleaned Khalkha TTS selection; MIT declared; **5.05 h train**, 0.35 h withheld. Card states native upstream rate 22.05 kHz and measured median bandwidth about 7.56 kHz. |

**Naming trap:** `mbspeech_mn` and `mbspeech-mn` are distinct repositories. The cleaned card's example also refers to `mbspeech-mn-clean`, inconsistent with the page's repository ID. Inspect the actual repository files and pin the revision before scripting ingestion. [Cleaned card](https://huggingface.co/datasets/btsee/mbspeech-mn).

### 1.4 WorldSpeech — Mongolian (`mn_mn`)

- **Contents / dialect:** **181.0 h** in the Mongolia/Mongolian configuration. Its source list includes Mongolian parliamentary proceedings and Latter-day Saints addresses. It is a likely Khalkha source pool, not a per-speaker dialect certification.
- **Access / license:** Public [WorldSpeech dataset](https://huggingface.co/datasets/disco-eth/WorldSpeech), **CC-BY-NC-4.0**, with additional source-specific terms. The repository's source-license descriptions should not be interpreted as independent clearance to commercially reuse the packaged collection.
- **TTS use:** Potentially useful for research on broader styles and acoustic adaptation. Segment, diarize, check transcript alignment, and reject unsuitable channels/backgrounds.
- **Derivative:** [btsee/WorldSpeech-mn](https://huggingface.co/datasets/btsee/WorldSpeech-mn) reports **4,228 clips / 5.15 h**, 24 kHz WAV, and no identified speakers. It is an overlapping cleaned selection, not an extra collection. Indexed card was available, but direct retrieval failed during research. Its reported median bandwidth is about 7.66 kHz; inspect files rather than assuming full-band fidelity from the sample rate.
- **Additional primary source:** [Publisher repository](https://github.com/ETH-DISCO/worldspeech).

### 1.5 IARPA Babel Mongolian — LDC2020S10

- **Identity:** IARPA-babel401b-v2.0b; explicitly **Halh Mongolian (`khk`)** spoken by native speakers in Mongolia.
- **Contents:** Approximately **204 h** of conversational and scripted telephone speech collected in 2014. Transcripts cover approximately **77%** of speech, in Cyrillic and romanization. Do not treat all 204 h as paired data.
- **Audio:** Catalog lists 8 kHz, 8-bit A-law SPHERE and 48 kHz, 24-bit PCM WAV. Inspect the documentation to establish which recordings have each format; 48 kHz containers do not guarantee studio bandwidth.
- **Access / commercial terms:** Commercial technology development is permitted through the [LDC for-profit membership agreement](https://catalog.ldc.upenn.edu/license/ldc-for-profit-membership.pdf) and the [Mongolian corpus for-profit agreement](https://catalog.ldc.upenn.edu/license/iarpa-babel-mongolian-agreement-for-profit.pdf). The corpus agreement requires attribution and prohibits redistributing the dataset. The [non-member agreement](https://catalog.ldc.upenn.edu/license/iarpa-babel-mongolian-agreement-non-member.pdf) permits only non-commercial education, research and technology development. Public distribution of trained model weights is not explicitly addressed; confirm that use with LDC.
- **TTS use:** Lower priority for a clean voice, useful for conversational pronunciation/prosody or supporting ASR. Review channel quality and permitted model uses first.
- **Source:** [Official LDC catalog, documentation, and license links](https://catalog.ldc.upenn.edu/LDC2020S10).

## 2. Merged corpora and task-specific derivatives

| Dataset | Contents and access | TTS assessment / overlap |
| --- | --- | --- |
| [Blgn94/mongolian-stt-dataset](https://huggingface.co/datasets/Blgn94/mongolian-stt-dataset) | Card reports **146.9 h**, 95,428 train / 2,260 validation rows; Common Voice v24 + FLEURS + MBSpeech; 16 kHz decoding; CC-BY-4.0 declared | Explicit Cyrillic Khalkha ASR blend. Includes unvalidated Common Voice `other` clips. Validation is not speaker-disjoint and 91% of validation sentences occur in train. Its 2026-07-30 revision removed YouTube audio; pin revision. Rebuild TTS splits and recover speaker metadata from upstream. |
| [ssxenon01/mn-voice](https://huggingface.co/datasets/ssxenon01/mn-voice) | Common Voice + FLEURS, normalized to 16 kHz WAV; viewer about 49.4k rows; CC-BY-4.0 | Useful provenance fields, including source/version, speaker ID, sentence ID and waveform hash. ASR normalization still needs review for spoken-form TTS. No independent new speech hours. |
| [CoVoST 2, Mongolian → English](https://mozilladatacollective.com/datasets/cmp77ufsd02w9mp07q65muzr3) | **5,590 Mongolian clips / 7:38:57**, derived from Common Voice 4.0; [official project](https://github.com/facebookresearch/covost) declares CC0 | Use Mongolian source transcripts, not English translations, as TTS targets. Useful as a reproducible legacy subset; overlapping Common Voice recordings. English → Mongolian is English audio with Mongolian translation, so it is not Mongolian speech training data. |

## 3. Commercial sources

These are acquisition leads, not downloaded or independently validated datasets. Confirm delivery quantities, dialect, text/audio alignment, speaker permissions, and TTS/model-distribution rights in the actual agreement.

| Provider / dataset | Publisher specifications | Khalkha confidence and usefulness |
| --- | --- | --- |
| [Mongolian Data](https://mongoliandata.com/) | Explicit standard **Khalkha**, Cyrillic; 48 kHz, 16-bit mono WAV; transcripts, speaker IDs, demographics and consent records advertised; **200+ h rolling production capacity**, pilot delivery quoted at 2–4 weeks | Strong dialect match on paper. The 200+ h is advertised production capacity, not an independently verified fixed downloadable corpus. Controlled read speech could be valuable for TTS; inspect samples and negotiate terms. |
| [DataoceanAI Mongolian Multi-speaker Speech Synthesis Corpus, King-TTS-206](https://dataoceanai.com/datasets/tts/mongolian-multi-speaker-speech-synthesis-corpus-multi-emotion/) | **13.63 h, 6,921 sentences, four professional speakers** (2 male / 2 female); 48 kHz, 24-bit studio audio; multiple emotions and phonetic/prosodic labels | Purpose-built TTS candidate. The `mn-mn_xsampa` phone-set label suggests Mongolia orientation, but the reviewed page does not explicitly certify Khalkha. Request native-speaker assessment and contract confirmation. Commercial quote/license. |
| [Twine: Mongolian (Mongolia) General Conversation data](https://www.twine.net/dataset/mongolian-mongolia-general-conversation-data) | Description says **2 h**, two speakers, timestamped transcription and speaker segments; WAV; commercial contact | Mongolia-specific, Khalkha not explicitly certified. Page duration fields conflict with the prose, so verify delivered duration. Phone-app conversational recording: supporting style data rather than a first clean-voice corpus. |

## 4. Synthetic paired speech

These datasets contain generated speech paired with text rather than new human recordings. Their inclusion does not establish permission for model distillation; keep them separate from human speech and evaluation sets.

| Resource | Evidence / access | Possible use and unresolved issue |
| --- | --- | --- |
| [Synthetic MBSpeech via Edge TTS](https://huggingface.co/datasets/Dorjzodovsuren/synthetic_mbspeech_dataset_edgetts) | Viewer about **3.85k rows**, paired audio/text | Synthetic rather than new human speech. No explicit dataset license established from reviewed page. Audit generator terms, pronunciation and voice provenance before any distillation. |
| [Synthetic MBSpeech via ElevenLabs](https://huggingface.co/datasets/Dorjzodovsuren/synthetic_mbspeech_dataset_eleven) | Viewer about **3.81k rows**, paired audio/text | Same underlying prompt domain; generator/model-training permissions and dataset license unresolved. Keep out of human-speech evaluation sets. |

## 5. Recommended acquisition and preparation order

1. **Start with the sources marked Yes in the table:** original Common Voice v27, original FLEURS, and MBSpeech after recording-rights review. Pin revisions, restrict Babel to its transcribed portion if acquired, and measure usable paired hours after filtering. Preserve evaluation partitions; do not add derivative hours to their original source totals.
2. **Compare with upstream audio.** Denoising and early downsampling can reduce naturalness or permanently remove high-frequency detail. A larger sample rate on a derivative cannot restore it. Keep original audio and process to the eventual model's required rate only once.
3. **Build provenance before merging.** Track source family, release/revision, original clip ID, speaker/session ID where available, original and spoken-form text, duration, native sample rate, license/access terms and audio hash. Hash decoded audio too, since different encodings can hide duplicates.
4. **Separate evaluation by text and speaker/session.** A text holdout is not a speaker holdout. Group Bible passages, repeated Common Voice sentences and parliamentary sessions appropriately. Check repeated and translated benchmark sentences for overlap across retained sources. Keep synthetic speech out of human evaluation.
5. **Have native Khalkha listeners review samples.** Assess dialect, pronunciation, transcript fidelity, speaker consistency, artifacts and prosody. Automated ASR/CER and quality scores can assist selection but cannot certify correctness.
6. **Expand deliberately.** Acquire additional paired studio recordings from a verified commercial provider. Use WorldSpeech only in a separately tracked research workflow compatible with its terms. Do not inflate the budget by adding mirrors and mixtures to their own upstream recordings.

## 6. Research coverage and remaining gaps

Searches covered English terms such as “Khalkha / Halh Mongolian TTS dataset,” “Mongolian speech corpus,” dataset cards on Hugging Face, publisher GitHub repositories, Mozilla Data Collective, LDC, academic papers, and commercial speech catalogs. Sources above are linked at each entry so the catalog remains usable without this chat.

The main unresolved gaps are original Bible recording permissions; per-speaker dialect verification in broadly tagged `mn` datasets; actual delivered commercial inventories; synthetic-data generator permissions; and independently measured usable paired hours. No claim is made that proprietary Chimege/Bolorsoft or university training corpora are openly available simply because a product or paper exists.

Only this catalog is stored here. Downloaded audio, private/restricted material, and large artifacts should be stored separately with source and license records, consistent with the repository's data policy.
