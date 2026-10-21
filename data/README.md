# Khalkha Mongolian datasets for TTS

Research date: **2026-10-21**. Intended use: data planning for this repository's Khalkha Mongolian TTS project.

This catalog covers publicly discoverable speech datasets, useful derivatives, commercial offerings, and supporting text resources. It separates original recordings from repackagings and distinguishes confirmed Khalkha from language labels that still require dialect checks. It is a broad research inventory, not a guarantee that every private or unindexed corpus has been found.

**Evidence limits:** Sizes, quality scores, and licenses below are publisher-reported unless explicitly described otherwise. Dataset cards and file listings were researched online; audio archives were not downloaded, listened to, or independently audited. A public download or license tag alone does not establish all underlying recording and text rights. “Unknown” means the reviewed source did not establish the fact.

## Comprehensive dataset table

Duplicate distributions are grouped in the **Sources** column. Versions, filtered subsets, synthetic audio and merged corpora remain distinct rows because their contents differ; their hours overlap and must not be added together. **Dataset size** gives published storage size where available, followed by clip/row or text counts to clarify scope. Storage figures refer to the named archive or repository, not necessarily decoded audio. **Unknown** means not established in this research; **N/A** means text-only. Durations are total audio unless qualified. Commercial, unpaired, synthetic and dialect-unverified entries are labeled explicitly; inclusion is not confirmation that they are ready for Khalkha TTS.

| No | Name | Sources (list all if duplicates) | License | Commercial use allowed? | Duration (in hours) | Dataset size |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Common Voice Mongolian v27 | [Mozilla Data Collective](https://mozilladatacollective.com/datasets/cmu5z0n5x00fdo10709hn7u8m) | CC0-1.0; additional access terms | Yes — subject to access terms | 130.38 total; 45.88 validated | 2.67 GB archive; 95,940 clips |
| 2 | Common Voice Mongolian v26 — cleaned | [btsee/common-voice-26-mn](https://huggingface.co/datasets/btsee/common-voice-26-mn) | CC0-1.0 | Yes — CC0 | 16.33 total; 15.74 train | About 2.83 GB repository; 15,092 clips |
| 3 | Common Voice Mongolian v24 — raw mirrors | [onlysainaa/common-voice-mn-24](https://huggingface.co/datasets/onlysainaa/common-voice-mn-24); [onlysainaa raw archive](https://huggingface.co/datasets/onlysainaa/common-voice-scripted-speech-24.0-mongolian/tree/main) | Upstream CC0-1.0; mirror declares Apache-2.0 (mismatch) | Yes under declared licenses — resolve mismatch | 140.56 total; 49.19 validated (mirror claims) | About 3.08 GB raw-archive repository; 96,308 clips claimed |
| 4 | Common Voice Mongolian v24 — cleaned | [btsee/common-voices-24-mn](https://huggingface.co/datasets/btsee/common-voices-24-mn) | CC0-1.0 claimed | Yes under claimed CC0 | About 40 claimed; inconsistent with current row count | 199 MB repository; 5,015 viewer rows versus about 28,900 clips claimed |
| 5 | Common Voice Mongolian v20 — normalized | [warmestman/common-voice-20-mn-normalized](https://huggingface.co/datasets/warmestman/common-voice-20-mn-normalized) | CC0-1.0 | Yes — CC0 | Unknown | Storage unknown; about 51.9k training rows |
| 6 | FLEURS Mongolian `mn_mn` — original | [google/fleurs](https://huggingface.co/datasets/google/fleurs) | CC-BY-4.0 | Yes — attribution required | Original subset total not independently established; ASR blend reports a 13.95 h selection | Mongolian subset storage/count unknown |
| 7 | FLEURS Mongolian — cleaned | [btsee/fleurs-mn](https://huggingface.co/datasets/btsee/fleurs-mn) | CC-BY-4.0 | Yes — attribution required | 4.30 total; 3.84 train | Storage unknown; 1,908 clips |
| 8 | MBSpeech / Mongolian Bible speech — original and repackagings | [Original downloader](https://github.com/tugstugi/mongolian-speech-recognition); [btsee/mbspeech_mn](https://huggingface.co/datasets/btsee/mbspeech_mn); [bilguun/mbspeech](https://huggingface.co/datasets/bilguun/mbspeech) | MIT declared by btsee; bilguun license unestablished; underlying Bible text/recording rights unresolved | Unknown — underlying recording/text rights unresolved | 6.63 in bilguun release | About 721 MB download / 822 MB dataset reported by btsee; 3,846 examples |
| 9 | MBSpeech — cleaned | [btsee/mbspeech-mn](https://huggingface.co/datasets/btsee/mbspeech-mn) | MIT declared; underlying rights unresolved | Unknown — underlying rights unresolved | 5.40 total; 5.05 train | Storage unknown; 2,995 clips |
| 10 | WorldSpeech Mongolian `mn_mn` | [disco-eth/WorldSpeech](https://huggingface.co/datasets/disco-eth/WorldSpeech); [publisher repository](https://github.com/ETH-DISCO/worldspeech) | CC-BY-NC-4.0 plus source terms | No — non-commercial license | 181.0 | Mongolian subset storage/count unknown |
| 11 | WorldSpeech Mongolian — cleaned | [btsee/WorldSpeech-mn](https://huggingface.co/datasets/btsee/WorldSpeech-mn) | Upstream CC-BY-NC-4.0 plus source terms | No — non-commercial license | 5.15 | Storage unknown; 4,228 clips |
| 12 | IARPA Babel Halh Mongolian — LDC2020S10 | [LDC catalog and access](https://catalog.ldc.upenn.edu/LDC2020S10) | Specific LDC/IARPA Babel agreements; paid/membership access | Conditional — applicable Babel agreement must permit it | About 204; about 77% transcribed | Storage/count unknown |
| 13 | Mongolian STT dataset — merged ASR corpus | [Blgn94/mongolian-stt-dataset](https://huggingface.co/datasets/Blgn94/mongolian-stt-dataset) | CC-BY-4.0 declared; component terms apply | Conditional — declared CC-BY permits it; component rights need review | 146.9 | About 5.21 GB download; 97,688 rows (95,428 train + 2,260 validation) |
| 14 | Mongolian Voice — merged ASR corpus | [ssxenon01/mn-voice](https://huggingface.co/datasets/ssxenon01/mn-voice) | CC-BY-4.0 | Yes — attribution required | Unknown | Storage unknown; about 49.4k rows |
| 15 | CoVoST 2 Mongolian → English — Common Voice derivative | [Mozilla Data Collective](https://mozilladatacollective.com/datasets/cmp77ufsd02w9mp07q65muzr3); [publisher project](https://github.com/facebookresearch/covost) | CC0 declared | Yes under declared CC0 — access terms apply | 7.649 (converted from 7:38:57) | Storage unknown; 5,590 clips |
| 16 | Multilingual Spoken Words — Mongolian word clips | [MLCommons](https://mlcommons.org/datasets/multilingual-spoken-words/) | Verify release-specific and upstream Common Voice terms | Unknown — verify release terms | Unknown | Mongolian subset storage/count not established |
| 17 | Mongolian Data — commercial Khalkha speech | [Provider](https://mongoliandata.com/) | Negotiated commercial license | Conditional — negotiated commercial agreement | 200+ advertised rolling production capacity; fixed delivered total unknown | Negotiated delivery; storage/count unknown |
| 18 | DataoceanAI King-TTS-206 — commercial multi-speaker, multi-emotion TTS | [Provider dataset page](https://dataoceanai.com/datasets/tts/mongolian-multi-speaker-speech-synthesis-corpus-multi-emotion/) | Commercial agreement; Khalkha confirmation required | Conditional — commercial agreement required | 13.63 | Storage unknown; 6,921 sentences, four speakers |
| 19 | Twine Mongolian (Mongolia) General Conversation — commercial | [Provider dataset page](https://www.twine.net/dataset/mongolian-mongolia-general-conversation-data) | Contact provider; Khalkha confirmation required | Unknown — obtain provider terms | 2 claimed; page fields conflict | Storage unknown; two files / two participants reported |
| 20 | VoxLingua107 Mongolian — unpaired speech | [TalTechNLP/voxlingua107_wds](https://huggingface.co/datasets/TalTechNLP/voxlingua107_wds) | CC-BY-4.0 declared; source-video rights review needed | Conditional — declared CC-BY permits it; source rights need review | About 71 | Mongolian subset storage/count unknown |
| 21 | MMS unlabeled v2 / DiscoPhon — unpaired Khalkha lead | [espnet/mms_ulab_v2](https://huggingface.co/datasets/espnet/mms_ulab_v2); [segmented coml/mmsulab](https://huggingface.co/datasets/coml/mmsulab) | CC-BY-NC-SA-4.0 | No — non-commercial license | Khalkha total unknown | Khalkha subset storage/count unknown; segmentation changes row counts |
| 22 | Synthetic MBSpeech — Edge TTS | [Dorjzodovsuren/synthetic_mbspeech_dataset_edgetts](https://huggingface.co/datasets/Dorjzodovsuren/synthetic_mbspeech_dataset_edgetts) | Unknown; generator terms unresolved | Unknown — dataset/generator permissions unresolved | Unknown | Storage unknown; about 3.85k rows |
| 23 | Synthetic MBSpeech — ElevenLabs | [Dorjzodovsuren/synthetic_mbspeech_dataset_eleven](https://huggingface.co/datasets/Dorjzodovsuren/synthetic_mbspeech_dataset_eleven) | Unknown; generator terms unresolved | Unknown — dataset/generator permissions unresolved | Unknown | Storage unknown; about 3.81k rows |
| 24 | Mongolian news speech variants — unverified leads | [Collection listing `mn_news_700m_0_3000`, `_v2`, `_v3_long`](https://huggingface.co/collections/Dorjzodovsuren/mongolian-speech-datasets) | Unknown | Unknown | Unknown | Storage unknown; collection lists about 3k / 5k / 2.08k rows respectively |
| 25 | Government accessibility TTS Corpus C17 — institutional lead | [Procurement notice](https://www.pwdp.mn/?p=913) | Open/freeware release intended; final data license unverified | Unknown — final data license unverified | Unknown | Completed release and size unverified |
| 26 | 2021 study's Mongolian Bible target corpus — research lead | [Authors' paper](https://tokushima-u.repo.nii.ac.jp/record/2009777/files/eurasip_2021_42.pdf) | Unknown; request from authors | Unknown — author permission needed | 12; experiment also uses a 0.5 h subset | Storage unknown; 307 utterances in the 0.5 h subset; possible MBSpeech overlap |
| 27 | FineWeb2 Halh Mongolian `khk_Cyrl` — text | [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | ODC-By database license; underlying content rights apply | Conditional — database license permits it; underlying content rights apply | N/A | Card lists 8.52 GB / 2.58 GB compressed; 1,622,882 documents / 824,211,882 tokens |
| 28 | CulturaX Mongolian `mn` — text | [uonlp/CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) | Inherits mC4/OSCAR terms; gated access | Unknown — component/content terms need review | N/A | Storage unknown; 1,928,828 documents / 1,850,667,656 tokens |
| 29 | CC-100 Mongolian — text | [Official download index](https://data.statmt.org/cc-100/) | No IP claim on preparation; underlying content rights apply | Unknown — underlying content permissions not established | N/A | Official index labels download 397M; document/token count unknown |
| 30 | OSCAR Mongolian — text | [oscar-corpus/oscar](https://huggingface.co/datasets/oscar-corpus/oscar) | CC0 metadata tag; content/access conditions apply; access suspended | Unknown — content terms unresolved; access suspended | N/A | Release-dependent; current accessible size unknown |
| 31 | 700-million-word Mongolian news corpus — text | [Resource index and download link](https://github.com/tugstugi/mongolian-nlp#datasets); [BERT training-data reference](https://huggingface.co/tugstugi/bert-large-mongolian-uncased) | Corpus-specific license unknown | Unknown | N/A | Storage unknown; about 700 million words reported |
| 32 | Eduge Mongolian news — text | [tugstugi/eduge](https://huggingface.co/datasets/tugstugi/eduge); [resource index](https://github.com/tugstugi/mongolian-nlp#datasets) | Unknown | Unknown | N/A | Storage unknown; about 75k news items, nine topics |
| 33 | FLORES-200 / FLORES+ — evaluation text | [facebook/flores](https://huggingface.co/datasets/facebook/flores); [FLORES+ successor](https://huggingface.co/datasets/openlanguagedata/flores_plus) | Legacy CC-BY-SA-4.0; verify successor terms separately | Yes for legacy CC-BY-SA — attribution/share-alike; successor unverified | N/A | Legacy public splits: 997 dev + 1,012 devtest sentences per configuration; storage/successor count unknown |
| 34 | Universal Dependencies Mongolian — annotated text | [Mongolian UD resources](https://universaldependencies.org/mn/index.html) | Treebank-specific; verify selected release | Unknown — treebank-specific | N/A | Treebank-dependent; storage/count not established |
| 35 | UniMorph Mongolian — morphology | [unimorph/mon](https://github.com/unimorph/mon) | Verify repository/release license | Unknown — verify release license | N/A | Storage/count not established |
| 36 | Mongolian names, places, abbreviations and NER — text resource group | [Index linking the individual resources](https://github.com/tugstugi/mongolian-nlp#datasets) | Resource-specific; verify each file | Unknown — resource-specific | N/A | Multiple distinct datasets; aggregate size not established |
| 37 | MnTTS — Khalkha unverified | [Paper](https://arxiv.org/abs/2209.10848); [publisher repository](https://github.com/walker-hyf/MnTTS) | Data-specific license unverified | Unknown | About 8 | Storage/clip count unknown; one speaker |
| 38 | MnTTS2 — Khalkha unverified | [Paper](https://arxiv.org/abs/2301.00657); [publisher repository](https://github.com/ssmlkl/MnTTS2); [authors' comparative study](https://www.mdpi.com/2076-3417/13/7/4237) | Data-specific license unverified | Unknown | 30 | Storage/clip count unknown; three speakers |
| 39 | NCMMSC2022-MTTSC — MnTTS subset, Khalkha unverified | [Authors' dataset account](https://www.mdpi.com/2076-3417/13/7/4237) | Challenge/data terms unverified | Unknown | 2 | Storage/clip count unknown; overlaps MnTTS |
| 40 | M2ASR-MONGO / Cyrillic transliteration — Chakhar | [Transfer-learning study](https://www.mdpi.com/2079-9292/14/6/1137) | Verify original dataset and transliteration terms | Unknown | 162.6 in described filtered version (147 train + 9.5 validation + 6.1 test) | Storage/clip count unknown |
| 41 | MnASR — Khalkha unverified | [Publisher paper](https://doi.org/10.1109/O-COCOSDA202257103.2022.9997919) | Free for research per abstract; full terms unverified | Not established — research use only documented | 345 | Storage/clip count unknown |
| 42 | IMUT-MC and other Inner Mongolian synthesis / Mongolian–Chinese translation collections — unverified lead group | No individual release verified in this review; see section 6 | Unknown | Unknown | Unknown | Unknown; not a single established dataset |

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
- **Access / terms:** LDC account and applicable paid/membership access; specific Babel agreements, not an open permissive license.
- **TTS use:** Lower priority for a clean voice, useful for conversational pronunciation/prosody or supporting ASR. Review channel quality and permitted model uses first.
- **Source:** [Official LDC catalog, documentation, and license links](https://catalog.ldc.upenn.edu/LDC2020S10).

## 2. Merged corpora and task-specific derivatives

| Dataset | Contents and access | TTS assessment / overlap |
| --- | --- | --- |
| [Blgn94/mongolian-stt-dataset](https://huggingface.co/datasets/Blgn94/mongolian-stt-dataset) | Card reports **146.9 h**, 95,428 train / 2,260 validation rows; Common Voice v24 + FLEURS + MBSpeech; 16 kHz decoding; CC-BY-4.0 declared | Explicit Cyrillic Khalkha ASR blend. Includes unvalidated Common Voice `other` clips. Validation is not speaker-disjoint and 91% of validation sentences occur in train. Its 2026-07-30 revision removed YouTube audio; pin revision. Rebuild TTS splits and recover speaker metadata from upstream. |
| [ssxenon01/mn-voice](https://huggingface.co/datasets/ssxenon01/mn-voice) | Common Voice + FLEURS, normalized to 16 kHz WAV; viewer about 49.4k rows; CC-BY-4.0 | Useful provenance fields, including source/version, speaker ID, sentence ID and waveform hash. ASR normalization still needs review for spoken-form TTS. No independent new speech hours. |
| [CoVoST 2, Mongolian → English](https://mozilladatacollective.com/datasets/cmp77ufsd02w9mp07q65muzr3) | **5,590 Mongolian clips / 7:38:57**, derived from Common Voice 4.0; [official project](https://github.com/facebookresearch/covost) declares CC0 | Use Mongolian source transcripts, not English translations, as TTS targets. Useful as a reproducible legacy subset; overlapping Common Voice recordings. English → Mongolian is English audio with Mongolian translation, so it is not Mongolian speech training data. |
| [Multilingual Spoken Words — Mongolian](https://mlcommons.org/datasets/multilingual-spoken-words/) | Word clips extracted from Common Voice; multilingual keyword dataset | Useful for isolated-word pronunciation/segmentation diagnostics, weak for sentence-level prosody. Preserve upstream provenance and license terms; not additional independent speech. |

## 3. Commercial sources

These are acquisition leads, not downloaded or independently validated datasets. Confirm delivery quantities, dialect, text/audio alignment, speaker permissions, and TTS/model-distribution rights in the actual agreement.

| Provider / dataset | Publisher specifications | Khalkha confidence and usefulness |
| --- | --- | --- |
| [Mongolian Data](https://mongoliandata.com/) | Explicit standard **Khalkha**, Cyrillic; 48 kHz, 16-bit mono WAV; transcripts, speaker IDs, demographics and consent records advertised; **200+ h rolling production capacity**, pilot delivery quoted at 2–4 weeks | Strong dialect match on paper. The 200+ h is advertised production capacity, not an independently verified fixed downloadable corpus. Controlled read speech could be valuable for TTS; inspect samples and negotiate terms. |
| [DataoceanAI Mongolian Multi-speaker Speech Synthesis Corpus, King-TTS-206](https://dataoceanai.com/datasets/tts/mongolian-multi-speaker-speech-synthesis-corpus-multi-emotion/) | **13.63 h, 6,921 sentences, four professional speakers** (2 male / 2 female); 48 kHz, 24-bit studio audio; multiple emotions and phonetic/prosodic labels | Purpose-built TTS candidate. The `mn-mn_xsampa` phone-set label suggests Mongolia orientation, but the reviewed page does not explicitly certify Khalkha. Request native-speaker assessment and contract confirmation. Commercial quote/license. |
| [Twine: Mongolian (Mongolia) General Conversation data](https://www.twine.net/dataset/mongolian-mongolia-general-conversation-data) | Description says **2 h**, two speakers, timestamped transcription and speaker segments; WAV; commercial contact | Mongolia-specific, Khalkha not explicitly certified. Page duration fields conflict with the prose, so verify delivered duration. Phone-app conversational recording: supporting style data rather than a first clean-voice corpus. |

## 4. Unpaired speech, synthetic speech, and research leads

| Resource | Evidence / access | Possible use and unresolved issue |
| --- | --- | --- |
| [VoxLingua107 — Mongolian `mn`](https://huggingface.co/datasets/TalTechNLP/voxlingua107_wds) | Publisher lists approximately **71 h** for Mongolian; YouTube-derived language-ID audio; CC-BY-4.0 declared | No sentence transcripts supplied for paired TTS. Could support language identification or acoustic pretraining after dialect/content checks. Language labels do not certify Khalkha, and source-video rights need review. |
| [MMS unlabeled v2](https://huggingface.co/datasets/espnet/mms_ulab_v2) / [DiscoPhon segmented derivative](https://huggingface.co/datasets/coml/mmsulab) | Segmented derivative's published language metadata includes `khk`; **CC-BY-NC-SA-4.0**; religious/Global Recordings Network provenance | Research-only unpaired acoustic material. Khalkha hours and recording quality were not established here. Not a verified paired MMS-TTS audio release. |
| [Synthetic MBSpeech via Edge TTS](https://huggingface.co/datasets/Dorjzodovsuren/synthetic_mbspeech_dataset_edgetts) | Viewer about **3.85k rows**, paired audio/text | Synthetic rather than new human speech. No explicit dataset license established from reviewed page. Audit generator terms, pronunciation and voice provenance before any distillation. |
| [Synthetic MBSpeech via ElevenLabs](https://huggingface.co/datasets/Dorjzodovsuren/synthetic_mbspeech_dataset_eleven) | Viewer about **3.81k rows**, paired audio/text | Same underlying prompt domain; generator/model-training permissions and dataset license unresolved. Keep out of human-speech evaluation sets. |
| [Dorjzodovsuren Mongolian speech collection](https://huggingface.co/collections/Dorjzodovsuren/mongolian-speech-datasets) | Also lists `mn_news_700m_0_3000`, `_v2`, and `_v3_long` | Additional speech-related leads. The collection alone does not establish whether these news variants contain human or synthetic audio, their duration, or rights. Inspect individual cards/files before classifying or counting them. |
| [Mongolian TTS Corpus, government accessibility project C17](https://www.pwdp.mn/?p=913) | 2021 procurement notice explicitly requests Cyrillic-to-Khalkha TTS and multi-speaker corpora; says open/freeware release was intended | Relevant institutional lead. A completed downloadable corpus, hours, and final data license were not verified. An announcement is not evidence of a released dataset. |
| [2021 low-resource Mongolian TTS study](https://tokushima-u.repo.nii.ac.jp/record/2009777/files/eurasip_2021_42.pdf) | Describes a **12 h** target Bible speech corpus and a 30-minute / 307-utterance experiment | Contact authors for access, dialect and rights. Relationship to MBSpeech is unverified; do not sum their hours. |

MMS model weights, commercial TTS APIs, and demos are not themselves downloadable human training corpora. This research did not establish a separately licensed public Khalkha MMS-TTS training-audio package or a Khalkha CMU Wilderness subset.

## 5. Text datasets for prompts and text processing

Text alone cannot teach the acoustic model to speak. These resources can supply recording prompts, vocabulary coverage, punctuation/normalization cases, or text-side adaptation. Filter language and script, remove boilerplate, and preserve source rights. Avoid translating/transliterating a different Mongolian variety and assuming that this establishes Khalkha pronunciation.

| Resource | Verified contents / language selection | TTS use / terms |
| --- | --- | --- |
| [FineWeb2 `khk_Cyrl`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | Explicit **Halh Mongolian in Cyrillic**; card's table lists **824,211,882 tokens / 1,622,882 documents** | Strong large prompt-mining candidate. ODC-By database license; underlying web-content rights still apply. Pin the release because the corpus evolves. |
| [CulturaX `mn`](https://huggingface.co/datasets/uonlp/CulturaX) | Card lists **1,928,828 documents / 1,850,667,656 tokens**; cleaned multilingual web corpus | Broad vocabulary and sentence mining; verify Cyrillic Khalkha at row level. Gated access; license follows mC4/OSCAR components rather than a blanket new content permission. |
| [CC-100 Mongolian](https://data.statmt.org/cc-100/) | Official index lists a Mongolian download of **397M**; this is the index's file-size label, not a word count | Public web-text source for coverage and prompt selection. Publisher makes no IP claim on corpus preparation; underlying content rights remain relevant. |
| [OSCAR Mongolian](https://huggingface.co/datasets/oscar-corpus/oscar) | Mongolian slice in a multilingual Common Crawl corpus | Useful historical text source, but **access is currently suspended** according to the card. CC0 metadata tag does not remove the card's content/legal conditions. |
| [700-million-word Mongolian news corpus](https://github.com/tugstugi/mongolian-nlp#datasets) | Resource index links the news corpus; also documented as BERT training material in [the model card](https://huggingface.co/tugstugi/bert-large-mongolian-uncased) | News prompts and vocabulary. Download availability and corpus-specific license need confirmation; repository/code licensing is insufficient. |
| [Eduge news dataset](https://huggingface.co/datasets/tugstugi/eduge) | Approximately **75k Mongolian news items**, nine topics | Balanced topic sampling for recording scripts. Card explicitly lists license **unknown**; obtain permission before adopting as redistributable training text. |
| [FLORES-200](https://huggingface.co/datasets/facebook/flores) / [FLORES+ successor](https://huggingface.co/datasets/openlanguagedata/flores_plus) | Mongolian Cyrillic parallel evaluation text; legacy FLORES uses `khk_Cyrl` | Best reserved for held-out text evaluation, especially if training on FLEURS. Legacy FLORES card lists CC-BY-SA-4.0 and gated access; inspect successor terms independently. |
| [Mongolian Universal Dependencies resources](https://universaldependencies.org/mn/index.html) | Language documentation explicitly identifies Khalkha; small annotated grammatical text | Morphology, segmentation and sentence-structure checks. Check the chosen treebank's script/transliteration and license; do not assume it is ready-to-read Cyrillic. |
| [Mongolian UniMorph](https://github.com/unimorph/mon) | Inflectional word forms and morphological annotations | Suffix/word-form coverage for pronunciation tests and prompt design. Check repository license and convert representations carefully. No audio. |
| [Names, abbreviations, places and NER resources](https://github.com/tugstugi/mongolian-nlp#datasets) | Index links personal/company/clan names, Mongolian place names, country names, abbreviations and annotated news | Useful for difficult names, abbreviation expansion and coverage tests. These are separate resources with potentially different permissions; confirm each file's provenance. No audio. |

The large web corpora overlap substantially. Their token counts are not additive measures of unique Mongolian text. Metadata tagged `mn` can also contain mixed languages or scripts; `khk_Cyrl` is more explicit but still warrants quality sampling.

## 6. Frequently confused resources: dialect unverified or different variety

These may help cross-variety research, but should not be presented as verified Khalkha training hours.

| Dataset | Reported size | Reason for separate treatment |
| --- | --- | --- |
| [MnTTS](https://arxiv.org/abs/2209.10848) / [publisher code](https://github.com/walker-hyf/MnTTS) | About **8 h**, one professional female announcer | Inner Mongolia University project with traditional-script/Latin text processing. Khalkha dialect is not established by the reviewed release. Request dialect confirmation and data-specific license; code availability does not settle either. |
| [MnTTS2](https://arxiv.org/abs/2301.00657) / [publisher code](https://github.com/ssmlkl/MnTTS2) | **30 h**, three professional speakers | Inner Mongolia-oriented corpus; no explicit Khalkha certification. The [authors' comparative study](https://www.mdpi.com/2076-3417/13/7/4237) explains its relation to MnTTS and the traditional-script/Latin workflow. |
| NCMMSC2022-MTTSC challenge subset | **2 h** in the authors' description | A MnTTS subset, not an independent Khalkha corpus. [Authors' dataset account](https://www.mdpi.com/2076-3417/13/7/4237). |
| M2ASR-MONGO and Cyrillic transliteration | A published filtered version uses 147 h train / 9.5 h validation / 6.1 h test | The [Khalkha transfer-learning study](https://www.mdpi.com/2079-9292/14/6/1137) explicitly identifies the source as **Chakhar**, and Common Voice as Khalkha. Converting script does not convert dialect. |
| [MnASR](https://doi.org/10.1109/O-COCOSDA202257103.2022.9997919) | **345 h** | Research-purpose Mongolian ASR corpus; reviewed abstract does not certify Khalkha or unrestricted commercial reuse. Confirm dialect and data access with authors. |
| IMUT-MC, older Inner Mongolian synthesis corpora, Mongolian–Chinese speech translation collections | Varies | Relevant search leads, but not established Khalkha data in this review. Require primary documentation of speakers, script, access and license before inclusion in a Khalkha training manifest. |

Kalmyk, Buryat and other Mongolic-language datasets should likewise remain separate from Khalkha; their possible use is multilingual transfer, not direct evidence of target-language coverage.

## 7. Recommended acquisition and preparation order

1. **Pin and inspect the three practical open candidates:** cleaned Common Voice 26, cleaned FLEURS, and MBSpeech after recording-rights review. Their published totals sum to **26.03 h across all partitions**, while published training partitions sum to **24.63 h**. This is an arithmetic planning estimate, not an independently measured or globally deduplicated usable total. [CV26](https://huggingface.co/datasets/btsee/common-voice-26-mn), [FLEURS](https://huggingface.co/datasets/btsee/fleurs-mn), [MBSpeech](https://huggingface.co/datasets/btsee/mbspeech-mn).
2. **Compare with upstream audio.** Denoising and early downsampling can reduce naturalness or permanently remove high-frequency detail. A larger sample rate on a derivative cannot restore it. Keep original audio and process to the eventual model's required rate only once.
3. **Build provenance before merging.** Track source family, release/revision, original clip ID, speaker/session ID where available, original and spoken-form text, duration, native sample rate, license/access terms and audio hash. Hash decoded audio too, since different encodings can hide duplicates.
4. **Separate evaluation by text and speaker/session.** A text holdout is not a speaker holdout. Group Bible passages, repeated Common Voice sentences and parliamentary sessions appropriately. Check FLEURS/FLORES-derived text for overlap. Keep synthetic speech out of human evaluation.
5. **Have native Khalkha listeners review samples.** Assess dialect, pronunciation, transcript fidelity, speaker consistency, artifacts and prosody. Automated ASR/CER and quality scores can assist selection but cannot certify correctness.
6. **Expand deliberately.** Mine FineWeb2 for new recording prompts, or acquire studio data from a verified commercial provider. Use WorldSpeech only in a separately tracked research workflow compatible with its terms. Do not inflate the budget by adding mirrors and mixtures to their own upstream recordings.

## 8. Research coverage and remaining gaps

Searches covered English terms such as “Khalkha / Halh Mongolian TTS dataset,” “Mongolian speech corpus,” dataset cards on Hugging Face, publisher GitHub repositories, Mozilla Data Collective, LDC, academic papers, and commercial speech catalogs. Sources above are linked at each entry so the catalog remains usable without this chat.

The main unresolved gaps are original Bible recording permissions; finalized availability of institutional corpora; per-speaker dialect verification in broadly tagged `mn` datasets; actual delivered commercial inventories; and independently measured usable hours. No claim is made that proprietary Chimege/Bolorsoft or university training corpora are openly available simply because a product or paper exists.

Only this catalog is stored here. Downloaded audio, private/restricted material, and large artifacts should be stored separately with source and license records, consistent with the repository's data policy.
