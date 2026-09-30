# General Image Forgery Datasets

This document provides a curated list of datasets for **general image forgery and tampering detection**. The collection focuses on image-level manipulations including copy-move, splicing, object insertion/removal, image compositing, inpainting, and related forensic artifacts.

The datasets include both traditional and synthetic manipulations. Some datasets contain only pristine/source imagery and are included because they are used to construct or support image-forgery datasets.

> **Terminology:** `Real` refers to pristine/original images, while `Tampered` refers to manipulated or forged images.

---

## Dataset Summary

Counts below are what is actually in **processed-v1** — the dataset built from `s3://authenta-general-image-forgery-dataset/dataset-v1/` and stored as MosaicML Streaming shards at `s3://authenta-data-rnd/image-tampering-detection/processed-v1/{train,test}` (see [MDS_DATASET.md](MDS_DATASET.md)). Size is the dataset's image + mask bytes, stored unchanged in the shards.

| Dataset | Split | Real | Tampered | Total | Size | Forgery Type |
|---|---|---:|---:|---:|---:|---|
| [CASIA v1](#casia-v1) | train | 800 | 919 | 1,719 | 0.07 GB | Copy-move, splicing |
| [CASIA v2](#casia-v2) | train | 7,437 | 5,105 | 12,542 | 3.47 GB | Copy-move, splicing |
| [COCO 2017](#coco-2017) (train2017) | train | 282,360 | 0 | 282,360 | 46.05 GB | Source imagery |
| [FantasticReality](#fantasticreality) | train | 16,592 | 19,423 | 36,015 | 5.94 GB | Splicing |
| [LAION-Mobile](#laion-mobile) | train | 671,346 | 0 | 671,346 | 955.58 GB | Pristine/natural images; source imagery |
| [MISD](#misd) | train | 619 | 296 | 915 | 0.06 GB | Multiple-image splicing |
| [compRAISE](#compraise) | train | 24,462 | 0 | 24,462 | 65.15 GB | Pristine/source images; compression-related forensic artifacts |
| [tampCOCO](#tampcoco) | train | 0 | 798,078 | 798,078 | 52.91 GB | Copy-move, splicing |
| [COCO 2017](#coco-2017) (val2017) | test | 5,000 | 0 | 5,000 | 0.82 GB | Source imagery |
| [DEFACTO](#defacto) | test | 0 | 18,194 | 18,194 | 10.50 GB | Splicing, copy-move, removal/inpainting |
| [IMD2020](#imd2020) | test | 414 | 1,968 | 2,382 | 0.60 GB | Splicing, copy-move, removal |
| [In-the-Wild](#in-the-wild) | test | 0 | 201 | 201 | 0.09 GB | Real-world splicing |
| [Realistic Tampering (Korus)](#realistic-tampering-korus) | test | 220 | 220 | 440 | 1.61 GB | Object insertion, object removal |
| [CocoGlide](#cocoglide) | test | 512 | 512 | 1,024 | 0.12 GB | AI-based object replacement / inpainting |
| [Columbia](#columbia) | test | 183 | 180 | 363 | 0.81 GB | Splicing |

Upstream release sizes differ from these counts for several datasets (e.g. DEFACTO, LAION-Mobile, COCO 2017) — the per-dataset sections below describe the full upstream releases; the tables here describe the subset in dataset-v1 / processed-v1.

---

## Train

`processed-v1/train` — 2,108 shards × 512 MB, 1,128.9 GB.

| Dataset | Real | Tampered | Total | Size | Skipped |
|---|---:|---:|---:|---:|---|
| CASIA v1 | 800 | 919 | 1,719 | 0.07 GB | 1 mask-size-mismatch |
| CASIA v2 | 7,437 | 5,105 | 12,542 | 3.47 GB | 17 mask-size-mismatch |
| COCO 2017 (train2017) | 282,360 | 0 | 282,360 | 46.05 GB | — |
| FantasticReality | 16,592 | 19,423 | 36,015 | 5.94 GB | — |
| LAION-Mobile | 671,346 | 0 | 671,346 | 955.58 GB | 950 truncated image |
| MISD | 619 | 296 | 915 | 0.06 GB | — |
| compRAISE | 24,462 | 0 | 24,462 | 65.15 GB | — |
| tampCOCO | 0 | 798,078 | 798,078 | 52.91 GB | 236 empty mask |
| **TOTAL** | **1,003,616** | **823,821** | **1,827,437** | **1,129.2 GB** | **1,204** |

---

## Test

`processed-v1/test` — 28 shards × 512 MB, 14.6 GB.

| Dataset | Real | Tampered | Total | Size |
|---|---:|---:|---:|---:|
| COCO 2017 (val2017) | 5,000 | 0 | 5,000 | 0.82 GB |
| DEFACTO | 0 | 18,194 | 18,194 | 10.50 GB |
| IMD2020 | 414 | 1,968 | 2,382 | 0.60 GB |
| In-the-Wild | 0 | 201 | 201 | 0.09 GB |
| Realistic Tampering (Korus) | 220 | 220 | 440 | 1.61 GB |
| CocoGlide | 512 | 512 | 1,024 | 0.12 GB |
| Columbia | 183 | 180 | 363 | 0.81 GB |
| **TOTAL** | **6,329** | **21,275** | **27,604** | **14.55 GB** |

---

## Grand Total (Train + Test)

| | Real | Tampered | Total | Size |
|---|---:|---:|---:|---:|
| **Combined** | **1,009,945** | **845,096** | **1,855,041** | **1,143.4 GB** |

Skipped files (not in processed-v1; listed with reasons in `processed-v1/train_failed.csv`): 1,204, all in train. Columbia masks are derived from `<stem>_edgemask.jpg` — bright red (camera 1, near the splicing boundary) = tampered — by `packages/mdsconverter/prepare_columbia.py`; compRAISE is extracted from `compRAISE_full.zip` by `prepare_compraise.py`.

---

## Forgery-Type Taxonomy

The datasets in this catalog cover the following major manipulation categories:

- **Copy-move** content is copied from one region of an image and pasted elsewhere in the same image.
- **Splicing** content from one or more images is composited into another image.
- **Object insertion** a new object or region is inserted into an image.
- **Object removal** an existing object or region is removed.
- **Image compositing** multiple image sources are combined into a single image.
- **Inpainting** missing or unwanted image regions are reconstructed or replaced.
- **Retouching / enhancement** local or global modifications intended to alter image appearance.
- **Compression / re-JPEG artifacts** compression-related traces useful for forensic analysis.

---

# Dataset Details

## CASIA v2

**Dataset type:** Traditional image forgery detection

**Images:**

- Real: **7,491**
- Tampered: **5,123**
- Total: **12,614**

**Forgery types:**

- Copy-move
- Splicing

The tampered portion contains both copy-move and splicing manipulations.

**Source:**

[CASIA Tampered Image Detection Evaluation Database](http://forensics.idealtest.org/)

**Notes:**

CASIA v2 is one of the most widely used benchmark datasets for image tampering detection. Different papers may construct different train/test splits from the released images, so this catalog reports the overall dataset size rather than assigning a universal train/test split.

---

## IMD2020

**Dataset type:** Real-world image manipulation dataset

**Images:**

- Real: **414**
- Tampered: **2,010**
- Total: **2,424**

**Forgery types:**

- Image splicing
- Copy-move
- Object removal
- Other real-world manipulations

**Source:**

[IMD2020](https://staff.utia.cas.cz/novozada/imd2020/)

**Notes:**

IMD2020 focuses on realistic image manipulations and is useful for evaluating general-purpose image manipulation detection systems.

---

## FantasticReality

**Dataset type:** Large-scale image forgery dataset

**Images:**

- Real: **16,592**
- Tampered: **19,423**
- Total: **36,015**

**Forgery type:**

- Splicing

The dataset contains semantically diverse manipulated images and was designed to provide more realistic manipulation scenarios.

**Source:**

[FantasticReality](https://github.com/liyidong86/FantasticReality)

---

## tampCOCO

**Dataset type:** Large-scale synthetic image tampering dataset

**Images:**

- Tampered: **799,441***
- Real: **0**
- Total: **799,441***

**Forgery types:**

- Splicing
- Copy-move

tampCOCO is constructed using imagery from the COCO dataset and provides a large number of automatically generated manipulated images.

**Source:**

[CAT-Net / tampCOCO](https://github.com/mjkwon2021/CAT-Net)

\* Published descriptions may round the dataset size to approximately **800K**. The catalog uses **799,441** for the specific version/count being tracked here.

**Important:**

Because this dataset is generated from source imagery, the tampered images should not be interpreted as independently captured real-world photographs.

---

## compRAISE

**Dataset type:** Pristine/source image dataset for image forensics

**Images:**

- Real/source: **24,462**
- Tampered: **0**
- Total: **24,462**

**Role:**

- Pristine images
- Compression-related forensic analysis
- Source imagery for tampering/compression experiments

**Source:**

[compRAISE](https://github.com/photometric-stereo/compRAISE)

**Notes:**

compRAISE is included because pristine/source images are useful for training and evaluating image forensic systems. It should not be treated as a standalone tampered-image dataset.

---

## MISD

**Dataset type:** Multiple-image splicing dataset

**Images:**

- Real: **618**
- Tampered: **300**
- Total: **918**

**Forgery type:**

- Multiple-image splicing

MISD contains images created by compositing content from multiple source images.

**Source:**

[MISD — Multiple Image Splicing Dataset](https://www.mdpi.com/2306-5729/6/10/102)

---

## Columbia

**Dataset type:** Image splicing dataset

**Images:**

- Real: **183**
- Tampered: **180**
- Total: **363**

**Forgery type:**

- Splicing

The Columbia dataset contains uncompressed TIFF images and is a commonly used benchmark for image splicing detection.

**Source:**

[Columbia Image Splicing Detection Evaluation Dataset](https://www.ee.columbia.edu/ln/dvmm/downloads/AuthSplicedDataSet/dlform.html)

**Notes:**

Individual papers may use different subsets or train/test partitions. The counts above refer to the overall dataset.

---

## In-the-Wild

**Dataset type:** Real-world image splicing dataset

**Images:**

- Real/source: **0**
- Tampered: **201**
- Total: **201**

**Forgery type:**

- Real-world splicing

The dataset contains manipulated images collected from real-world online sources rather than being generated entirely through a controlled synthetic pipeline.

**Source:**

[In-the-Wild Image Forgery Dataset](https://github.com/RU-System-Forensics/MediaForensics)

**Reference:**

Huh et al., *Fighting Fake News: Image Splice Detection via Learned Self-Consistency*, ECCV 2018.

---

## Realistic Tampering (Korus)

**Dataset type:** Realistic image tampering dataset

**Images:**

- Real: **220**
- Tampered: **220**
- Total: **440**

**Forgery types:**

- Object insertion
- Object removal
- Splicing
- Copy-move

The dataset provides realistic tampering examples together with corresponding original images.

**Source:**

[Korus — Realistic Tampering Dataset](https://www.pkorus.pl/downloads/dataset-realistic-tampering)

---

## CASIA v1

**Dataset type:** Traditional image forgery dataset

**Images:**

- Real: **800**
- Tampered: **920†**
- Total: **1,720†**

**Forgery types:**

- Copy-move
- Splicing

A commonly reported breakdown is:

- Copy-move: **459**
- Splicing: **461**

**Source:**

[CASIA Tampered Image Detection Evaluation Database](http://forensics.idealtest.org/)

† Some sources report **921** tampered images and therefore a total of **1,721**. This catalog uses 920 because it is the count reported by several commonly used image-forensics dataset lists.

---

## COCO 2017

**Dataset type:** Source imagery dataset

**Images:**

- Train: **118,287**
- Validation: **5,000**
- Train + validation: **123,287**
- Tampered: **0**

**Role:**

COCO 2017 is not itself an image-forgery dataset. It is included because its images are used as source imagery for constructing synthetic tampering datasets, including datasets such as tampCOCO and other COCO-based manipulation benchmarks.

**Source:**

[COCO Dataset](https://cocodataset.org/)

**Important:**

The standard COCO 2017 train + validation count is **123,287**. The COCO test set is separate and should not be added to this figure when reporting the labeled train/validation source collection.

---

## LAION-Mobile

**Dataset type:** Pristine/source image dataset for image forensics

**Images:**

- Real/source: **822,296**
- Tampered: **0**
- Total: **822,296**

**Role:**

- Large-scale pristine/natural image collection
- Real-image source for training image-forgery detection systems
- Useful for balancing the real-image side of forgery datasets

**Dataset metadata:**

- Dataset: **LAION-Mobile**
- Phase: **Phase1**
- Domain: **General**
- Category: **NaturalImages**
- All records are labeled **Real**

**Source:**

[LAION-Mobile streaming dataset](https://huggingface.co/datasets/sumathiselvan/LAION-Mobile-streaming)

**Notes:**

LAION-Mobile is a pristine/source-image collection rather than a standalone tampered-image dataset. The streaming release contains **822,296 image records**, all labeled **Real**. The dataset consists of JPEG/JPG images and can be used as an additional real-image source when constructing a balanced training set.

---


## CocoGlide

**Dataset type:** AI-generated image manipulation dataset

**Images:**

- Real: **512**
- Tampered: **512**
- Total: **1,024**

**Forgery type:**

- AI-based object replacement
- Inpainting / generative image editing

CocoGlide contains manipulated images generated using the GLIDE generative model on COCO imagery.

**Source:**

[CocoGlide](https://github.com/ymhzyj/DEAL-300K)

**Notes:**

Unlike traditional copy-move and splicing datasets, CocoGlide represents a newer class of image manipulation in which generative models are used to modify image content. It is a fixed evaluation benchmark (512 real + 512 tampered) and is used entirely as a **test** set — it does not contribute Train images.

---

## DEFACTO

**Dataset type:** Large-scale synthetic image forgery dataset

**Images:**

- Real/source: **0**
- Tampered: **149,000**
- Total: **149,000**

**Forgery types:**

- Splicing: **105,000**
- Copy-move: **19,000**
- Removal / inpainting: **25,000**

**Source:**

[DEFACTO](https://github.com/namhocho/DEFACTO)

**Notes:**

DEFACTO provides a large-scale collection of automatically generated image manipulations covering several major forgery categories.

---

# Source Datasets

Some datasets in this catalog are primarily source-image collections rather than forgery datasets.

| Dataset | Role |
|---|---|
| **COCO 2017** | Source imagery used to construct synthetic manipulation datasets |
| **compRAISE** | Pristine/source images for image-forensic and compression-related experiments |
| **LAION-Mobile** | Large-scale pristine/natural images for real-image training and forensic experiments |

These datasets should therefore not be interpreted as containing the same type of tampering labels as CASIA, MISD, Columbia, or DEFACTO.

---

# Dataset Selection by Forgery Type

| Forgery Type | Representative Datasets |
|---|---|
| Copy-move | CASIA v1, CASIA v2, tampCOCO, DEFACTO |
| Splicing | CASIA v1, CASIA v2, FantasticReality, MISD, Columbia, In-the-Wild, tampCOCO, DEFACTO |
| Object insertion | Realistic Tampering (Korus), DEFACTO |
| Object removal | IMD2020, Realistic Tampering (Korus), DEFACTO |
| Inpainting / generative replacement | CocoGlide, DEFACTO |
| Compression / forensic artifacts | compRAISE |
| Source imagery | COCO 2017, LAION-Mobile |

---

# Notes and Limitations

1. **Dataset counts may differ between sources.** Different papers and repositories may use different versions, filtering procedures, or subsets of the same dataset.

2. **CASIA v1 has a reported count discrepancy.** Some sources report 920 tampered images while others report 921. This catalog uses 920 and explicitly records the discrepancy.

3. **tampCOCO counts may be reported approximately.** Published descriptions commonly refer to the dataset as approximately 800K images. The cataloged version used here contains 799,441 images.

4. **Train/test splits are not necessarily official.** Many image-forensics papers create their own splits. Such experimental splits should not be confused with the total dataset size. CocoGlide in particular is treated here as a fixed benchmark that sits entirely in Test.

5. **Synthetic versus real-world manipulations should be distinguished.** Datasets such as tampCOCO and DEFACTO contain automatically generated manipulations, whereas datasets such as In-the-Wild and Realistic Tampering contain more realistic or naturally occurring manipulation scenarios.

6. **COCO 2017 is a source dataset, not a forgery dataset.** It is included because COCO imagery is used to generate several synthetic image-tampering datasets.

7. **compRAISE is primarily a pristine/source dataset.** It should not be counted as a collection of tampered images.

8. **Licensing and redistribution terms vary by dataset.** Users should consult each dataset's official source before downloading or redistributing the data.

---

# References

- CASIA Tampered Image Detection Evaluation Database
- IMD2020
- FantasticReality
- tampCOCO / CAT-Net
- compRAISE
- MISD
- Columbia Image Splicing Detection Evaluation Dataset
- In-the-Wild
- Realistic Tampering Dataset
- COCO 2017
- CocoGlide
- DEFACTO
- LAION-Mobile
