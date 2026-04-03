# Dataset Links for Multimodal Biometric Fusion

This project expects the following datasets (as described in `README.md` and `scripts/prepare_dataset.py`):

1. **CASIA-IrisV4-Interval** (iris modality)
2. **CASIA-FingerprintV5** (fingerprint modality)
3. **WebFace260M (subset)** (face modality)

---

## Required Datasets (Official / Primary Links)

### 1) CASIA-IrisV4 (includes Interval subset)
- Official page: https://hycasia.github.io/dataset/casia-irisv4/

### 2) CASIA-FingerprintV5
- Dataset overview: http://english.ia.cas.cn/db/201611/t20161101_169922.html
- CASIA biometrics portal: http://biometrics.idealtest.org/findTotalDbByMode.do?mode=Fingerprint

### 3) WebFace260M / WebFace42M
- Benchmark site: https://www.face-benchmark.org
- Paper: https://arxiv.org/abs/2204.10149
- ICCV challenge repo/info: https://github.com/WebFace260M/webface260m-iccv21-mfr

---

## Alternatives by Modality

### Iris alternatives
- Other CASIA-IrisV4 subsets (Lamp, Twins, Thousand, Distance, Syn):  
  https://hycasia.github.io/dataset/casia-irisv4/
- Curated iris dataset index:  
  https://iapr-tc4.org/iris-datasets/

### Fingerprint alternatives
- FVC2002: https://bias.csr.unibo.it/fvc2002/
- FVC2004: https://bias.csr.unibo.it/fvc2004/
- SOCOFing (academic use): https://www.kaggle.com/datasets/ruizgara/socofing
- Curated fingerprint dataset index: https://iapr-tc4.org/fingerprint-datasets/

### Face alternatives
- VGGFace2 (official page): https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/
- CASIA-WebFace (official CASIA page): http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html
- CelebA: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- LFW (mostly for evaluation): http://vis-www.cs.umass.edu/lfw/
- Curated face dataset index: https://iapr-tc4.org/face-datasets/

---

## Practical Compatibility Notes for This Project

The script `scripts/prepare_dataset.py` builds a **virtual homologous multimodal dataset** by pairing identities across modalities.  
Any alternative datasets should provide:

- sufficient number of identities,
- at least `min_samples` images per identity,
- a directory or filename structure that the script can parse.

Supported identity layouts in the script:
- `root/{identity_id}/{img}.jpg`
- `root/{img_with_id_prefix}.jpg` (prefix-based grouping)

---

## Local Note

A torrent file is already present in the repository root:

- `article/VGG-Face2-535113b8395832f09121bc53ac85d7bc8ef6fa5b.torrent`

You can use this as a face-data source if licensing and institutional policy allow it.