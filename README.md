# Dual-network fusion framework

This repository contains an SSD/YOLO fire-detection workspace and several
box-level fusion/evaluation scripts. The current release is best understood as
an **artifact snapshot**: prediction files, evaluation annotations, source
code, and many checkpoints are present, but not every training experiment is
described well enough to be reproduced from a clean machine.

## Release status

The table below distinguishes files that are actually included from items that
are only represented by code, configuration, or prose.

| Artifact | Status in this repository | Notes |
| --- | --- | --- |
| Fusion source code | Released | `Main/ensemble_boxes/` contains WBF, NMS, Soft-NMS, NMW, DIoU-NMS, and CIoU-NMS implementations. |
| COCO fusion benchmark inputs | Released through Git LFS | `Main/benchmark_coco/*.csv` and `instances_val2017.json` are present. Run `git lfs pull` after cloning. |
| Fire fusion inputs and evaluation annotation | Released | `Main/benchmark_fire/redataset_last/*.csv` and `Main/benchmark_fire/WBF.json` are present. `WBF.json` describes 800 images, 639 annotations, and one category. |
| YOLO-format fire images and labels | Released as a flat snapshot | `NewFire/` contains 3,997 JPG files and 3,997 matching TXT label files. No authoritative train/validation/test split or dataset license is currently supplied. |
| SSD/VOC-format data | Partially released | `SSD/VOCdevkit/VOC2007/` contains 2,059 images and 2,059 XML annotations plus split files. Other image/annotation copies also exist under `SSD/`; their relationship to the paper's final split is not documented. |
| SSD checkpoints | Released, with caveats | Checkpoints are present under `SSD/model_data/` and `SSD/best_weight/logs/`. Some files under `SSD/best_weight/logs/` are zero bytes and must not be treated as usable checkpoints. |
| YOLO checkpoints | Released, with caveats | Multiple `.pt` files are present under `YOLO/`. The repository does not currently identify one canonical paper checkpoint or provide checksums. |
| Exact detector training provenance | Incomplete | SSD hyperparameters and seed are in `SSD/train.py`; an exact paper-level YOLO command, split, hardware record, and run-to-checkpoint mapping are not supplied. |
| Confidence intervals / repeated-run variance | Not released | Existing benchmark scripts report point estimates only. |

Consequently, the repository permits direct inspection and reproduction of
several **fusion/evaluation calculations**, but it does not yet support an
unqualified claim that every detector-training result in a paper can be
reproduced end to end.

## Obtaining the files

Clone with Git LFS enabled:

```bash
git lfs install
git clone <repository-url>
cd "dual-network fusion framework"
git lfs pull
```

There is no separate, versioned dataset download archive in this release.
The data visible in `NewFire/` and `SSD/VOCdevkit/` can be obtained by cloning
the repository, subject to the unresolved dataset licensing issue below.

## Environment

The recommended reproducibility environment is defined in
[`environment.yml`](environment.yml):

```bash
conda env create -f environment.yml
conda activate dual-network-fusion
python -m pip install -e Main
```

The environment targets Python 3.10. GPU-enabled PyTorch installation depends
on the host CUDA version; replace the CPU PyTorch packages after environment
creation when GPU training is required.

The historical pins in `SSD/requirements.txt` are retained for provenance but
are too old for many current Python installations. They should not be treated
as the primary clean-install specification.

## Licensing

Original code and modifications authored for this repository are licensed
under **GNU AGPL-3.0-only**; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
Existing third-party components retain their original licenses:

| Component | License |
| --- | --- |
| Original repository code and modifications, unless stated otherwise | AGPL-3.0-only (`LICENSE`) |
| `Main/` fusion code | MIT (`Main/LICENSE`) |
| `SSD/` code | MIT (`SSD/LICENSE`) |
| `YOLO/` vendored Ultralytics code | GNU AGPL-3.0 (`YOLO/LICENSE`) |
| `NewFire/`, VOC/COCO-derived data, annotations, prediction files, and trained weights | Not explicitly licensed by this repository |

The root AGPL-3.0 declaration does not relicense MIT components or third-party
material, and it does not grant permission to redistribute images,
annotations, model weights, or third-party benchmark predictions. The authors
should add provenance and explicit usage terms for every dataset and weight
release. Until then, downstream redistribution and commercial use of those
artifacts cannot be considered clearly authorized by this repository.
