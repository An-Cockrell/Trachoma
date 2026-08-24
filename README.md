# Trachoma — Automated TF Grading from Conjunctival Photographs

Research code for automated detection of **trachomatous inflammation—follicular (TF)**
from photographs of the everted upper tarsal conjunctiva.

The repository holds several years of work along two lines: a **whole-image binary
classifier** (TF vs. No-TF), which is the subject of the accompanying paper, and an
exploratory **follicle-level detection and counting** pipeline, which is ongoing and
not reported in the paper.

---

## Start here

> **The code and evaluation artifacts for the paper are in
> [`blackbox/trachoma_resnet/`](blackbox/trachoma_resnet/).**
>
> That directory has its own README covering the model, the two cross-validation
> experiments, the full hyperparameter configuration, and — importantly — how to
> **recompute every reported metric without access to the image data**.

Nothing outside `blackbox/trachoma_resnet/` is part of the paper. The rest of this
repository is exploratory and historical work, kept for provenance. If you are
reviewing the manuscript, you can stop at that directory.

---

## Data availability

**The image data is not in this repository and cannot be redistributed.** The
photographs are governed by data-use agreements with the contributing studies
(SOCIT, TANA II, Gambia PRET, ICAPS, Solomon Islands, CC_EA2017, 2022 Australia
Trachoma Images, and Kim et al.). Source photographs, the master label file, model
checkpoints, training logs, and Grad-CAM exemplar images (which are themselves
patient photographs) are all excluded.

Requests for data access should be directed to the corresponding author and routed
through the originating studies.

This does **not** block verification of the results. Every metric, threshold, and
confidence interval in the paper can be recomputed from committed files with no
image access — see
[Verifying the reported results](blackbox/trachoma_resnet/README.md#verifying-the-reported-results).

---

## Repository layout

| Path | What it is | Paper? |
|---|---|---|
| [`blackbox/trachoma_resnet/`](blackbox/trachoma_resnet/) | **ResNet-50 TF classifier.** Training, 5-fold grouped CV and leave-one-source-out CV, evaluation artifacts, and the paper's figures and tables. | **yes** |
| `annotated_data/FollicleDetection/` | Follicle-level segmentation and counting (U-Net over SAM-derived pseudo-labels). Separate, ongoing line of work. | no |
| `annotated_data/` (other) | Gradable-area / eyelid-boundary segmentation experiments. | no |
| `annotation_tool.py` | Standalone browser tool for hand-drawing follicle masks. Run it and open `http://localhost:5000`. | no |

---

## Citation

If you use this code, please cite the accompanying paper.

<!-- TODO before submission: replace with the full citation, and pin a tagged
     release or commit SHA here so readers land on exactly the version the
     paper describes. -->
