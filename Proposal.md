# Bachelor Thesis Extensions — CT-to-PET Translation on autoPET

Four candidate directions building on the completed practical-work reimplementation of CPDM on the autoPET dataset. Each is framed as a standalone thesis proposal with a central research question, methodology, expected chapter structure, and an honest risk assessment.

## Existing foundation

The practical work delivered a working CPU-only end-to-end pipeline: preprocessing → attention-map UNet → bulk attention-map export → VQGAN → CPDM (Brownian Bridge diffusion in latent space). Final metrics on the autoPET val set: best `val_epoch/loss = 0.01394` at epoch 7, best LPIPS = 0.166, MAE = 0.0053, SSIM = 0.89, PSNR = 41.7 dB. A baseline comparison confirmed that CPDM beats trivial "predict-background" and "predict-mean-image" baselines on PSNR by 2–7 dB. Full protocol in `NOTES.md`; reproducible from `report.py`.

Any of the proposals below reuses this infrastructure (config + small-script change rather than ground-up reimplementation).

---

## Proposal A — Component-wise contribution and failure mode analysis

**Research question.** Which of CPDM's design choices contribute meaningfully to CT-to-PET translation quality on autoPET, and where do they systematically fail?

**Why it matters.** The original CPDM ablation (their Table 4) was conducted on a curated brain dataset. Whether the relative importance of attention map vs. attenuation map vs. domain knowledge transfers to autoPET — a noisier, multi-center, clinical-grade dataset — is an open question. If it does not transfer, that is evidence the paper's design choices are dataset-specific rather than universal.

**Approach.** Four training configurations under matched compute budget: (1) full CPDM, (2) no attention map, (3) no attenuation map, (4) neither. Report LPIPS / MAE / SSIM / PSNR. Then add a failure-mode taxonomy: cluster val slices by characteristics (background fraction, hotspot presence, anatomical region) and compute per-cluster metrics to identify what makes a slice hard.

**Chapter sketch.** (1) Background — diffusion models, BB, medical synthesis. (2) Method — CPDM and the reimplementation. (3) Ablation experiments. (4) Failure mode taxonomy. (5) Discussion.

**Risk.** Low. All infrastructure already exists; runs are short on CPU.

**Deliverable shape.** Coherent ablation table + failure-mode taxonomy. Workshop-paper-shaped.

---

## Proposal B — Clinical utility: a downstream-task evaluation framework

**Research question.** Does synthetic PET from CPDM carry enough diagnostic signal to be useful for the clinical task it would replace?

**Why it matters.** Image-quality metrics (LPIPS, PSNR, SSIM) measure visual similarity, not clinical utility. A model that produces beautiful but diagnostically-blurred PET is worse than one that produces unattractive but informative PET. autoPET is itself a tumor-segmentation challenge, so the task that PET is clinically used for is well-defined and has labels.

**Approach.** Train three tumor segmenters on autoPET labels: (a) CT + real PET (oracle), (b) CT + synthetic PET from CPDM, (c) CT only. Compare Dice / IoU / lesion-detection sensitivity. Optionally include a fourth condition using the paper's released checkpoint if available.

**Chapter sketch.** (1) Background. (2) CPDM reproduction. (3) Downstream task setup and segmenter design. (4) Comparative evaluation. (5) Where synthetic PET helps and where it does not.

**Risk.** Medium. The segmenter pipeline is real engineering but well-supported by libraries (MONAI, segmentation_models_pytorch). GPU access likely needed for the segmenter training.

**Deliverable shape.** Strongest publication shape on this list — directly relevant to clinical adoption.

---

## Proposal C — Beyond Brownian Bridges: a paradigm comparison

**Research question.** Are BB diffusion, vanilla DDPM, EDM, flow matching, and consistency models substitutable for cross-modality medical synthesis, or do their inductive biases lead to systematically different behavior?

**Why it matters.** The CPDM paper picked BB diffusion for principled reasons (it natively models source-to-target translation). The diffusion field has moved fast since — EDM and flow matching are now considered stronger baselines in general image generation. Whether that holds for medical cross-modality is unclear.

**Approach.** Reimplement the same autoPET CT-to-PET task with 3-4 different diffusion paradigms, holding everything else (VQGAN, attention map, attenuation map, conditioning, eval) constant. Compare quality at matched training compute *and* at matched inference speed.

**Chapter sketch.** (1) Background — survey of diffusion paradigms. (2) Adaptation of each paradigm to CT-to-PET. (3) Quality comparison. (4) Inference-speed comparison. (5) Recommendation.

**Risk.** Medium-high. Implementation-heavy. Each paradigm has its own training subtleties. Best fit for a methods-focused supervisor.

**Deliverable shape.** Method-paper-shaped.

---

## Proposal D — Domain generalization across scanners and centers

**Research question.** Does a CPDM trained on data from one center / scanner produce reliable PET when applied to data from another, or does scanner-domain shift break it?

**Why it matters.** Any clinical synthesis tool must be robust to the diversity of imaging hardware encountered in real deployment. autoPET aggregates studies from multiple centers and scanner vendors, making this experiment design clean.

**Approach.** Partition autoPET by source center using TCIA metadata. Train CPDM on one cohort, test on others. Quantify the domain gap per metric and per anatomical region. Then test simple mitigation strategies (instance normalization, scanner-specific batch norm, mild domain-adversarial training).

**Chapter sketch.** (1) Background — domain shift in medical imaging. (2) Multi-center autoPET analysis. (3) Cross-center experiments. (4) Mitigation methods. (5) Recommendations for clinical deployment.

**Risk.** Low-medium. Hardest part is reliably partitioning autoPET by center from metadata. CPU-trainable per-cohort given smaller sub-cohorts.

**Deliverable shape.** Clinically-relevant; fits well in a "robustness for medical AI" workshop.

---

## How to choose

Three questions for the supervisor meeting:

1. **What is the lab's research focus?** Medical AI broadly → B. Methods / diffusion models → A or C. Clinical translation / deployment → D.
2. **What compute is realistically available?** CPU only → A safest, C feasible (inference dominates), D fine with small subsets. GPU access → B opens up.
3. **What do you want to learn?** A sharpens diffusion-model understanding; B teaches downstream-task engineering; C broadens diffusion-paradigm literacy; D builds clinical-ML intuition.

**Default safe bet:** Proposal **A** with Proposal **D** as a stretch chapter. Combined title: *"Reproducing and stress-testing CPDM: a component-wise and cross-cohort analysis of CT-to-PET diffusion on autoPET."* Every chapter has well-defined experiments, the infrastructure exists, and the scope fits a bachelor's thesis.

**Maximum-impact bet (with GPU):** Proposal **B**. Most likely to matter to a clinician reading the thesis.
