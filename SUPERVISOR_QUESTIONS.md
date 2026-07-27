# Questions for the supervisor

Open questions to raise in the next meeting. **Answers not yet received** — record the
supervisor's response inline under each item as we get them, then act on it.

Status legend: 🟡 open · 🟢 answered · ⚪ decided/closed

---

## 1. Scope & framing of the comparison

### 1.1 The 2×2 is confounded — is that acceptable? 🟡
The intended axis is *paradigm-native vs. concat conditioning* within each of {diffusion, flow}.
But in the current setup **both diffusion models run in the VQGAN latent** (CPDM, concat-diff)
and **both flow models run in pixel space** (PMRF, concat-flow). So a diffusion-vs-flow gap is
confounded with a latent-vs-pixel gap; only *within-family* claims (PMRF vs concat-flow; CPDM
vs concat-diff) are clean.
- **Ask:** Is the within-family framing enough for the thesis, or do you want a deconfounding
  cell (e.g. a latent-space flow, or a pixel-space diffusion) added?
- **Answer:**

### 1.2 Is the dropped RF baseline a gap? 🟡
The paper's rectified-flow baseline (`--mode rf`, z0 = CT + noise) is implemented but dropped
from thesis scope in favour of concat-flow.
- **Ask:** Reinstate it for completeness / faithfulness to PMRF, or is it fine to note it as
  implemented-but-out-of-scope?
- **Answer:**

---

## 2. The attention-map fix (CPDM)

### 2.1 How to present the blob→focal change? 🟡
The original attention target was a preprocessing bug (75th percentile over the *whole*
background-cropped slice ≈ a whole-body blob, zero localization). Fixed to an absolute SUV>2.0
focal target; retrained the U-Net; fine-tuned CPDM → FID 87.1→79.7.
- **Ask:** Present this as a *methodological contribution* (diagnosing why the practical-work
  CPDM produced blurry clouds), or downplay it as a bugfix in an appendix? Report both blob
  and focal CPDM rows, or only focal?
- **Answer:**

### 2.2 Is CT-predictable attention the right prior at all? 🟡
Focal attention localizes *physiological* uptake (organs at predictable positions) but CT
fundamentally cannot predict *pathological* uptake (tumors). So the prior helps anatomy, not
disease.
- **Ask:** Worth stating as a fundamental limitation of the CPDM conditioning idea for CT→PET?
- **Answer:**

---

## 3. Evaluation rigor

### 3.1 Test set is only 2 patients (256 slices). 🟡
The shared comparison test split is small (2 patients, 170+86 slices). FID/KID on ~256 images
is noisy.
- **Ask:** Is this a critical weakness for the thesis grade? Should I expand the test set
  (more patients from autoPET) before submission, accepting extra preprocessing/eval time?
- **Answer:**

### 3.2 Are the masked metrics (active/lesion ROI) convincing? 🟡
PET is ~89% near-zero, so global MAE/PSNR/SSIM are dominated by background. We report global +
active-SUV>0.5 + lesion-SUV>2.5 tiers.
- **Ask:** Is the ROI-tiered reporting the right honest choice, or do you want a specific
  clinical metric (e.g. SUVmax error, lesion detectability)?
- **Answer:**

---

## 4. Thesis structure & depth

### 4.1 Literature depth. 🟡
Added a Related Work section (~1 page, 3 threads) and expanded the perception–distortion theory
with rate-distortion-perception + Wasserstein D-P references.
- **Ask:** Is the current depth appropriate for a bachelor thesis, or do you want more breadth
  (e.g. more CT→PET / cross-modality synthesis prior work)?
- **Answer:**

### 4.2 How much to foreground the CPU-only constraint? 🟡
All training was single-CPU (i7-8650U), which capped model size, dataset size, and resolution
(128² was dropped).
- **Ask:** Present CPU-only as an explicit scope/limitation section, or footnote it? Does it
  undermine the comparison's validity in your view?
- **Answer:**

### 4.3 Hypotheses framing. 🟡
Reframed the thesis around RQ1–3 / H1–5 with a "hypotheses revisited" verdict table. Note H2
is a genuine *mixed* result (paradigm-native beats concat for flow, but concat-diff beats CPDM).
- **Ask:** Is presenting a partially-refuted hypothesis as a finding the right scientific tone,
  or would you prefer it reframed?
- **Answer:**

---

## 5. Next-step / effort allocation

### 5.1 Where is more compute best spent? 🟡
Options: (a) continue-training blob CPDM to real convergence [in progress]; (b) expand the test
set; (c) add a deconfounding model; (d) longer PMRF/concat runs.
- **Ask:** Given the remaining time budget, which single improvement would most strengthen the
  thesis?
- **Answer:**
